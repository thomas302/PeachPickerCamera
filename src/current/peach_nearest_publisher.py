#!/usr/bin/env python3

# ── Imports ─────────────────────────────────────────────────────────────────────
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

# Helper: only use OpenCV GUI if DISPLAY is available
_DISPLAY_AVAILABLE = bool(os.environ.get("DISPLAY"))

# Reuse the full threaded Darknet+DepthAI pipeline from camera_threaded.py
import camera_threaded as ct


# ── ROS 2 Node: NearestPeachPublisher ────────────────────────────────────────────
# Publishes the nearest detected peach in both camera and world frames.
# Topics:
#   /peach/nearest/camera  -> geometry_msgs/PointStamped (frame_id: "camera")
#   /peach/nearest/world   -> geometry_msgs/PointStamped (frame_id: "world")
class NearestPeachPublisher(Node):
    def __init__(self):
        super().__init__("nearest_peach_publisher")

        # ROS 2 publishers
        self._pub_cam = self.create_publisher(PointStamped, "/peach/nearest/camera", 10)
        self._pub_world = self.create_publisher(PointStamped, "/peach/nearest/world", 10)

        # Camera + detection manager from camera_threaded
        self._mgr = ct.Manager()
        self._mgr.start()

        # Shared atomic slots for thread-safe data passing
        self._frame_slot = ct.AtomicSlot()          # camera thread -> (rgb, disparity)
        self._detection_slot = ct.AtomicSlot()      # inference thread -> (locations, infer_ms)
        self._stop_event = threading.Event()

        # Start background threads
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._camera_thread.start()
        self._infer_thread.start()

        # State for display loop
        self._current_locations = []
        self._current_infer_ms = 0.0
        self._last_det_seq = 0

        # ROS 2 timer for UI + publishing (20 Hz)
        self._timer = self.create_timer(0.05, self._tick)

    # ── Threading pipeline ───────────────────────────────────────────────────────────
# Camera thread: grabs latest synchronized RGB + disparity and pushes into a slot.
    def _camera_loop(self) -> None:
        while not self._stop_event.is_set():
            result = self._mgr.cam.get_synced()
            if result is None:
                time.sleep(0.002)
                continue
            rgb, disparity = result
            self._frame_slot.put(rgb, disparity)

# Inference thread: pulls new frames, runs detection + depth + pose, publishes results.
    def _inference_loop(self) -> None:
        last_seq = 0
        while not self._stop_event.is_set():
            snapshot, last_seq = self._frame_slot.get_if_new(last_seq)
            if snapshot is None:
                time.sleep(0.001)
                continue

            rgb, disparity = snapshot

            t0 = time.monotonic()
            detections = self._mgr.detector.find_objects(rgb)
            infer_ms = (time.monotonic() - t0) * 1000.0

            locations = []
            for label, conf, bbox in detections:
                depth_m = self._mgr.cam.get_depth_in_bbox(
                    disparity, bbox, self._mgr.focal_length_px, self._mgr.baseline_m
                )

                if depth_m <= 0:
                    pos_cam = np.zeros(3)
                    pos_world = None
                else:
                    pos_cam = ct.bbox_depth_to_camera_frame(
                        bbox, depth_m, self._mgr.fx, self._mgr.fy, self._mgr.cx, self._mgr.cy
                    )
                    pos_world = ct.camera_to_world_frame(
                        pos_cam,
                        yaw_deg=0.0,
                        pitch_deg=0.0,
                        roll_deg=0.0,
                        camera_world_pos=np.array([5.5/100, 5.0/100, 0.0]),
                    )

                locations.append(
                    ct.ObjectLocation(
                        label=label,
                        conf=float(conf),
                        bbox=bbox,
                        depth_m=float(depth_m),
                        pos_camera=pos_cam,
                        pos_world=pos_world,
                    )
                )

            self._detection_slot.put(locations, infer_ms)

# ── Helpers ─────────────────────────────────────────────────────────────────────
# Select the nearest peach by depth (smallest depth_m > 0).
    def _select_nearest_peach(
        self, locations
    ) -> Optional[ct.ObjectLocation]:
        peaches = [
            loc
            for loc in locations
            if loc.label == "peach" and loc.depth_m is not None and loc.depth_m > 0
        ]
        if not peaches:
            return None

        nearest = min(peaches, key=lambda l: l.depth_m)
        return nearest

# Publish a 3D point as geometry_msgs/PointStamped.
    def _publish_point(self, pub, frame_id: str, xyz: np.ndarray) -> None:
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.point.x = float(xyz[0])
        msg.point.y = float(xyz[1])
        msg.point.z = float(xyz[2])
        pub.publish(msg)

    # ── Main UI/publish loop (ROS 2 timer callback) ─────────────────────────────────────
# Runs at 20 Hz. Pulls latest detections, publishes nearest peach, and shows the frame.
    def _tick(self) -> None:
        # Update latest detections (non-blocking)
        det_snapshot, self._last_det_seq = self._detection_slot.get_if_new(self._last_det_seq)
        if det_snapshot is not None:
            self._current_locations, self._current_infer_ms = det_snapshot

        # Publish nearest peach (if any)
        nearest = self._select_nearest_peach(self._current_locations)
        if nearest is not None:
            self._publish_point(self._pub_cam, "camera", nearest.pos_camera)
            if nearest.pos_world is not None:
                self._publish_point(self._pub_world, "world", nearest.pos_world)

        # Display latest frame with overlays
        frame_data = self._frame_slot.get_latest()
        if frame_data is not None:
            rgb, _ = frame_data
            display_frame = rgb.copy()
            ct.draw_locations(display_frame, self._current_locations, self._current_infer_ms)
            if nearest is not None:
                x1, y1, x2, y2 = nearest.bbox
                cv2.rectangle(
                    display_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 255, 255),
                    2,
                )
                cam = nearest.pos_camera
                world = nearest.pos_world
                cv2.putText(
                    display_frame,
                    f"nearest cam=({cam[0]:+.3f},{cam[1]:+.3f},{cam[2]:+.3f})m",
                    (20, ct.DISPLAY_H - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                if world is not None:
                    cv2.putText(
                        display_frame,
                        f"nearest world=({world[0]:+.3f},{world[1]:+.3f},{world[2]:+.3f})m",
                        (20, ct.DISPLAY_H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            if _DISPLAY_AVAILABLE:
                cv2.imshow("Nearest Peach Publisher", display_frame)
                if cv2.waitKey(1) == ord("q"):
                    self.get_logger().info("q pressed; shutting down")
                    rclpy.shutdown()

    # ── Cleanup ───────────────────────────────────────────────────────────────────────
    def destroy_node(self):
        self._stop_event.set()
        if hasattr(self, "_camera_thread"):
            self._camera_thread.join(timeout=5)
        if hasattr(self, "_infer_thread"):
            self._infer_thread.join(timeout=5)
        if _DISPLAY_AVAILABLE:
            cv2.destroyAllWindows()
        super().destroy_node()


# ── Entry point ───────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = NearestPeachPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
