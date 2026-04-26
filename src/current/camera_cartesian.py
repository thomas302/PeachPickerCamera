#!/usr/bin/env python3
import sys
import time
import numpy as np
import cv2
sys.path.append('/home/peach/src/darknet/src-python')
import darknet
import depthai as dai
from datetime import timedelta
from dataclasses import dataclass

cfg_file     = "new_sample.cfg"
names_file   = "new_sample.names"
weights_file = "new_sample.weights"

DISPLAY_W, DISPLAY_H = 1280, 720
OVERLAP      = 64
NMS_IOU      = 0.3
CENTER_THRESH = 40
THRESH       = 0.4


# ── Darknet globals ───────────────────────────────────────────────────────────

network     = darknet.load_net_custom(cfg_file.encode("ascii"), weights_file.encode("ascii"), 0, 1)
class_names = open(names_file).read().splitlines()
colours     = darknet.class_colors(class_names)
NET_W       = darknet.network_width(network)
NET_H       = darknet.network_height(network)


# ── Localization ──────────────────────────────────────────────────────────────

@dataclass
class ObjectLocation:
    """3D position of a detected object."""
    label:   str
    conf:    float
    bbox:    tuple           # (x1, y1, x2, y2) in pixels
    depth_m: float           # planar depth from camera
    pos_camera: np.ndarray   # [x, y, z] in camera frame (metres)
    pos_world:  np.ndarray   # [x, y, z] in world frame  (metres), or None


def bbox_depth_to_camera_frame(
    bbox: tuple,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Convert bbox center + planar depth to a 3D point in camera frame.

    Camera frame convention:
        +X = right,  +Y = down,  +Z = forward

    Planar depth D is the perpendicular distance to the depth plane —
    the real-world offsets are D*tan(θ), not D*sin(θ).
    """
    x1, y1, x2, y2 = bbox
    px = (x1 + x2) / 2.0
    py = (y1 + y2) / 2.0

    # Angles from optical axis
    theta_x = np.arctan2(px - cx, fx)
    theta_y = np.arctan2(py - cy, fy)

    return np.array([
        depth_m * np.tan(theta_x),   # lateral offset (right)
        depth_m * np.tan(theta_y),   # vertical offset (down)
        depth_m,                     # forward (= planar depth)
    ], dtype=float)


def camera_to_world_frame(
    pos_camera: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    camera_world_pos: np.ndarray = None,
) -> np.ndarray:
    """
    Rotate a point from camera frame into world frame (ZYX extrinsic convention).

    Args:
        pos_camera:       [x,y,z] in camera frame
        yaw_deg:          rotation about world Z (+ = left)
        pitch_deg:        rotation about world Y (+ = nose up)
        roll_deg:         rotation about world X (+ = right wing down)
        camera_world_pos: camera origin in world frame; defaults to (0,0,0)
    """
    y, p, r = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)

    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,         1]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0,          1, 0        ],
                   [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1, 0,         0        ],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])

    pos_world = (Rz @ Ry @ Rx) @ pos_camera
    if camera_world_pos is not None:
        pos_world += camera_world_pos
    return pos_world


# ── Tiling + NMS helpers ──────────────────────────────────────────────────────

def get_tiles(frame_bgr, tile_w, tile_h, overlap):
    fh, fw = frame_bgr.shape[:2]
    step_x = tile_w - overlap
    step_y = tile_h - overlap
    tiles  = []
    y = 0
    while y < fh:
        x = 0
        while x < fw:
            x1 = min(x, fw - tile_w)
            y1 = min(y, fh - tile_h)
            tiles.append((frame_bgr[y1:y1 + tile_h, x1:x1 + tile_w], x1, y1))
            if x + tile_w >= fw:
                break
            x += step_x
        if y + tile_h >= fh:
            break
        y += step_y
    return tiles


def darknet_infer(tile_bgr):
    tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    dn_img   = darknet.make_image(NET_W, NET_H, 3)
    darknet.copy_image_from_bytes(dn_img, tile_rgb.tobytes())
    dets = darknet.detect_image(network, class_names, dn_img, thresh=THRESH)
    darknet.free_image(dn_img)
    return dets


def to_abs_bbox(bbox, x_off, y_off):
    cx, cy, bw, bh = bbox
    return (
        x_off + cx - bw / 2,
        y_off + cy - bh / 2,
        x_off + cx + bw / 2,
        y_off + cy + bh / 2,
    )


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1  = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2  = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


def nms(detections, iou_thresh=NMS_IOU, center_thresh=CENTER_THRESH):
    if not detections:
        return []
    by_class = {}
    for label, conf, box in detections:
        by_class.setdefault(label, []).append((float(conf), box))
    kept = []
    for label, items in by_class.items():
        items.sort(key=lambda x: x[0], reverse=True)
        accepted = []
        for conf, box in items:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2;  cy = (y1 + y2) / 2
            too_close = any(
                ((cx - (ax1+ax2)/2)**2 + (cy - (ay1+ay2)/2)**2)**0.5 < center_thresh
                for _, (ax1, ay1, ax2, ay2) in accepted
            )
            if too_close:
                continue
            if any(iou(box, ab) >= iou_thresh for _, ab in accepted):
                continue
            accepted.append((conf, box))
            kept.append((label, conf, box))
    return kept


# ── Camera ────────────────────────────────────────────────────────────────────

class Camera:
    def __init__(self):
        self.pipeline      = dai.Pipeline()
        self.device        = self.pipeline.getDefaultDevice()
        self.subpixel_bits = 5
        self.q_sync        = None
        self._configure()

        print(f"DeviceID:  {self.device.getDeviceInfo().getDeviceId()}")
        print(f"USB speed: {self.device.getUsbSpeed()}")

    def _configure(self):
        # RGB
        cam_rgb = self.pipeline.create(dai.node.Camera)
        cam_rgb.build(dai.CameraBoardSocket.CAM_A)
        rgb_out = cam_rgb.requestOutput(
            size=(DISPLAY_W, DISPLAY_H),
            type=dai.ImgFrame.Type.BGR888p,
            fps=30
        )

        # Stereo
        cam_left  = self.pipeline.create(dai.node.Camera)
        cam_right = self.pipeline.create(dai.node.Camera)
        cam_left.build(dai.CameraBoardSocket.CAM_B)
        cam_right.build(dai.CameraBoardSocket.CAM_C)

        left_out  = cam_left.requestOutput(size=(640, 400), fps=30)
        right_out = cam_right.requestOutput(size=(640, 400), fps=30)

        stereo = self.pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setSubpixelFractionalBits(self.subpixel_bits)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(DISPLAY_W, DISPLAY_H)

        left_out.link(stereo.left)
        right_out.link(stereo.right)

        # Sync
        sync = self.pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=50))
        rgb_out.link(sync.inputs["rgb"])
        stereo.disparity.link(sync.inputs["disparity"])

        self.q_sync = sync.out.createOutputQueue(maxSize=4, blocking=False)

    def start(self):
        self.pipeline.start()

    def get_synced(self):
        group = self.q_sync.tryGet()
        if group is None:
            return None
        return group["rgb"].getCvFrame(), group["disparity"].getFrame()

    def get_depth_in_bbox(self, disparity, bbox, focal_length_px, baseline_m):
        x1, y1, x2, y2 = bbox
        disp_h, disp_w  = disparity.shape[:2]
        x1 = max(0, min(int(x1), disp_w - 1))
        y1 = max(0, min(int(y1), disp_h - 1))
        x2 = max(0, min(int(x2), disp_w - 1))
        y2 = max(0, min(int(y2), disp_h - 1))
        roi   = disparity[y1:y2, x1:x2]
        valid = roi[roi > 0].astype(np.float32)
        if valid.size == 0:
            return 0.0
        median_disp_px = np.median(valid) / (2 ** self.subpixel_bits)
        return float((focal_length_px * baseline_m) / median_disp_px)


# ── ObjectDetector ────────────────────────────────────────────────────────────

class ObjectDetector:
    def find_objects(self, frame_bgr):
        """
        Returns [(label, confidence, (x1,y1,x2,y2)), ...] sorted by area descending.
        """
        all_dets = []
        for tile_bgr, x_off, y_off in get_tiles(frame_bgr, NET_W, NET_H, OVERLAP):
            for label, conf, bbox in darknet_infer(tile_bgr):
                all_dets.append((label, conf, to_abs_bbox(bbox, x_off, y_off)))

        detections = nms(all_dets)

        detections.sort(
            key=lambda d: (d[2][2] - d[2][0]) * (d[2][3] - d[2][1]),
            reverse=True
        )
        return detections


# ── Manager ───────────────────────────────────────────────────────────────────

class Manager:
    def __init__(self):
        self.cam      = Camera()
        self.detector = ObjectDetector()
        self.focal_length_px = None
        self.baseline_m      = None
        self.fx  = None
        self.fy  = None
        self.cx  = None
        self.cy  = None

    def start(self):
        self.cam.start()
        calib = self.cam.device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, DISPLAY_W, DISPLAY_H
        )
        # Intrinsic matrix layout:
        # [[fx,  0, cx],
        #  [ 0, fy, cy],
        #  [ 0,  0,  1]]
        self.fx = intrinsics[0][0]
        self.fy = intrinsics[1][1]
        self.cx = intrinsics[0][2]
        self.cy = intrinsics[1][2]
        self.focal_length_px = self.fx          # used for disparity → depth
        self.baseline_m      = calib.getBaselineDistance() / 100.0
        print(f"Focal: {self.focal_length_px:.2f}px  Baseline: {self.baseline_m*100:.2f}cm")
        print(f"Principal point: ({self.cx:.1f}, {self.cy:.1f})")

    def update(self, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
               camera_world_pos=None):
        """
        Returns (rgb_frame, [ObjectLocation, ...]) or None if no frame yet.

        Pass yaw/pitch/roll (degrees) and camera_world_pos (np.ndarray [x,y,z])
        to get pos_world populated; otherwise pos_world is None.
        """
        result = self.cam.get_synced()
        if result is None:
            return None

        rgb, disparity = result
        detections     = self.detector.find_objects(rgb)

        locations = []
        for label, conf, bbox in detections:
            depth_m = self.cam.get_depth_in_bbox(
                disparity, bbox, self.focal_length_px, self.baseline_m
            )

            if depth_m <= 0:
                pos_cam   = np.zeros(3)
                pos_world = None
            else:
                pos_cam = bbox_depth_to_camera_frame(
                    bbox, depth_m,
                    self.fx, self.fy, self.cx, self.cy,
                )
                pos_world = camera_to_world_frame(
                    pos_cam, yaw_deg, pitch_deg, roll_deg, camera_world_pos
                )

            locations.append(ObjectLocation(
                label=label,
                conf=float(conf),
                bbox=bbox,
                depth_m=depth_m,
                pos_camera=pos_cam,
                pos_world=pos_world,
            ))

        return rgb, locations


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mgr = Manager()
    mgr.start()

    TARGET_LOOP_MS = 20
    print("Running — press 'q' to quit.")

    try:
        while True:
            loop_start = time.monotonic()

            # Plug in real yaw/pitch/roll from your IMU/pose source here
            result = mgr.update(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                                camera_world_pos=np.array([0.0, 0.0, 0.0]))
            if result is None:
                continue

            rgb, locations = result

            for loc in locations:
                x1, y1, x2, y2 = loc.bbox
                color = colours[loc.label]

                depth_str = f"{loc.depth_m:.2f}m" if loc.depth_m > 0 else "N/A"
                cam_str   = (f"cam=({loc.pos_camera[0]:+.2f}, "
                             f"{loc.pos_camera[1]:+.2f}, "
                             f"{loc.pos_camera[2]:.2f})m")

                cv2.rectangle(rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    rgb, f"{loc.label} {loc.conf:.0f}% {depth_str}",
                    (int(x1), max(int(y1) - 20, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                )
                cv2.putText(
                    rgb, cam_str,
                    (int(x1), max(int(y1) - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1
                )

                print(
                    f"{loc.label}: {loc.conf:.1f}%  "
                    f"depth={depth_str}  "
                    f"cam=({loc.pos_camera[0]:+.3f}, {loc.pos_camera[1]:+.3f}, {loc.pos_camera[2]:.3f})m"
                    + (f"  world=({loc.pos_world[0]:+.3f}, {loc.pos_world[1]:+.3f}, {loc.pos_world[2]:.3f})m"
                       if loc.pos_world is not None else "")
                )

            if not locations:
                cv2.putText(rgb, "No detections", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Darknet - OAK-D", rgb)

            elapsed_ms   = (time.monotonic() - loop_start) * 1000
            remaining_ms = TARGET_LOOP_MS - elapsed_ms
            if remaining_ms > 0:
                time.sleep(remaining_ms / 1000)
            else:
                print(f"Loop overrun: {-remaining_ms:.1f}ms")

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        darknet.free_network_ptr(network)
        cv2.destroyAllWindows()
