#!/usr/bin/env python3
import sys
import time
import queue
import threading
import concurrent.futures
import numpy as np
import cv2
import math
sys.path.append('/home/peach/src/darknet/src-python')
import darknet
import depthai as dai
from datetime import timedelta
from dataclasses import dataclass
from typing import Optional

cfg_file     = "new_sample.cfg"
names_file   = "new_sample.names"
weights_file = "new_sample.weights"

DISPLAY_W, DISPLAY_H = 1280, 720
OVERLAP       = 64
NMS_IOU       = 0.3
CENTER_THRESH = 40
THRESH        = 0.4
POOL_SIZE     = 2


# ── Darknet network pool ──────────────────────────────────────────────────────

class NetworkPool:
    def __init__(self, cfg, weights, names, size=POOL_SIZE):
        self._pool = queue.Queue()
        self._class_names = open(names).read().splitlines()
        for _ in range(size):
            net = darknet.load_net_custom(
                cfg.encode("ascii"),
                weights.encode("ascii"),
                0, 1
            )
            self._pool.put(net)

    def acquire(self):          return self._pool.get()
    def release(self, net):     self._pool.put(net)
    def class_names(self):      return self._class_names


NET_POOL    = NetworkPool(cfg_file, weights_file, names_file, size=POOL_SIZE)
class_names = NET_POOL.class_names()
colours     = darknet.class_colors(class_names)

_tmp = NET_POOL.acquire()
NET_W = darknet.network_width(_tmp)
NET_H = darknet.network_height(_tmp)
NET_POOL.release(_tmp)

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=POOL_SIZE)


# ── Localization ──────────────────────────────────────────────────────────────

@dataclass
class ObjectLocation:
    label:      str
    conf:       float
    bbox:       tuple          # (x1, y1, x2, y2) in pixels
    depth_m:    float
    pos_camera: np.ndarray
    pos_world:  Optional[np.ndarray]


def bbox_depth_to_camera_frame(bbox, depth_m, fx, fy, cx, cy):
    x1, y1, x2, y2 = bbox
    px = (x1 + x2) / 2.0
    py = (y1 + y2) / 2.0
    theta_x = np.arctan2(px - cx, fx)
    theta_y = np.arctan2(py - cy, fy)
    return np.array([
        depth_m * np.tan(theta_x),
        -depth_m * np.tan(theta_y),
        depth_m,
    ], dtype=float)


def camera_to_world_frame(pos_camera, yaw_deg, pitch_deg, roll_deg,
                           camera_world_pos=None):
    y, p, r = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,         1]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0,          1, 0        ],
                   [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1, 0,          0         ],
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
            if x + tile_w >= fw: break
            x += step_x
        if y + tile_h >= fh: break
        y += step_y
    return tiles


def darknet_infer(tile_bgr):
    tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    net = NET_POOL.acquire()
    try:
        dn_img = darknet.make_image(NET_W, NET_H, 3)
        darknet.copy_image_from_bytes(dn_img, tile_rgb.tobytes())
        dets = darknet.detect_image(net, class_names, dn_img, thresh=THRESH)
        darknet.free_image(dn_img)
        return dets
    finally:
        NET_POOL.release(net)


def to_abs_bbox(bbox, x_off, y_off):
    cx, cy, bw, bh = bbox
    return (x_off + cx - bw/2, y_off + cy - bh/2,
            x_off + cx + bw/2, y_off + cy + bh/2)


def iou(a, b):
    ax1,ay1,ax2,ay2 = a;  bx1,by1,bx2,by2 = b
    ix1 = max(ax1,bx1);   iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2);   iy2 = min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


def nms(detections, iou_thresh=NMS_IOU, center_thresh=CENTER_THRESH):
    if not detections: return []
    by_class = {}
    for label, conf, box in detections:
        by_class.setdefault(label, []).append((float(conf), box))
    kept = []
    for label, items in by_class.items():
        items.sort(key=lambda x: x[0], reverse=True)
        accepted = []
        for conf, box in items:
            x1,y1,x2,y2 = box
            cx = (x1+x2)/2;  cy = (y1+y2)/2
            too_close = any(
                ((cx-(ax1+ax2)/2)**2 + (cy-(ay1+ay2)/2)**2)**0.5 < center_thresh
                for _, (ax1,ay1,ax2,ay2) in accepted
            )
            if too_close: continue
            if any(iou(box, ab) >= iou_thresh for _, ab in accepted): continue
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

    def _configure(self):
        cam_rgb = self.pipeline.create(dai.node.Camera)
        cam_rgb.build(dai.CameraBoardSocket.CAM_A)
        rgb_out = cam_rgb.requestOutput(
            size=(DISPLAY_W, DISPLAY_H),
            type=dai.ImgFrame.Type.BGR888p, fps=30
        )

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

        sync = self.pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=50))
        rgb_out.link(sync.inputs["rgb"])
        stereo.disparity.link(sync.inputs["disparity"])

        self.q_sync = sync.out.createOutputQueue(maxSize=4, blocking=False)

    def start(self):
        self.pipeline.start()

    def get_synced(self):
        # Drain the queue — always return the most recent frame.
        group = None
        while True:
            latest = self.q_sync.tryGet()
            if latest is None:
                break
            group = latest
        if group is None:
            return None
        return group["rgb"].getCvFrame(), group["disparity"].getFrame()
        
    def get_depth_in_bbox(self, disparity, bbox, focal_length_px, baseline_m, center_fraction=0.05):
        x1, y1, x2, y2 = bbox
        dh, dw = disparity.shape[:2]
        x1 = max(0, min(int(x1), dw-1));  y1 = max(0, min(int(y1), dh-1))
        x2 = max(0, min(int(x2), dw-1));  y2 = max(0, min(int(y2), dh-1))

        box_w = x2 - x1
        box_h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        half_w = max(math.ceil(box_w * center_fraction / 2), 3)  # 3 -> 6px min span
        half_h = max(math.ceil(box_h * center_fraction / 2), 3)

        rx1 = max(0,    int(cx) - half_w)
        rx2 = min(dw-1, int(cx) + half_w)
        ry1 = max(0,    int(cy) - half_h)
        ry2 = min(dh-1, int(cy) + half_h)

        # Guarantee at least 5px slice after clamping
        if rx2 - rx1 < 5: rx2 = min(rx1 + 5, dw-1)
        if ry2 - ry1 < 5: ry2 = min(ry1 + 5, dh-1)

        roi   = disparity[ry1:ry2, rx1:rx2]
        valid = roi[roi > 0].astype(np.float32)
        if valid.size == 0: return 0.0
        median_disp_px = np.median(valid) / (2 ** self.subpixel_bits)
        return float((focal_length_px * baseline_m) / median_disp_px)
"""    def get_depth_in_bbox(self, disparity, bbox, focal_length_px, baseline_m):
        x1,y1,x2,y2 = bbox
        dh, dw = disparity.shape[:2]
        x1 = max(0, min(int(x1), dw-1));  y1 = max(0, min(int(y1), dh-1))
        x2 = max(0, min(int(x2), dw-1));  y2 = max(0, min(int(y2), dh-1))
        roi   = disparity[y1:y2, x1:x2]
        valid = roi[roi > 0].astype(np.float32)
        if valid.size == 0: return 0.0
        median_disp_px = np.median(valid) / (2 ** self.subpixel_bits)
        return float((focal_length_px * baseline_m) / median_disp_px) """


# ── ObjectDetector ────────────────────────────────────────────────────────────

class ObjectDetector:
    def find_objects(self, frame_bgr):
        tiles = get_tiles(frame_bgr, NET_W, NET_H, OVERLAP)
        futures = {
            _EXECUTOR.submit(darknet_infer, tile_bgr): (x_off, y_off)
            for tile_bgr, x_off, y_off in tiles
        }
        all_dets = []
        for future, (x_off, y_off) in futures.items():
            for label, conf, bbox in future.result():
                all_dets.append((label, conf, to_abs_bbox(bbox, x_off, y_off)))
        detections = nms(all_dets)
        detections.sort(
            key=lambda d: (d[2][2]-d[2][0]) * (d[2][3]-d[2][1]),
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
        self.fx = self.fy = self.cx = self.cy = None

    def start(self):
        self.cam.start()
        calib = self.cam.device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, DISPLAY_W, DISPLAY_H
        )
        self.fx = intrinsics[0][0];  self.fy = intrinsics[1][1]
        self.cx = intrinsics[0][2];  self.cy = intrinsics[1][2]
        self.focal_length_px = self.fx
        self.baseline_m      = calib.getBaselineDistance() / 100.0
        print(f"Focal: {self.focal_length_px:.2f}px  "
              f"Baseline: {self.baseline_m*100:.2f}cm")
        print(f"Principal point: ({self.cx:.1f}, {self.cy:.1f})")


# ── Shared slots ──────────────────────────────────────────────────────────────

class AtomicSlot:
    """
    GIL-atomic single-producer / single-consumer slot.

    put()        — overwrite with a new payload (producer)
    get_if_new() — return payload only if seq advanced (consumer, skips dupes)
    get_latest() — always return current payload regardless of seq (display)
    """
    def __init__(self):
        self._value = None
        self._seq   = 0

    def put(self, *payload):
        self._seq  += 1
        self._value = (self._seq, *payload)

    def get_if_new(self, last_seq):
        v = self._value
        if v is None or v[0] == last_seq:
            return None, last_seq
        return v[1:], v[0]

    def get_latest(self):
        v = self._value
        return v[1:] if v is not None else None


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_locations(frame, locations, infer_ms):
    for loc in locations:
        x1, y1, x2, y2 = loc.bbox
        color     = colours[loc.label]
        depth_str = f"{loc.depth_m:.2f}m" if loc.depth_m > 0 else "N/A"
        cam_str   = (f"cam=({loc.pos_camera[0]:+.2f}, "
                     f"{loc.pos_camera[1]:+.2f}, "
                     f"{loc.pos_camera[2]:.2f})m")
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{loc.label} {loc.conf:.0f}% {depth_str}",
                    (int(x1), max(int(y1) - 20, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, cam_str,
                    (int(x1), max(int(y1) - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    if not locations:
        cv2.putText(frame, "No detections", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"infer {infer_ms:.0f}ms", (DISPLAY_W - 130, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mgr = Manager()
    mgr.start()

    # frame_slot     : camera thread  → (rgb, disparity)
    # detection_slot : inference thread → (locations, infer_ms)
    frame_slot     = AtomicSlot()
    detection_slot = AtomicSlot()
    stop_event     = threading.Event()

    # ── Camera thread ─────────────────────────────────────────────────────────
    # Sole job: drain the OAK-D sync queue and publish the latest raw frame.
    # Runs independently of inference so the display is never starved.
    def camera_loop():
        while not stop_event.is_set():
            result = mgr.cam.get_synced()
            if result is None:
                time.sleep(0.002)
                continue
            rgb, disparity = result
            frame_slot.put(rgb, disparity)

    # ── Inference thread ──────────────────────────────────────────────────────
    # Reads the latest frame from frame_slot (skipping frames it can't keep up
    # with), runs detection + depth, then publishes ObjectLocations.
    def inference_loop():
        infer_count = 0
        total_infer = 0.0
        last_seq    = 0

        while not stop_event.is_set():
            snapshot, last_seq = frame_slot.get_if_new(last_seq)
            if snapshot is None:
                time.sleep(0.001)
                continue

            rgb, disparity = snapshot

            t0         = time.monotonic()
            detections = mgr.detector.find_objects(rgb)
            infer_ms   = (time.monotonic() - t0) * 1000

            locations = []
            for label, conf, bbox in detections:
                depth_m = mgr.cam.get_depth_in_bbox(
                    disparity, bbox, mgr.focal_length_px, mgr.baseline_m
                )
                if depth_m <= 0:
                    pos_cam   = np.zeros(3)
                    pos_world = None
                else:
                    pos_cam = bbox_depth_to_camera_frame(
                        bbox, depth_m, mgr.fx, mgr.fy, mgr.cx, mgr.cy
                    )
                    pos_world = camera_to_world_frame(
                        pos_cam, 0.0, 0.0, 0.0,
                        np.array([0.0, 0.0, 0.0])
                    )
                locations.append(ObjectLocation(
                    label=label, conf=float(conf), bbox=bbox,
                    depth_m=depth_m, pos_camera=pos_cam, pos_world=pos_world,
                ))

            detection_slot.put(locations, infer_ms)

            infer_count += 1
            total_infer += infer_ms
            if infer_count % 30 == 0:
                avg = total_infer / infer_count
                print(f"[infer] avg {avg:.1f}ms  ({1000/avg:.1f} fps)")
                for loc in locations:
                    print(
                        f"  {loc.label}: {loc.conf:.1f}%  "
                        f"depth={loc.depth_m:.2f}m  "
                        f"cam=({loc.pos_camera[0]:+.3f}, "
                        f"{loc.pos_camera[1]:+.3f}, "
                        f"{loc.pos_camera[2]:.3f})m"
                        + (f"  world=({loc.pos_world[0]:+.3f}, "
                           f"{loc.pos_world[1]:+.3f}, "
                           f"{loc.pos_world[2]:.3f})m"
                           if loc.pos_world is not None else "")
                    )

    camera_thread = threading.Thread(target=camera_loop,    daemon=True)
    infer_thread  = threading.Thread(target=inference_loop, daemon=True)
    camera_thread.start()
    infer_thread.start()

    # ── Display loop (main thread) ────────────────────────────────────────────
    # Always pulls the freshest raw frame from frame_slot — no sequence check —
    # then overlays whatever detections are currently available.  Bounding boxes
    # may lag behind fast-moving objects but the video feed is always live.
    print("Running — press 'q' to quit.")

    current_locations = []
    current_infer_ms  = 0.0
    last_det_seq      = 0

    try:
        while True:
            # Fresh detections if inference finished a new batch
            det_snapshot, last_det_seq = detection_slot.get_if_new(last_det_seq)
            if det_snapshot is not None:
                current_locations, current_infer_ms = det_snapshot

            # Always display the latest camera frame
            frame_data = frame_slot.get_latest()
            if frame_data is not None:
                rgb, _ = frame_data
                display_frame = rgb.copy()
                draw_locations(display_frame, current_locations, current_infer_ms)
                cv2.imshow("Darknet - OAK-D", display_frame)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        stop_event.set()
        camera_thread.join(timeout=5)
        infer_thread.join(timeout=5)
        _EXECUTOR.shutdown(wait=True)
        for _ in range(POOL_SIZE):
            net = NET_POOL.acquire()
            darknet.free_network_ptr(net)
        cv2.destroyAllWindows()
