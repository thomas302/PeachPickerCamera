#!/usr/bin/env python3
"""
OAK-D + TensorRT object detector — optimized for Orin Nano.
5 fixed tiles: 4 corners + centre. Each source region is exactly 1/4 of the
frame (960×540), resized to NET_W×NET_H. One CUDA push/pop per frame,
one H→D, 5×execute_async_v3, one D→H.
"""

import sys
import time
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import depthai as dai
from datetime import timedelta
from dataclasses import dataclass
from typing import Optional, List

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--profile", action="store_true")
args = ap.parse_args()

# ── CUDA init ─────────────────────────────────────────────────────────────────
cuda.init()
_cuda_device  = cuda.Device(0)
_cuda_context = _cuda_device.make_context()

# ── Config ────────────────────────────────────────────────────────────────────
ENGINE_PATH = "model.engine"
NAMES_FILE  = "new_sample.names"

DISPLAY_W, DISPLAY_H = 1920, 1080
NMS_IOU   = 0.1
THRESH    = 0.40
BOX_SCALE = 1.0

NET_W = 416
NET_H = 416
N_ANCHORS = 10647   # 13×13×3 + 26×26×3 + 52×52×3

N_TILES = 5

# ── Byte strides per tile in the flat device buffers ─────────────────────────
# Engine tensors shapes per tile: input=(1,3,416,416)  confs=(1,A,1)  boxes=(1,A,1,4)
_BYTES_PER_TILE_INPUT = 1 * 3 * NET_H * NET_W * 4
_BYTES_PER_TILE_CONFS = 1 * N_ANCHORS * 1 * 4
_BYTES_PER_TILE_BOXES = 1 * N_ANCHORS * 1 * 4 * 4

# ── 5-tile source regions ─────────────────────────────────────────────────────
# 4 corners = even quadrants (960×540). Centre tile is 960×540 centred on the
# frame so it overlaps all four corners by 480px horiz and 270px vert.
# All regions are resized to NET_W×NET_H before inference, so decode applies
# per-tile scale factors to map box coords back to full-frame pixels.
#
#   ┌──────────┬──────────┐
#   │  TL      │  TR      │
#   │    ╔═════╪════╗     │
#   │    ║  CENTER  ║     │
#   │    ╚═════╪════╝     │
#   │  BL      │  BR      │
#   └──────────┴──────────┘
#
_QW = DISPLAY_W // 2    # 960  quadrant width
_QH = DISPLAY_H // 2    # 540  quadrant height
_CX = (DISPLAY_W - _QW) // 2   # 480  centre tile left edge
_CY = (DISPLAY_H - _QH) // 2   # 270  centre tile top edge

# (x1, y1, x2, y2) in full-frame pixels
TILE_REGIONS = [
    (0,   0,   _QW,        _QH),           # 0 TL
    (_QW, 0,   DISPLAY_W,  _QH),           # 1 TR
    (0,   _QH, _QW,        DISPLAY_H),     # 2 BL
    (_QW, _QH, DISPLAY_W,  DISPLAY_H),     # 3 BR
    (_CX, _CY, _CX + _QW, _CY + _QH),     # 4 Centre (overlaps all)
]

# Scale factors: how many full-frame pixels each NET pixel represents per tile
_TILE_SX = [(x2 - x1) / NET_W for x1, y1, x2, y2 in TILE_REGIONS]
_TILE_SY = [(y2 - y1) / NET_H for x1, y1, x2, y2 in TILE_REGIONS]


# ── TRT Batch Inferencer ──────────────────────────────────────────────────────

class TRTBatchInferencer:
    """
    Static batch=1 engine.
    Pre-allocates one large pinned+device buffer for MAX_TILES tiles.
    Per frame: one H→D memcpy, N×execute_async_v3 with offset pointers,
    one D→H memcpy, one stream.synchronize().
    Only one CUDA context push/pop per frame.
    """

    def __init__(self, engine_path: str):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

        _cuda_context.push()
        self.ctx    = engine.create_execution_context()
        self.stream = cuda.Stream()

        # Pinned host buffers sized for exactly N_TILES
        self.h_input = cuda.pagelocked_empty(
            (N_TILES, 3, NET_H, NET_W), dtype=np.float32)
        self.h_confs = cuda.pagelocked_empty(
            (N_TILES, N_ANCHORS, 1),    dtype=np.float32)
        self.h_boxes = cuda.pagelocked_empty(
            (N_TILES, N_ANCHORS, 1, 4), dtype=np.float32)

        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_confs = cuda.mem_alloc(self.h_confs.nbytes)
        self.d_boxes = cuda.mem_alloc(self.h_boxes.nbytes)

        _cuda_context.pop()
        print(f"✓ TRT engine loaded  tiles={N_TILES}")
        print(f"  strides — input:{_BYTES_PER_TILE_INPUT}B  "
              f"confs:{_BYTES_PER_TILE_CONFS}B  "
              f"boxes:{_BYTES_PER_TILE_BOXES}B")

    def infer_batch(self, tiles_rgb: list):
        """
        tiles_rgb : list of exactly N_TILES (NET_H, NET_W, 3) uint8 arrays,
                    already resized to net resolution.
        Returns   : (h_confs view N×A×1, h_boxes view N×A×1×4) — pinned memory
        """
        # CPU preprocessing → pinned input
        for i, tile in enumerate(tiles_rgb):
            np.divide(tile.transpose(2, 0, 1), 255.0, out=self.h_input[i])

        _cuda_context.push()
        try:
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

            d_in_base    = int(self.d_input)
            d_confs_base = int(self.d_confs)
            d_boxes_base = int(self.d_boxes)

            for i in range(N_TILES):
                self.ctx.set_tensor_address(
                    'frame', d_in_base    + i * _BYTES_PER_TILE_INPUT)
                self.ctx.set_tensor_address(
                    'confs', d_confs_base + i * _BYTES_PER_TILE_CONFS)
                self.ctx.set_tensor_address(
                    'boxes', d_boxes_base + i * _BYTES_PER_TILE_BOXES)
                self.ctx.execute_async_v3(self.stream.handle)

            cuda.memcpy_dtoh_async(self.h_confs, self.d_confs, self.stream)
            cuda.memcpy_dtoh_async(self.h_boxes, self.d_boxes, self.stream)
            self.stream.synchronize()

        finally:
            _cuda_context.pop()

        return self.h_confs, self.h_boxes


# ── Globals ───────────────────────────────────────────────────────────────────
inferencer  = TRTBatchInferencer(ENGINE_PATH)
class_names = open(NAMES_FILE).read().splitlines()
colours     = {name: tuple(int(x) for x in np.random.randint(50, 255, 3))
               for name in class_names}


# ── Localization ──────────────────────────────────────────────────────────────

@dataclass
class ObjectLocation:
    label:      str
    conf:       float
    bbox:       tuple
    depth_m:    float
    pos_camera: np.ndarray
    pos_world:  Optional[np.ndarray]


def bbox_depth_to_camera_frame(bbox, depth_m, fx, fy, cx, cy):
    x1, y1, x2, y2 = bbox
    px = (x1 + x2) / 2.0
    py = (y1 + y2) / 2.0
    return np.array([
        depth_m * np.tan(np.arctan2(px - cx, fx)),
        depth_m * np.tan(np.arctan2(py - cy, fy)),
        depth_m,
    ], dtype=np.float32)


class RotationCache:
    """Recomputes rotation matrix only when angles change."""
    def __init__(self):
        self._last = None
        self._R    = np.eye(3, dtype=np.float64)

    def get(self, yaw_deg, pitch_deg, roll_deg):
        key = (yaw_deg, pitch_deg, roll_deg)
        if key == self._last:
            return self._R
        y, p, r = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
        Rz = np.array([[np.cos(y), -np.sin(y), 0],
                       [np.sin(y),  np.cos(y), 0],
                       [0,          0,          1]])
        Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                       [0,          1, 0         ],
                       [-np.sin(p), 0, np.cos(p)]])
        Rx = np.array([[1, 0,           0        ],
                       [0, np.cos(r), -np.sin(r) ],
                       [0, np.sin(r),  np.cos(r) ]])
        self._R    = Rz @ Ry @ Rx
        self._last = key
        return self._R

_rot_cache = RotationCache()


# ── Tile extraction ───────────────────────────────────────────────────────────

# Pre-allocate resize output buffers (avoids malloc per frame)
_tile_bufs = [np.empty((NET_H, NET_W, 3), dtype=np.uint8) for _ in TILE_REGIONS]

def build_tiles(frame_rgb):
    """
    Crop each of the 5 fixed regions and resize to NET_W×NET_H in-place.
    Returns list of (resized_hwc_uint8, tile_index) — always length N_TILES.
    """
    tiles = []
    for i, (x1, y1, x2, y2) in enumerate(TILE_REGIONS):
        cv2.resize(frame_rgb[y1:y2, x1:x2], (NET_W, NET_H),
                   dst=_tile_bufs[i], interpolation=cv2.INTER_LINEAR)
        tiles.append(_tile_bufs[i])
    return tiles


# ── Vectorised decode ─────────────────────────────────────────────────────────

def decode_batch(h_confs, h_boxes):
    """
    h_confs : (N_TILES, A, 1)    pinned view
    h_boxes : (N_TILES, A, 1, 4) pinned view

    Box coords from the model are normalised to [0,1] within the NET tile.
    We map back to full-frame pixels using each tile's origin + scale:
        frame_x = box_x_norm * NET_W * sx + origin_x

    The centre tile overlaps all corners so duplicate detections are expected
    and resolved by NMS.
    """
    confs = h_confs[:, :, 0]       # (N, A)
    boxes = h_boxes[:, :, 0, :]    # (N, A, 4)

    mask = confs > THRESH
    if not mask.any():
        return []

    tile_idx, anc_idx = np.where(mask)

    c = confs[tile_idx, anc_idx]   # (K,)
    b = boxes[tile_idx, anc_idx]   # (K, 4)  normalised [0,1] in tile space

    # Per-detection origin and scale from the tile it came from
    origins = np.array([(TILE_REGIONS[t][0], TILE_REGIONS[t][1])
                        for t in tile_idx], dtype=np.float32)   # (K, 2)
    scales  = np.array([(_TILE_SX[t], _TILE_SY[t])
                        for t in tile_idx], dtype=np.float32)   # (K, 2)

    x1 = b[:, 0] * NET_W * scales[:, 0] + origins[:, 0]
    y1 = b[:, 1] * NET_H * scales[:, 1] + origins[:, 1]
    x2 = b[:, 2] * NET_W * scales[:, 0] + origins[:, 0]
    y2 = b[:, 3] * NET_H * scales[:, 1] + origins[:, 1]

    if BOX_SCALE != 1.0:
        cx = (x1 + x2) / 2;  hw = (x2 - x1) / 2 * BOX_SCALE
        cy = (y1 + y2) / 2;  hh = (y2 - y1) / 2 * BOX_SCALE
        x1, x2 = cx - hw, cx + hw
        y1, y2 = cy - hh, cy + hh

    label = class_names[0]
    return [
        (label, float(c[i]), (float(x1[i]), float(y1[i]),
                               float(x2[i]), float(y2[i])))
        for i in range(len(c))
    ]


# ── NMS ───────────────────────────────────────────────────────────────────────

def nms(detections):
    if not detections:
        return []

    by_class = {}
    for label, conf, box in detections:
        by_class.setdefault(label, []).append((conf, box))

    kept = []
    for label, items in by_class.items():
        cv_boxes = [[x1, y1, x2 - x1, y2 - y1]
                    for _, (x1, y1, x2, y2) in items]
        scores   = [float(c) for c, _ in items]
        indices  = cv2.dnn.NMSBoxes(cv_boxes, scores,
                                    score_threshold=THRESH,
                                    nms_threshold=NMS_IOU)
        for i in indices:
            conf, box = items[i]
            kept.append((label, conf, box))

    kept.sort(key=lambda d: (d[2][2]-d[2][0])*(d[2][3]-d[2][1]), reverse=True)
    return kept


# ── ObjectDetector ────────────────────────────────────────────────────────────

class ObjectDetector:
    def find_objects(self, frame_bgr):
        frame_rgb        = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tiles            = build_tiles(frame_rgb)
        h_confs, h_boxes = inferencer.infer_batch(tiles)
        return nms(decode_batch(h_confs, h_boxes))


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
            type=dai.ImgFrame.Type.BGR888p,
            fps=30,
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
        group = self.q_sync.tryGet()
        if group is None:
            return None
        return group["rgb"].getCvFrame(), group["disparity"].getFrame()

    def get_depth_batch(self, disparity, bboxes, focal_length_px, baseline_m):
        disp_h, disp_w = disparity.shape[:2]
        scale = 1.0 / (2 ** self.subpixel_bits)
        depths = []
        for (x1, y1, x2, y2) in bboxes:
            x1c = max(0, min(int(x1), disp_w - 1))
            y1c = max(0, min(int(y1), disp_h - 1))
            x2c = max(0, min(int(x2), disp_w - 1))
            y2c = max(0, min(int(y2), disp_h - 1))
            roi   = disparity[y1c:y2c, x1c:x2c]
            valid = roi[roi > 0].ravel()
            if valid.size == 0:
                depths.append(0.0)
                continue
            mid = valid.size // 2
            med = float(np.partition(valid, mid)[mid]) * scale
            depths.append(float(focal_length_px * baseline_m / med)
                          if med > 0 else 0.0)
        return depths


# ── Manager ───────────────────────────────────────────────────────────────────

class Manager:
    def __init__(self):
        self.cam             = Camera()
        self.detector        = ObjectDetector()
        self.focal_length_px = None
        self.baseline_m      = None
        self.fx = self.fy = self.cx = self.cy = None

    def start(self):
        self.cam.start()
        calib      = self.cam.device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, DISPLAY_W, DISPLAY_H)
        self.fx = intrinsics[0][0]
        self.fy = intrinsics[1][1]
        self.cx = intrinsics[0][2]
        self.cy = intrinsics[1][2]
        self.focal_length_px = self.fx
        self.baseline_m      = calib.getBaselineDistance() / 100.0
        print(f"Focal: {self.focal_length_px:.2f}px  "
              f"Baseline: {self.baseline_m*100:.2f}cm")

    def update(self, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
               camera_world_pos=None):
        result = self.cam.get_synced()
        if result is None:
            return None

        bgr, disparity = result
        detections     = self.detector.find_objects(bgr)

        if not detections:
            return bgr, []

        bboxes = [d[2] for d in detections]
        depths = self.cam.get_depth_batch(
            disparity, bboxes, self.focal_length_px, self.baseline_m)

        R = _rot_cache.get(yaw_deg, pitch_deg, roll_deg)

        locations = []
        for (label, conf, bbox), depth_m in zip(detections, depths):
            if depth_m > 0:
                pos_cam   = bbox_depth_to_camera_frame(
                    bbox, depth_m, self.fx, self.fy, self.cx, self.cy)
                pos_world = R @ pos_cam
                if camera_world_pos is not None:
                    pos_world = pos_world + camera_world_pos
            else:
                pos_cam   = np.zeros(3, dtype=np.float32)
                pos_world = None

            locations.append(ObjectLocation(
                label=label, conf=float(conf), bbox=bbox,
                depth_m=depth_m, pos_camera=pos_cam, pos_world=pos_world,
            ))

        return bgr, locations


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mgr = Manager()
    mgr.start()

    TARGET_LOOP_MS = 20
    frame_count    = 0
    total_ms       = 0.0

    if args.profile:
        import cProfile, pstats
        profiler = cProfile.Profile()

    print("Running — press 'q' to quit.")

    try:
        while True:
            if args.profile:
                profiler.enable()

            loop_start = time.monotonic()
            result = mgr.update(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                                camera_world_pos=np.zeros(3, dtype=np.float32))
            if result is None:
                if args.profile:
                    profiler.disable()
                continue

            bgr, locations = result
            elapsed_ms  = (time.monotonic() - loop_start) * 1000
            frame_count += 1
            total_ms    += elapsed_ms

            for loc in locations:
                x1, y1, x2, y2 = loc.bbox
                color     = colours[loc.label]
                depth_str = f"{loc.depth_m:.2f}m" if loc.depth_m > 0 else "N/A"
                cam_str   = (f"cam=({loc.pos_camera[0]:+.2f},"
                             f"{loc.pos_camera[1]:+.2f},"
                             f"{loc.pos_camera[2]:.2f})m")

                cv2.rectangle(bgr,
                              (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(bgr,
                            f"{loc.label} {loc.conf*100:.0f}% {depth_str}",
                            (int(x1), max(int(y1) - 20, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(bgr, cam_str,
                            (int(x1), max(int(y1) - 4, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

                print(
                    f"{loc.label}: {loc.conf*100:.1f}%  depth={depth_str}  "
                    f"cam=({loc.pos_camera[0]:+.3f},"
                    f"{loc.pos_camera[1]:+.3f},"
                    f"{loc.pos_camera[2]:.3f})m"
                    + (f"  world=({loc.pos_world[0]:+.3f},"
                       f"{loc.pos_world[1]:+.3f},"
                       f"{loc.pos_world[2]:.3f})m"
                       if loc.pos_world is not None else "")
                )

            if not locations:
                cv2.putText(bgr, "No detections", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if frame_count % 30 == 0:
                avg_ms = total_ms / frame_count
                print(f"[perf] avg loop: {avg_ms:.1f}ms  ({1000/avg_ms:.1f} fps)")

            cv2.putText(bgr, f"{elapsed_ms:.0f}ms",
                        (DISPLAY_W - 80, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.imshow("TRT - OAK-D", bgr)

            if args.profile:
                profiler.disable()

            remaining_ms = TARGET_LOOP_MS - elapsed_ms
            if remaining_ms > 0:
                time.sleep(remaining_ms / 1000)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        _cuda_context.pop()
        cv2.destroyAllWindows()
        if args.profile:
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.print_stats(20)
