#!/usr/bin/env python3
import sys
import time
import queue
import threading
import numpy as np
import cv2
import cProfile
import pstats
import tensorrt as trt
import pycuda.driver as cuda
import depthai as dai
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from dataclasses import dataclass

# ── CUDA init — manual, NOT pycuda.autoinit ───────────────────────────────────
# autoinit creates a context bound to the main thread only.
# We create it manually so worker threads can push/pop it.
cuda.init()
_cuda_device  = cuda.Device(0)
_cuda_context = _cuda_device.make_context()

engine_path = "model.engine"
names_file  = "new_sample.names"

DISPLAY_W, DISPLAY_H = 1920, 1080
OVERLAP   = 32
NMS_IOU   = 0.1
THRESH    = 0.40
BOX_SCALE = 1.0
WORKERS   = 3

NET_W = 416
NET_H = 416

N_ANCHORS = 10647   # 13×13×3 + 26×26×3 + 52×52×3


# ── TensorRT engine + buffer pool ─────────────────────────────────────────────

class TRTInferencer:
    """
    Loads one TRT engine. Creates WORKERS execution contexts, each with its
    own stream and pinned buffers. Workers acquire a slot, push the shared
    CUDA context, run inference, pop the context, release the slot.
    """
    def __init__(self, engine_path, num_slots):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime    = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self._slots = queue.Queue()
        for _ in range(num_slots):
            _cuda_context.push()
            ctx    = self.engine.create_execution_context()
            stream = cuda.Stream()

            h_input = cuda.pagelocked_empty((1, 3, NET_H, NET_W), dtype=np.float32)
            h_confs = cuda.pagelocked_empty((1, N_ANCHORS, 1),    dtype=np.float32)
            h_boxes = cuda.pagelocked_empty((1, N_ANCHORS, 1, 4), dtype=np.float32)

            d_input = cuda.mem_alloc(h_input.nbytes)
            d_confs = cuda.mem_alloc(h_confs.nbytes)
            d_boxes = cuda.mem_alloc(h_boxes.nbytes)

            ctx.set_tensor_address('frame', int(d_input))
            ctx.set_tensor_address('confs',  int(d_confs))
            ctx.set_tensor_address('boxes',  int(d_boxes))

            _cuda_context.pop()

            self._slots.put({
                'ctx':     ctx,
                'stream':  stream,
                'h_input': h_input, 'h_confs': h_confs, 'h_boxes': h_boxes,
                'd_input': d_input, 'd_confs': d_confs, 'd_boxes': d_boxes,
            })

        print(f"✓ TRT engine loaded: {engine_path} ({num_slots} slots)")

    def infer(self, tile_rgb_hwc_uint8):
        """
        Thread-safe inference. Acquires a slot, pushes CUDA context,
        runs inference, pops context, releases slot.
        """
        slot = self._slots.get()
        try:
            # CPU prep — no CUDA context needed
            np.copyto(
                slot['h_input'],
                (tile_rgb_hwc_uint8.astype(np.float32) / 255.0)
                .transpose(2, 0, 1)
                [np.newaxis],
            )

            # GPU work — push context for this thread
            _cuda_context.push()
            try:
                cuda.memcpy_htod_async(slot['d_input'], slot['h_input'], slot['stream'])
                slot['ctx'].execute_async_v3(slot['stream'].handle)
                cuda.memcpy_dtoh_async(slot['h_confs'], slot['d_confs'], slot['stream'])
                cuda.memcpy_dtoh_async(slot['h_boxes'], slot['d_boxes'], slot['stream'])
                slot['stream'].synchronize()
            finally:
                _cuda_context.pop()

            return (
                slot['h_confs'][0, :, 0].copy(),
                slot['h_boxes'][0, :, 0, :].copy(),
            )
        finally:
            self._slots.put(slot)


# ── Globals ───────────────────────────────────────────────────────────────────

inferencer  = TRTInferencer(engine_path, num_slots=WORKERS)
class_names = open(names_file).read().splitlines()
colours     = {name: tuple(int(x) for x in np.random.randint(50, 255, 3))
               for name in class_names}

_executor = ThreadPoolExecutor(max_workers=WORKERS)


# ── Localization ──────────────────────────────────────────────────────────────

@dataclass
class ObjectLocation:
    label:      str
    conf:       float
    bbox:       tuple
    depth_m:    float
    pos_camera: np.ndarray
    pos_world:  np.ndarray


def bbox_depth_to_camera_frame(bbox, depth_m, fx, fy, cx, cy):
    x1, y1, x2, y2 = bbox
    px = (x1 + x2) / 2.0
    py = (y1 + y2) / 2.0
    theta_x = np.arctan2(px - cx, fx)
    theta_y = np.arctan2(py - cy, fy)
    return np.array([
        depth_m * np.tan(theta_x),
        depth_m * np.tan(theta_y),
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
    Rx = np.array([[1, 0,          0        ],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    pos_world = (Rz @ Ry @ Rx) @ pos_camera
    if camera_world_pos is not None:
        pos_world += camera_world_pos
    return pos_world


# ── Tiling ────────────────────────────────────────────────────────────────────

def get_tiles(frame_rgb, tile_w, tile_h, overlap):
    fh, fw = frame_rgb.shape[:2]
    step_x = tile_w - overlap
    step_y = tile_h - overlap
    tiles  = []
    y = 0
    while y < fh:
        x = 0
        while x < fw:
            x1 = min(x, fw - tile_w)
            y1 = min(y, fh - tile_h)
            tiles.append((frame_rgb[y1:y1 + tile_h, x1:x1 + tile_w], x1, y1))
            if x + tile_w >= fw:
                break
            x += step_x
        if y + tile_h >= fh:
            break
        y += step_y
    return tiles


# ── TRT inference + decode ────────────────────────────────────────────────────

def trt_infer_tile(tile_rgb, x_off, y_off):
    confs, boxes = inferencer.infer(tile_rgb)

    mask = confs > THRESH
    if not mask.any():
        return []

    confs = confs[mask]
    boxes = boxes[mask]

    x1 = boxes[:, 0] * NET_W + x_off
    y1 = boxes[:, 1] * NET_H + y_off
    x2 = boxes[:, 2] * NET_W + x_off
    y2 = boxes[:, 3] * NET_H + y_off

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    hw = (x2 - x1) / 2 * BOX_SCALE
    hh = (y2 - y1) / 2 * BOX_SCALE
    x1 = cx - hw
    x2 = cx + hw
    y1 = cy - hh
    y2 = cy + hh

    label = class_names[0]

    return [(label, float(c), (float(x1[i]), float(y1[i]),
                               float(x2[i]), float(y2[i])))
            for i, c in enumerate(confs)]


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

        indices = cv2.dnn.NMSBoxes(
            cv_boxes,
            scores,
            score_threshold=THRESH,
            nms_threshold=NMS_IOU,
        )

        for i in indices:
            conf, box = items[i]
            kept.append((label, conf, box))

    kept.sort(
        key=lambda d: (d[2][2] - d[2][0]) * (d[2][3] - d[2][1]),
        reverse=True,
    )
    return kept


# ── ObjectDetector ────────────────────────────────────────────────────────────

class ObjectDetector:
    def find_objects(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tiles     = get_tiles(frame_rgb, NET_W, NET_H, OVERLAP)

        futures = [
            _executor.submit(trt_infer_tile, tile_rgb, x_off, y_off)
            for tile_rgb, x_off, y_off in tiles
        ]

        all_dets = []
        for f in futures:
            all_dets.extend(f.result())

        return nms(all_dets)


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
        cam_rgb = self.pipeline.create(dai.node.Camera)
        cam_rgb.build(dai.CameraBoardSocket.CAM_A)
        rgb_out = cam_rgb.requestOutput(
            size=(DISPLAY_W, DISPLAY_H),
            type=dai.ImgFrame.Type.BGR888p,
            fps=30
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
        calib      = self.cam.device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, DISPLAY_W, DISPLAY_H
        )
        self.fx = intrinsics[0][0]
        self.fy = intrinsics[1][1]
        self.cx = intrinsics[0][2]
        self.cy = intrinsics[1][2]
        self.focal_length_px = self.fx
        self.baseline_m      = calib.getBaselineDistance() / 100.0
        print(f"Focal: {self.focal_length_px:.2f}px  Baseline: {self.baseline_m*100:.2f}cm")
        print(f"Principal point: ({self.cx:.1f}, {self.cy:.1f})")

    def update(self, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
               camera_world_pos=None):
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
                pos_cam   = bbox_depth_to_camera_frame(
                    bbox, depth_m, self.fx, self.fy, self.cx, self.cy,
                )
                pos_world = camera_to_world_frame(
                    pos_cam, yaw_deg, pitch_deg, roll_deg, camera_world_pos
                )
            locations.append(ObjectLocation(
                label=label, conf=float(conf), bbox=bbox,
                depth_m=depth_m, pos_camera=pos_cam, pos_world=pos_world,
            ))

        return rgb, locations


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mgr = Manager()
    mgr.start()

    TARGET_LOOP_MS = 20
    frame_count    = 0
    total_ms       = 0.0

    print("Running — press 'q' to quit.")
    profiler = cProfile.Profile()

    try:
        while True:
            profiler.enable()
            loop_start = time.monotonic()

            result = mgr.update(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                                camera_world_pos=np.array([0.0, 0.0, 0.0]))
            if result is None:
                profiler.disable()
                continue

            rgb, locations = result
            elapsed_ms  = (time.monotonic() - loop_start) * 1000
            frame_count += 1
            total_ms    += elapsed_ms

            for loc in locations:
                x1, y1, x2, y2 = loc.bbox
                color     = colours[loc.label]
                depth_str = f"{loc.depth_m:.2f}m" if loc.depth_m > 0 else "N/A"
                cam_str   = (f"cam=({loc.pos_camera[0]:+.2f}, "
                             f"{loc.pos_camera[1]:+.2f}, "
                             f"{loc.pos_camera[2]:.2f})m")

                cv2.rectangle(rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(rgb, f"{loc.label} {loc.conf*100:.0f}% {depth_str}",
                            (int(x1), max(int(y1) - 20, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(rgb, cam_str,
                            (int(x1), max(int(y1) - 4, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

                print(
                    f"{loc.label}: {loc.conf*100:.1f}%  depth={depth_str}  "
                    f"cam=({loc.pos_camera[0]:+.3f}, "
                    f"{loc.pos_camera[1]:+.3f}, "
                    f"{loc.pos_camera[2]:.3f})m"
                    + (f"  world=({loc.pos_world[0]:+.3f}, "
                       f"{loc.pos_world[1]:+.3f}, "
                       f"{loc.pos_world[2]:.3f})m"
                       if loc.pos_world is not None else "")
                )

            if not locations:
                cv2.putText(rgb, "No detections", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if frame_count % 30 == 0:
                avg_ms = total_ms / frame_count
                print(f"[perf] avg loop: {avg_ms:.1f}ms  ({1000/avg_ms:.1f} fps)")

            cv2.putText(rgb, f"{elapsed_ms:.0f}ms",
                        (DISPLAY_W - 80, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.imshow("TRT - OAK-D", rgb)

            profiler.disable()

            remaining_ms = TARGET_LOOP_MS - elapsed_ms
            if remaining_ms > 0:
                time.sleep(remaining_ms / 1000)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        _executor.shutdown(wait=False)
        _cuda_context.pop()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(20)
        cv2.destroyAllWindows()
