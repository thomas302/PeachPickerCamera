#!/usr/bin/env python3
import sys
import time
import numpy as np
import cv2
import cProfile
import pstats
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit          # initialises CUDA context on import
import depthai as dai
from datetime import timedelta
from dataclasses import dataclass

engine_path = "model.engine"
names_file  = "new_sample.names"

DISPLAY_W, DISPLAY_H = 1280, 720
OVERLAP       = 64
NMS_IOU       = 0.3
CENTER_THRESH = 40
THRESH        = 0.5

NET_W = 416
NET_H = 416

# Number of candidate detections from the engine
# 13×13×3 + 26×26×3 + 52×52×3 = 10647
N_ANCHORS = 10647


# ── TensorRT engine + buffers ─────────────────────────────────────────────────

class TRTInferencer:
    """
    Wraps a single TensorRT engine.
    Allocates pinned host memory + GPU memory once at init.
    Call infer(tile_rgb_chw_float32) → (confs, boxes) numpy arrays.
    """
    def __init__(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime    = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        # Allocate pinned host buffers + device buffers
        # Input:  (1, 3, 416, 416) float32
        # confs:  (1, 10647, 1)    float32
        # boxes:  (1, 10647, 1, 4) float32
        self.h_input = cuda.pagelocked_empty((1, 3, NET_H, NET_W),        dtype=np.float32)
        self.h_confs = cuda.pagelocked_empty((1, N_ANCHORS, 1),           dtype=np.float32)
        self.h_boxes = cuda.pagelocked_empty((1, N_ANCHORS, 1, 4),        dtype=np.float32)

        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_confs = cuda.mem_alloc(self.h_confs.nbytes)
        self.d_boxes = cuda.mem_alloc(self.h_boxes.nbytes)

        self.context.set_tensor_address('frame', int(self.d_input))
        self.context.set_tensor_address('confs',  int(self.d_confs))
        self.context.set_tensor_address('boxes',  int(self.d_boxes))

        print(f"✓ TRT engine loaded: {engine_path}")

    def infer(self, tile_rgb_hwc_uint8):
        """
        Args:
            tile_rgb_hwc_uint8: (H, W, 3) uint8 RGB tile, already NET_H×NET_W

        Returns:
            confs: (N_ANCHORS,) float32  — confidence scores
            boxes: (N_ANCHORS, 4) float32 — normalised [x1,y1,x2,y2]
        """
        # Normalise + convert to NCHW float32 into pinned memory
        np.copyto(
            self.h_input,
            (tile_rgb_hwc_uint8.astype(np.float32) / 255.0)
            .transpose(2, 0, 1)        # HWC → CHW
            [np.newaxis],              # add batch dim
        )

        # H2D → inference → D2H, all on the same stream
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_confs, self.d_confs, self.stream)
        cuda.memcpy_dtoh_async(self.h_boxes, self.d_boxes, self.stream)
        self.stream.synchronize()
        

        return (
            self.h_confs[0, :, 0].copy(),          # (N_ANCHORS,)
            self.h_boxes[0, :, 0, :].copy(),        # (N_ANCHORS, 4)
        )

    def __del__(self):
        try:
            self.d_input.free()
            self.d_confs.free()
            self.d_boxes.free()
        except Exception:
            pass


# ── Globals ───────────────────────────────────────────────────────────────────

inferencer  = TRTInferencer(engine_path)
class_names = open(names_file).read().splitlines()
colours     = {name: tuple(int(x) for x in np.random.randint(50, 255, 3))
               for name in class_names}


# ── Localization ──────────────────────────────────────────────────────────────

@dataclass
class ObjectLocation:
    label:      str
    conf:       float
    bbox:       tuple          # (x1, y1, x2, y2) pixels in full frame
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
    """
    BGR→RGB conversion already done upstream.
    Returns [(tile_rgb_hwc, x_off, y_off), ...]
    """
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
    """
    Run TRT inference on one tile.
    Returns [(label, conf, (x1,y1,x2,y2)), ...] in full-frame pixel coords.
    Boxes from engine are normalised 0-1 relative to tile size.
    """
    confs, boxes = inferencer.infer(tile_rgb)

    mask = confs > THRESH
    if not mask.any():
        return []

    confs = confs[mask]
    boxes = boxes[mask]   # (M, 4) normalised x1y1x2y2

    print(f"tile({x_off},{y_off}) {len(confs)} dets:")
    for i in range(min(3, len(confs))):
        print(f"  conf={confs[i]:.3f} box={boxes[i]}")

    # Scale normalised → tile pixel coords → full frame pixel coords
    x1 = boxes[:, 0] * NET_W + x_off
    y1 = boxes[:, 1] * NET_H + y_off
    x2 = boxes[:, 2] * NET_W + x_off
    y2 = boxes[:, 3] * NET_H + y_off

    label = class_names[0]

    return [(label, float(c), (float(x1[i]), float(y1[i]),
                               float(x2[i]), float(y2[i])))
            for i, c in enumerate(confs)]


# ── NMS ───────────────────────────────────────────────────────────────────────

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1  = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2  = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


def is_contained(inner, outer, threshold=0.8):
    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter      = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    inner_area = (ax2 - ax1) * (ay2 - ay1)
    if inner_area == 0:
        return False
    return (inter / inner_area) >= threshold


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
            if any(iou(box, ab) >= iou_thresh or is_contained(box, ab)
                   for _, ab in accepted):
                continue
            accepted.append((conf, box))
            kept.append((label, conf, box))
    return kept


# ── ObjectDetector ────────────────────────────────────────────────────────────

class ObjectDetector:
    def find_objects(self, frame_bgr):
        # BGR→RGB once for all tiles
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        all_dets = []
        for tile_rgb, x_off, y_off in get_tiles(frame_rgb, NET_W, NET_H, OVERLAP):
            all_dets.extend(trt_infer_tile(tile_rgb, x_off, y_off))

        detections = nms(all_dets)
        detections.sort(
            key=lambda d: (d[2][2] - d[2][0]) * (d[2][3] - d[2][1]),
            reverse=True,
        )
        return detections


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
                cv2.putText(rgb, f"{loc.label} {loc.conf:.0f}% {depth_str}",
                            (int(x1), max(int(y1) - 20, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(rgb, cam_str,
                            (int(x1), max(int(y1) - 4, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

                print(
                    f"{loc.label}: {loc.conf*100.0:.1f}%  depth={depth_str}  "
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
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(20)
        cv2.destroyAllWindows()
