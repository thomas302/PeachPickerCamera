#!/usr/bin/env python3
import sys
import time
import multiprocessing as mp
import multiprocessing.shared_memory as shm
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
OVERLAP       = 64
NMS_IOU       = 0.3
CENTER_THRESH = 40
THRESH        = 0.4
WORKER_COUNT  = 2

FRAME_SHAPE       = (DISPLAY_H, DISPLAY_W, 3)
FRAME_DTYPE       = np.uint8
FRAME_BYTES       = int(np.prod(FRAME_SHAPE)) * np.dtype(FRAME_DTYPE).itemsize

MAX_DETS_PER_TILE = 64
FLOATS_PER_DET    = 6
SLOT_FLOATS       = 1 + MAX_DETS_PER_TILE * FLOATS_PER_DET


def slot_base(tile_idx):
    return tile_idx * SLOT_FLOATS


def result_buffer_size(n_tiles):
    return n_tiles * SLOT_FLOATS * 4


def write_slot(shared_results, tile_idx, detections, class_names):
    base  = slot_base(tile_idx)
    count = min(len(detections), MAX_DETS_PER_TILE)
    shared_results[base] = float(count)
    for i, (label, conf, (x1, y1, x2, y2)) in enumerate(detections[:count]):
        off = base + 1 + i * FLOATS_PER_DET
        shared_results[off + 0] = x1
        shared_results[off + 1] = y1
        shared_results[off + 2] = x2
        shared_results[off + 3] = y2
        shared_results[off + 4] = conf
        shared_results[off + 5] = float(class_names.index(label))


def read_all_slots(shared_results, class_names, n_tiles):
    all_dets = []
    for tile_idx in range(n_tiles):
        base  = slot_base(tile_idx)
        count = int(shared_results[base])
        for i in range(count):
            off   = base + 1 + i * FLOATS_PER_DET
            x1    = float(shared_results[off + 0])
            y1    = float(shared_results[off + 1])
            x2    = float(shared_results[off + 2])
            y2    = float(shared_results[off + 3])
            conf  = float(shared_results[off + 4])
            label = class_names[int(shared_results[off + 5])]
            all_dets.append((label, conf, (x1, y1, x2, y2)))
    return all_dets


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


# ── Tiling + NMS ──────────────────────────────────────────────────────────────

def get_tile_meta(frame_w, frame_h, tile_w, tile_h, overlap):
    step_x = tile_w - overlap
    step_y = tile_h - overlap
    meta   = []
    y = 0
    while y < frame_h:
        x = 0
        while x < frame_w:
            x1 = min(x, frame_w - tile_w)
            y1 = min(y, frame_h - tile_h)
            meta.append((x1, y1))
            if x + tile_w >= frame_w:
                break
            x += step_x
        if y + tile_h >= frame_h:
            break
        y += step_y
    return meta


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


# ── Detection worker process ──────────────────────────────────────────────────

def detection_worker(
    worker_id,
    frame_shm_name,
    result_shm_name,
    result_size,
    tile_ready,
    tile_done,
    tile_idx_val,
    x_off_val,
    y_off_val,
    stop_event,
):
    from multiprocessing.shared_memory import SharedMemory
    import darknet as dn
    import numpy as np
    import time

    f_shm = SharedMemory(name=frame_shm_name)
    r_shm = SharedMemory(name=result_shm_name)

    frame          = np.ndarray(FRAME_SHAPE,       dtype=np.uint8,   buffer=f_shm.buf)
    shared_results = np.ndarray((result_size//4,),  dtype=np.float32, buffer=r_shm.buf)

    t_load = time.monotonic()
    network   = dn.load_net_custom(cfg_file.encode(), weights_file.encode(), 0, 1)
    cls_names = open(names_file).read().splitlines()
    net_w     = dn.network_width(network)
    net_h     = dn.network_height(network)
    print(f"[worker {worker_id}] ready — model load {(time.monotonic()-t_load)*1000:.0f}ms",
          flush=True)

    tile_count = 0

    while not stop_event.is_set():
        t_wait = time.monotonic()
        tile_ready.wait()
        tile_ready.clear()
        if stop_event.is_set():
            break

        t_got  = time.monotonic()
        tile_idx = tile_idx_val.value
        x_off    = x_off_val.value
        y_off    = y_off_val.value

        # How long did we sit waiting for a tile?
        wait_ms = (t_got - t_wait) * 1000

        t0 = time.monotonic()
        tile_rgb = frame[y_off:y_off + net_h, x_off:x_off + net_w].copy()
        t1 = time.monotonic()

        dn_img = dn.make_image(net_w, net_h, 3)
        dn.copy_image_from_bytes(dn_img, tile_rgb.tobytes())
        t2 = time.monotonic()

        dets = dn.detect_image(network, cls_names, dn_img, thresh=THRESH)
        t3 = time.monotonic()

        dn.free_image(dn_img)
        abs_dets = [(l, c, to_abs_bbox(b, x_off, y_off)) for l, c, b in dets]
        write_slot(shared_results, tile_idx, abs_dets, cls_names)
        t4 = time.monotonic()

        tile_done.set()

        tile_count += 1
        print(
            f"[w{worker_id}|t{tile_idx}] "
            f"waited={wait_ms:.1f}ms "
            f"copy={1000*(t1-t0):.1f}ms "
            f"prep={1000*(t2-t1):.1f}ms "
            f"infer={1000*(t3-t2):.1f}ms "
            f"write={1000*(t4-t3):.1f}ms "
            f"total={1000*(t4-t0):.1f}ms",
            flush=True
        )

    dn.free_network_ptr(network)
    f_shm.close()
    r_shm.close()
    print(f"[worker {worker_id}] shutdown after {tile_count} tiles", flush=True)


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


# ── ObjectDetector ────────────────────────────────────────────────────────────

class ObjectDetector:
    def __init__(self):
        self.tiles_meta = get_tile_meta(DISPLAY_W, DISPLAY_H, 416, 416, OVERLAP)
        self.n_tiles    = len(self.tiles_meta)
        print(f"Tile count: {self.n_tiles}")

        res_bytes       = result_buffer_size(self.n_tiles)
        self.frame_shm  = shm.SharedMemory(create=True, size=FRAME_BYTES)
        self.result_shm = shm.SharedMemory(create=True, size=res_bytes)

        self.shared_frame   = np.ndarray(FRAME_SHAPE,      dtype=np.uint8,
                                          buffer=self.frame_shm.buf)
        self.shared_results = np.ndarray((res_bytes // 4,), dtype=np.float32,
                                          buffer=self.result_shm.buf)

        self.tile_ready = [mp.Event()       for _ in range(WORKER_COUNT)]
        self.tile_done  = [mp.Event()       for _ in range(WORKER_COUNT)]
        self.tile_idx_v = [mp.Value('i', 0) for _ in range(WORKER_COUNT)]
        self.x_off_v    = [mp.Value('i', 0) for _ in range(WORKER_COUNT)]
        self.y_off_v    = [mp.Value('i', 0) for _ in range(WORKER_COUNT)]
        self.stop_event = mp.Event()

        self.workers = [
            mp.Process(
                target=detection_worker,
                args=(
                    i,
                    self.frame_shm.name,
                    self.result_shm.name,
                    res_bytes,
                    self.tile_ready[i],
                    self.tile_done[i],
                    self.tile_idx_v[i],
                    self.x_off_v[i],
                    self.y_off_v[i],
                    self.stop_event,
                ),
                daemon=True,
            )
            for i in range(WORKER_COUNT)
        ]
        for w in self.workers:
            w.start()

        self.class_names = open(names_file).read().splitlines()

    def find_objects(self, frame_bgr):
        t0 = time.monotonic()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        np.copyto(self.shared_frame, frame_rgb)

        t1 = time.monotonic()

        tile_queue  = list(range(self.n_tiles))
        worker_tile = [None] * WORKER_COUNT

        def dispatch(w_idx):
            if not tile_queue:
                return
            t            = tile_queue.pop(0)
            x_off, y_off = self.tiles_meta[t]
            self.tile_idx_v[w_idx].value = t
            self.x_off_v[w_idx].value    = x_off
            self.y_off_v[w_idx].value    = y_off
            self.tile_done[w_idx].clear()
            worker_tile[w_idx] = t
            self.tile_ready[w_idx].set()

        # Initial dispatch — both workers get their first tile simultaneously
        t_dispatch_start = time.monotonic()
        for w_idx in range(WORKER_COUNT):
            dispatch(w_idx)

        # Drain
        while any(wt is not None for wt in worker_tile):
            for w_idx in range(WORKER_COUNT):
                if worker_tile[w_idx] is None:
                    continue
                if self.tile_done[w_idx].is_set():
                    worker_tile[w_idx] = None
                    dispatch(w_idx)

        t2 = time.monotonic()

        all_dets   = read_all_slots(self.shared_results, self.class_names, self.n_tiles)
        detections = nms(all_dets)
        detections.sort(
            key=lambda d: (d[2][2] - d[2][0]) * (d[2][3] - d[2][1]),
            reverse=True,
        )

        t3 = time.monotonic()

        print(
            f"[find_objects] "
            f"cvt+copy={1000*(t1-t0):.1f}ms "
            f"dispatch+wait={1000*(t2-t1):.1f}ms "
            f"nms={1000*(t3-t2):.1f}ms "
            f"total={1000*(t3-t0):.1f}ms",
            flush=True
        )

        return detections

    def shutdown(self):
        self.stop_event.set()
        for e in self.tile_ready:
            e.set()
        for w in self.workers:
            w.join(timeout=3)
        self.frame_shm.close();  self.frame_shm.unlink()
        self.result_shm.close(); self.result_shm.unlink()


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

    def shutdown(self):
        self.detector.shutdown()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn")

    mgr     = Manager()
    colours = darknet.class_colors(open(names_file).read().splitlines())
    mgr.start()

    TARGET_LOOP_MS = 20
    frame_count    = 0
    total_ms       = 0.0

    print("Running — press 'q' to quit.")

    try:
        while True:
            loop_start = time.monotonic()

            result = mgr.update(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                                camera_world_pos=np.array([0.0, 0.0, 0.0]))
            if result is None:
                continue

            rgb, locations = result
            elapsed_ms  = (time.monotonic() - loop_start) * 1000
            frame_count += 1
            total_ms    += elapsed_ms

            for loc in locations:
                print(
                    f"{loc.label}: {loc.conf:.1f}%  "
                    f"depth={'N/A' if loc.depth_m <= 0 else f'{loc.depth_m:.2f}m'}  "
                    f"cam=({loc.pos_camera[0]:+.3f}, "
                    f"{loc.pos_camera[1]:+.3f}, "
                    f"{loc.pos_camera[2]:.3f})m"
                    + (f"  world=({loc.pos_world[0]:+.3f}, "
                       f"{loc.pos_world[1]:+.3f}, "
                       f"{loc.pos_world[2]:.3f})m"
                       if loc.pos_world is not None else "")
                )

            if frame_count % 30 == 0:
                avg_ms = total_ms / frame_count
                print(f"[perf] avg loop: {avg_ms:.1f}ms  ({1000/avg_ms:.1f} fps)",
                      flush=True)

            print(f"[main] frame={frame_count} elapsed={elapsed_ms:.1f}ms",
                  flush=True)

            remaining_ms = TARGET_LOOP_MS - elapsed_ms
            if remaining_ms > 0:
                time.sleep(remaining_ms / 1000)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        mgr.shutdown()
        cv2.destroyAllWindows()
