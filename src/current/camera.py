import depthai as dai
import numpy as np
from datetime import timedelta

class Camera:
    def __init__(self):
        self.pipeline = dai.Pipeline()
        self.device = self.pipeline.getDefaultDevice()

        print('DeviceID:', self.device.getDeviceInfo().getDeviceId())
        print('USB speed:', self.device.getUsbSpeed())
        print('Connected cameras:', self.device.getConnectedCameras())

        self.q_sync = None
        self.subpixel_bits = 5  # updated: matches firmware default for setSubpixel(True)
        self.configure_pipeline()

    def configure_pipeline(self):
        # --- RGB ---
        cam_rgb = self.pipeline.create(dai.node.Camera)
        cam_rgb.build(dai.CameraBoardSocket.CAM_A)
        rgb_out = cam_rgb.requestOutput(
            size=(1280, 720),
            type=dai.ImgFrame.Type.BGR888p,
            fps=30
        )

        # --- Stereo ---
        cam_left = self.pipeline.create(dai.node.Camera)
        cam_left.build(dai.CameraBoardSocket.CAM_B)

        cam_right = self.pipeline.create(dai.node.Camera)
        cam_right.build(dai.CameraBoardSocket.CAM_C)

        left_out = cam_left.requestOutput(size=(640, 400), fps=30)
        right_out = cam_right.requestOutput(size=(640, 400), fps=30)

        stereo = self.pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setSubpixelFractionalBits(5)  # explicitly set to match subpixel_bits
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(1280, 720)

        left_out.link(stereo.left)
        right_out.link(stereo.right)

        # --- Sync node: temporal alignment ---
        sync = self.pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=50))

        rgb_out.link(sync.inputs["rgb"])
        stereo.disparity.link(sync.inputs["disparity"])

        self.q_sync = sync.out.createOutputQueue(maxSize=4, blocking=False)

    def start(self):
        self.pipeline.start()

    def get_synced(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Returns (rgb, disparity) as a matched pair, or None if not yet available.
        Disparity is spatially aligned to the RGB frame (via setDepthAlign).
        """
        group = self.q_sync.tryGet()
        if group is None:
            return None

        rgb_frame = group["rgb"].getCvFrame()
        disp_frame = group["disparity"].getFrame()

        return rgb_frame, disp_frame

    def get_depth_from_disparity(
        self,
        disparity: np.ndarray,
        focal_length_px: float,
        baseline_m: float,
    ) -> np.ndarray:
        """
        Convert disparity map to metric depth (meters).
        Accounts for subpixel scaling: raw values are multiplied by 2^subpixel_bits on device.
        """
        disp_px = disparity.astype(np.float32) / (2 ** self.subpixel_bits)
        with np.errstate(divide='ignore'):
            depth = np.where(disp_px > 0, (focal_length_px * baseline_m) / disp_px, 0)
        return depth.astype(np.float32)

    def map_bbox_to_depth(
        self,
        bbox: tuple[int, int, int, int],
        rgb_size: tuple[int, int] = (1280, 720),
        disparity_size: tuple[int, int] = (1280, 720)  # matches setOutputSize
    ) -> tuple[int, int, int, int]:
        """
        Maps a bounding box from RGB space to disparity/depth space.
        Since setDepthAlign + setOutputSize(1280,720) makes them the same resolution,
        this is now a 1:1 mapping — but kept general in case sizes differ.
        """
        x1, y1, x2, y2 = bbox

        scale_x = disparity_size[0] / rgb_size[0]
        scale_y = disparity_size[1] / rgb_size[1]

        dx1 = int(x1 * scale_x)
        dy1 = int(y1 * scale_y)
        dx2 = int(x2 * scale_x)
        dy2 = int(y2 * scale_y)

        dx1 = max(0, min(dx1, disparity_size[0] - 1))
        dy1 = max(0, min(dy1, disparity_size[1] - 1))
        dx2 = max(0, min(dx2, disparity_size[0] - 1))
        dy2 = max(0, min(dy2, disparity_size[1] - 1))

        return dx1, dy1, dx2, dy2

    def get_depth_in_bbox(
        self,
        disparity: np.ndarray,
        bbox: tuple[int, int, int, int],
        focal_length_px: float,
        baseline_m: float,
        rgb_size: tuple[int, int] = (1280, 720),
    ) -> float:
        """
        Returns the median metric depth (meters) within a bounding box
        defined in RGB pixel space.
        """
        disp_h, disp_w = disparity.shape[:2]
        dx1, dy1, dx2, dy2 = self.map_bbox_to_depth(
            bbox, rgb_size, (disp_w, disp_h)
        )

        roi = disparity[dy1:dy2, dx1:dx2]
        valid = roi[roi > 0].astype(np.float32)

        if valid.size == 0:
            return 0.0

        # Normalize subpixel disparity to real pixel units before depth formula
        median_disp_px = np.median(valid) / (2 ** self.subpixel_bits)
        return float((focal_length_px * baseline_m) / median_disp_px)


if __name__ == "__main__":
    import cv2
    import time

    cam = Camera()
    cam.start()

    input("Press Enter to continue")

    calib = cam.device.readCalibration()
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1280, 720)
    focal_length_px = intrinsics[0][0]
    baseline_m = calib.getBaselineDistance() / 100.0

    print(f"Raw baseline from calibration: {calib.getBaselineDistance()}")
    print(f"Focal length: {focal_length_px:.2f}px  Baseline: {baseline_m*100:.2f}cm")

    RGB_W, RGB_H = 1280, 720
    box_half = 20
    cx, cy = RGB_W // 2, RGB_H // 2
    center_bbox = (cx - box_half, cy - box_half, cx + box_half, cy + box_half)

    print("Press 'q' to quit.")

    TARGET_LOOP_MS = 20  # ~50fps cap
    while True:
        loop_start = time.monotonic()

        result = cam.get_synced()
        if result is None:
            continue

        rgb, disparity = result

        depth_m = cam.get_depth_in_bbox(
            disparity, center_bbox, focal_length_px, baseline_m
        )

        cv2.rectangle(rgb, (center_bbox[0], center_bbox[1]), (center_bbox[2], center_bbox[3]), (0, 255, 0), 2)
        cv2.putText(
            rgb,
            f"{depth_m:.2f}m" if depth_m > 0 else "N/A",
            (center_bbox[0], center_bbox[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        cv2.imshow("OAK-D Lite - Center Depth", rgb)

        elapsed_ms = (time.monotonic() - loop_start) * 1000
        remaining_ms = TARGET_LOOP_MS - elapsed_ms
        if remaining_ms > 0:
            time.sleep(remaining_ms / 1000)
        else:
            print(f"Loop overrun: {-remaining_ms:.1f}ms")

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
