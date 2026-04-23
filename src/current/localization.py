import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CameraIntrinsics:
    """
    Supports both intrinsic matrix style and FOV style.
    For OAK-D Lite, you can get fx/fy/cx/cy directly from the SDK:

        calibData = device.readCalibration()
        intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB)
        fx, fy = intrinsics[0][0], intrinsics[1][1]
        cx, cy = intrinsics[0][2], intrinsics[1][2]

    Or construct from FOV if you only have that:
        CameraIntrinsics.from_fov(hfov=69.0, vfov=54.0, width=1920, height=1080)
    """
    fx: float  # horizontal focal length in pixels
    fy: float  # vertical focal length in pixels
    cx: float  # principal point x (usually image_width / 2)
    cy: float  # principal point y (usually image_height / 2)

    @classmethod
    def from_fov(cls, hfov: float, vfov: float, width: int, height: int) -> "CameraIntrinsics":
        """Build intrinsics from field of view angles (in degrees) and image size."""
        fx = (width / 2) / np.tan(np.radians(hfov / 2))
        fy = (height / 2) / np.tan(np.radians(vfov / 2))
        return cls(fx=fx, fy=fy, cx=width / 2, cy=height / 2)


def bbox_depth_to_camera_frame(
    bbox_center: Tuple[float, float],
    depth: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    Convert a bounding box center pixel + planar depth to a 3D point
    in the camera frame.

    Camera frame convention (standard):
        +X = right
        +Y = down
        +Z = forward (into scene)

    The depth here is PLANAR — it is the perpendicular distance from
    the camera to the depth plane, NOT the true range along the ray.
    This is what the OAK-D reports for a given pixel.

    Args:
        bbox_center:  (px, py) pixel coordinates of the bounding box center
        depth:        planar depth in metres (or whatever unit you use)
        intrinsics:   CameraIntrinsics for the camera

    Returns:
        np.ndarray shape (3,): [x, y, z] in camera frame, same units as depth
    """
    px, py = bbox_center

    # Pixel offset from principal point
    dx = px - intrinsics.cx
    dy = py - intrinsics.cy

    # Angular offset from optical axis
    theta_x = np.arctan2(dx, intrinsics.fx)  # horizontal angle (+ = right)
    theta_y = np.arctan2(dy, intrinsics.fy)  # vertical angle   (+ = down)

    # Project onto the depth plane using planar depth D
    # x = D * tan(theta_x),  y = D * tan(theta_y),  z = D
    x = depth * np.tan(theta_x)
    y = depth * np.tan(theta_y)
    z = depth

    return np.array([x, y, z], dtype=float)


def camera_to_world_frame(
    point_camera: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    camera_position: np.ndarray | None = None,
) -> np.ndarray:
    """
    Rotate a point from camera frame into world frame using
    yaw / pitch / roll (in degrees, extrinsic ZYX convention).

    Args:
        point_camera:     (3,) point in camera frame
        yaw:              rotation about world Z axis (degrees, + = left)
        pitch:            rotation about world Y axis (degrees, + = nose up)
        roll:             rotation about world X axis (degrees, + = right wing down)
        camera_position:  (3,) camera origin in world frame. If None, origin is (0,0,0).

    Returns:
        np.ndarray shape (3,): point in world frame
    """
    y = np.radians(yaw)
    p = np.radians(pitch)
    r = np.radians(roll)

    # Rotation matrices (extrinsic, applied Z -> Y -> X)
    Rz = np.array([
        [ np.cos(y), -np.sin(y), 0],
        [ np.sin(y),  np.cos(y), 0],
        [0,           0,         1],
    ])
    Ry = np.array([
        [ np.cos(p), 0, np.sin(p)],
        [0,          1, 0        ],
        [-np.sin(p), 0, np.cos(p)],
    ])
    Rx = np.array([
        [1, 0,          0         ],
        [0, np.cos(r), -np.sin(r) ],
        [0, np.sin(r),  np.cos(r) ],
    ])

    R = Rz @ Ry @ Rx
    point_world = R @ point_camera

    if camera_position is not None:
        point_world = point_world + camera_position

    return point_world


# --- Example usage ---
if __name__ == "__main__":
    # OAK-D Lite RGB camera: ~69 deg hfov, ~54 deg vfov at 1920x1080
    # Swap for real intrinsics from calibData if you have them
    intrinsics = CameraIntrinsics.from_fov(
        hfov=69.0, vfov=54.0, width=1920, height=1080
    )

    # Bounding box center in pixels, depth in metres
    bbox_center = (1100, 480)   # slightly right and above image center
    depth = 3.5                 # metres

    pos_cam = bbox_depth_to_camera_frame(bbox_center, depth, intrinsics)
    print(f"Camera frame:  x={pos_cam[0]:.3f}m  y={pos_cam[1]:.3f}m  z={pos_cam[2]:.3f}m")

    # When you have orientation — plug yaw/pitch/roll in here
    pos_world = camera_to_world_frame(
        pos_cam,
        yaw=15.0,
        pitch=5.0,
        roll=0.0,
        camera_position=np.array([0.0, 1.5, 0.0]),  # camera is 1.5m off ground
    )
    print(f"World frame:   x={pos_world[0]:.3f}m  y={pos_world[1]:.3f}m  z={pos_world[2]:.3f}m")
