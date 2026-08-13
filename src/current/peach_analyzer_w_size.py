import os
import json
import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
import sys
from pathlib import Path

sys.path.append('/home/peach/src/darknet/src-python')
import darknet as dn

# === CONFIG ===
SESSION_DIR = "/home/peach/Desktop/Camera/src/current/dataset_captures/session_20260806_174821/"  # Set this to your session folder
cfg_file = "new_sample.cfg"
names_file = "new_sample.names"
weights_file = "new_sample.weights"
THRESH = 0.3

def filter_sphere_outliers(points_3d, center, radius, threshold=0.02):
    """Keep only points within threshold meters of sphere surface."""
    dists = np.linalg.norm(points_3d - center, axis=1)
    residuals = np.abs(dists - radius)
    mask = residuals < threshold
    return points_3d[mask]

# === SPHERE FITTING ===
def fit_sphere_algebraic(points):
    """Fit a sphere to exactly 4 points using an algebraic method."""
    A = np.c_[2 * points, np.ones(4)]
    b = np.sum(points**2, axis=1)
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        center = sol[:3]
        radius = np.sqrt(sol[3] + np.sum(center**2))
        return center, radius
    except np.linalg.LinAlgError:
        return None, None

def fit_sphere(points, n=10, strategy="median", threshold=0.01):
    """
    RANSAC-like sphere fitting sampling 4 points n times.
    Strategies: 'median', 'mean', 'geometric_mean', 'trimmed_mean', 'inlier_consensus'
    """
    points = np.asarray(points)
    if len(points) < 4:
        return None, None

    centers = []
    radii = []
    scores = []

    for _ in range(n):
        sample_idx = np.random.choice(len(points), 4, replace=False)
        center, radius = fit_sphere_algebraic(points[sample_idx])
        
        if center is not None and not np.isnan(radius) and radius > 0:
            centers.append(center)
            radii.append(radius)
            
            if strategy == "inlier_consensus":
                dists = np.abs(np.linalg.norm(points - center, axis=1) - radius)
                inliers = np.sum(dists < threshold)
                scores.append(inliers)

    if not radii:
        return None, None

    centers = np.array(centers)
    radii = np.array(radii)

    if strategy == "median":
        return np.median(centers, axis=0), np.median(radii)
    elif strategy == "mean":
        return np.mean(centers, axis=0), np.mean(radii)
    elif strategy == "geometric_mean":
        return np.mean(centers, axis=0), np.exp(np.mean(np.log(radii)))
    elif strategy == "trimmed_mean":
        # Discard top and bottom 10% of radii to handle degenerate 4-point samples
        lower, upper = np.percentile(radii, [10, 90])
        mask = (radii >= lower) & (radii <= upper)
        if not np.any(mask):
            return np.median(centers, axis=0), np.median(radii)
        return np.mean(centers[mask], axis=0), np.mean(radii[mask])
    elif strategy == "inlier_consensus":
        best_idx = np.argmax(scores)
        best_center = centers[best_idx]
        best_radius = radii[best_idx]
        
        # Optional: Refine using all inliers of the best model
        dists = np.abs(np.linalg.norm(points - best_center, axis=1) - best_radius)
        inlier_points = points[dists < threshold]
        if len(inlier_points) >= 4:
            # Fall back to your original least-squares refinement on inliers
            center_init = inlier_points.mean(axis=0)
            def residuals(p):
                return np.linalg.norm(inlier_points - p[:3], axis=1) - p[3]
            p0 = np.append(center_init, np.linalg.norm(inlier_points - center_init).mean())
            from scipy.optimize import least_squares
            res = least_squares(residuals, p0)
            return res.x[:3], res.x[3]
            
        return best_center, best_radius
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
def find_tip_blob_detector_clahe(rgb, mask, disparity, intrinsics, frame_idx=0, debug_dir="./debug/"):
    """
    Find tip by detecting suture lines converging at tip using CLAHE + blob detection.
    Saves intermediate steps for debugging.
    
    Args:
        rgb: RGB image
        mask: Binary mask of peach
        disparity: Disparity map (raw, subpixel units)
        intrinsics: Camera intrinsics dict
        frame_idx: Frame index for naming
        debug_dir: Directory to save debug images (if None, skips saving)
    
    Returns:
        dict with '3d_point', 'blob_center_2d', 'debug_info', or None if failed
    """
    h, w = mask.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Crop to peach region
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) == 0:
        print(f"[Frame {frame_idx}] No mask pixels found")
        return None
    
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()
    crop = gray[y_min:y_max+1, x_min:x_max+1]
    crop_mask = mask[y_min:y_max+1, x_min:x_max+1]
    
    if debug_dir:
        cv2.imwrite(f"{debug_dir}/frame_{frame_idx:06d}_01_crop.png", crop)
        cv2.imwrite(f"{debug_dir}/frame_{frame_idx:06d}_02_crop_mask.png", crop_mask * 255)
    
    # CLAHE: enhance subtle suture lines
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(crop)
    enhanced[crop_mask == 0] = 0
    
    if debug_dir:
        cv2.imwrite(f"{debug_dir}/frame_{frame_idx:06d}_03_clahe_enhanced.png", enhanced)
    
    # Configure SimpleBlobDetector for small, dark blobs
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = crop.size // 20
    params.filterByColor = True
    params.blobColor = 0  # Dark blobs
    params.filterByConvexity = True
    params.minConvexity = 0.7
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(enhanced)
    
    if debug_dir:
        kp_vis = cv2.drawKeypoints(enhanced, keypoints, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(f"{debug_dir}/frame_{frame_idx:06d}_04_keypoints.png", kp_vis)
        print(f"[Frame {frame_idx}] Detected {len(keypoints)} keypoints")
        for i, kp in enumerate(keypoints):
            print(f"  Keypoint {i}: pos={kp.pt}, size={kp.size:.2f}")
    
    if not keypoints:
        print(f"[Frame {frame_idx}] No keypoints detected")
        return None
    
    # Use smallest blob
    blob = min(keypoints, key=lambda kp: kp.size)
    blob_u = int(blob.pt[0]) + x_min
    blob_v = int(blob.pt[1]) + y_min
    blob_u = np.clip(blob_u, 0, w - 1)
    blob_v = np.clip(blob_v, 0, h - 1)
    
    if debug_dir:
        vis = rgb.copy()
        cv2.circle(vis, (blob_u, blob_v), 8, (255, 0, 0), 2)
        cv2.imwrite(f"{debug_dir}/frame_{frame_idx:06d}_05_selected_blob.png", 
                   cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"[Frame {frame_idx}] Selected blob at ({blob_u}, {blob_v})")
    
    # Disparity -> depth
    disparity_px = disparity[blob_v, blob_u].astype(np.float32) / 32.0
    if disparity_px <= 0:
        print(f"[Frame {frame_idx}] Invalid disparity at blob: {disparity_px}")
        return None
    
    fx = intrinsics.get('fx', 903.0)
    fy = intrinsics.get('fy', 903.0)
    cx = intrinsics.get('cx', w / 2)
    cy = intrinsics.get('cy', h / 2)
    baseline = intrinsics.get('baseline_m', 0.075)
    
    z = (fx * baseline) / disparity_px
    x = (blob_u - cx) * z / fx
    y = (blob_v - cy) * z / fy
    
    return  np.array([x, y, z])
    '''
    {
        '3d_point': np.array([x, y, z]),
        'blob_center_2d': (blob_u, blob_v),
        'debug_info': {
            'disparity_px': disparity_px,
            'keypoint_count': len(keypoints),
            'blob_size': blob.size
        }
    }
    '''
        
def find_tip_directional_outlier_robust(points_3d, center, radius):
    """Find tip as the point that deviates most from sphere surface."""
    dists_to_center = np.linalg.norm(points_3d - center, axis=1)
    residuals = dists_to_center - radius  # positive = outside
    
    # Find point with max positive deviation (furthest from surface outward)
    tip_idx = np.argmax(residuals)
    return points_3d[tip_idx]
        
## TIP FINDING
def find_tip_pca(points_3d):
    """Original strategy: extreme point on primary axis."""
    if len(points_3d) < 3:
        return None
    
    centroid = points_3d.mean(axis=0)
    centered = points_3d - centroid
    _, _, vh = np.linalg.svd(centered)
    primary_axis = vh[0]
    projections = np.dot(centered, primary_axis)
    tip_idx = np.argmax(projections)
    return points_3d[tip_idx]
 
def find_tip_convex_hull(points_3d, center):
    """Find tip as furthest convex hull vertex from center."""
    from scipy.spatial import ConvexHull
    
    if len(points_3d) < 4:
        return None
    
    try:
        hull = ConvexHull(points_3d)
        hull_verts = points_3d[hull.vertices]
        dists = np.linalg.norm(hull_verts - center, axis=1)
        tip_idx_in_hull = np.argmax(dists)
        return hull_verts[tip_idx_in_hull]
    except:
        return None
 
def find_tip_directional_outlier(points_3d, center, radius, direction='neg_z'):
    """Find point with max surface outlier in specified direction."""
    # Compute residuals: distance from each point to sphere surface
    dists_to_center = np.linalg.norm(points_3d - center, axis=1)
    residuals = dists_to_center - radius  # positive = outside, negative = inside
    
    if direction == 'neg_z':
        # Favor points in -Z (down in camera frame)
        directional_score = residuals + (center[2] - points_3d[:, 2]) * 0.1
    elif direction == 'pos_z':
        directional_score = residuals + (points_3d[:, 2] - center[2]) * 0.1
    else:
        directional_score = residuals
    
    tip_idx = np.argmax(directional_score)
    return points_3d[tip_idx]
 
def find_tip_curvature(points_3d, center, k=20):
    """Find point with highest local surface curvature."""
    if len(points_3d) < k + 1:
        k = max(5, len(points_3d) // 4)
    
    from scipy.spatial import cKDTree
    
    tree = cKDTree(points_3d)
    curvatures = []
    
    for i, point in enumerate(points_3d):
        # Find k nearest neighbors
        _, indices = tree.query(point, k=k+1)
        neighbors = points_3d[indices[1:]]  # Exclude self
        
        # Local PCA to estimate surface curvature
        centered = neighbors - neighbors.mean(axis=0)
        if len(centered) >= 2:
            _, s, _ = np.linalg.svd(centered)
            # Curvature proxy: ratio of smallest to largest singular value
            # High curvature = small singular value (sharp peak)
            curvature = s[-1] / (s[0] + 1e-6) if s[0] > 0 else 0
        else:
            curvature = 0
        curvatures.append(curvature)
    
    tip_idx = np.argmax(curvatures)
    return points_3d[tip_idx]


def predict_tip_cnn(clahe_crop, model):
    """
    Predict tip location in crop coordinates.
    
    Args:
        clahe_crop: CLAHE-enhanced crop
        model: Trained TipDetectorCNN
    
    Returns:
        (tip_x_crop, tip_y_crop) in crop coordinates, or None
    """
    crop_h, crop_w = clahe_crop.shape
    
    # 1. Resize to match the 125x125 training input size
    target_size = 125
    clahe_resized = cv2.resize(clahe_crop, (target_size, target_size))
    
    img_tensor = torch.from_numpy(clahe_resized).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)[0].cpu().numpy()
    
    # 2. Scale predictions back from the 125x125 model space to the original crop dimensions
    tip_x_crop = int(round(pred[0] * crop_w / target_size))
    tip_y_crop = int(round(pred[1] * crop_h / target_size))
    
    return (tip_x_crop, tip_y_crop)

def tip_2d_to_3d(tip_x_crop, tip_y_crop, disparity_crop, intrinsics, x_off, y_off, img_h, img_w):
    """
    Convert 2D tip location to 3D using disparity.
    
    Args:
        tip_x_crop, tip_y_crop: Tip location in crop coordinates
        disparity_crop: Disparity map in crop coordinates
        intrinsics: Camera intrinsics dict
        x_off, y_off: Crop offset in full image
        img_h, img_w: Full image dimensions
    
    Returns:
        3D point as numpy array, or None
    """
    # Ensure coordinates fall safely inside the disparity crop bounds
    ch, cw = disparity_crop.shape
    tip_x_crop = np.clip(tip_x_crop, 0, cw - 1)
    tip_y_crop = np.clip(tip_y_crop, 0, ch - 1)
    
    # Map to full image
    tip_u = np.clip(tip_x_crop + x_off, 0, img_w - 1)
    tip_v = np.clip(tip_y_crop + y_off, 0, img_h - 1)
    
    # Disparity -> 3D
    disparity_px = disparity_crop[tip_y_crop, tip_x_crop].astype(np.float32) / 32.0
    if disparity_px <= 0:
        return None
    
    fx = intrinsics.get('fx', 903.0)
    fy = intrinsics.get('fy', 903.0)
    cx = intrinsics.get('cx', img_w / 2)
    cy = intrinsics.get('cy', img_h / 2)
    baseline = intrinsics.get('baseline_m', 0.075)
    
    z = (fx * baseline) / disparity_px
    x = (tip_u - cx) * z / fx
    y = (tip_v - cy) * z / fy
    
    return np.array([x, y, z])

def find_tip(points_3d, center, radius, strategy='convex_hull', model=None):
    """
    Find tip point using specified strategy.
    
    Strategies:
      - 'pca': Extreme projection on primary axis
      - 'convex_hull': Furthest vertex of convex hull from center
      - 'directional_outlier': Point with max surface deviation in -Z direction (camera frame)
      - 'curvature': Point with highest local surface curvature
    """
    if strategy == 'pca':
        return find_tip_pca(points_3d)
    
    elif strategy == 'convex_hull':
        return find_tip_convex_hull(points_3d, center)
    
    elif strategy == 'directional_outlier':
        return find_tip_directional_outlier_robust(points_3d, center, radius)
    
    elif strategy == 'curvature':
        return find_tip_curvature(points_3d, center)
    elif strategy == 'nn':
        if model == None:
            raise ValueError("No Tip Detection Model Provided")
            
        tip_2d = predict_tip_cnn(clahe_crop, model)
        tip_2d_to_3d(*tip_2, disparity_crop, intrinsics, x_off, y_off, img_h, img_w)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# === PROJECTION & OVERLAY ===
def project_to_image(point_3d, intrinsics, img_h, img_w):
    """Project 3D point to image space."""
    fx = intrinsics.get('fx', intrinsics.get('fx_', 1.0))
    fy = intrinsics.get('fy', intrinsics.get('fy_', 1.0))
    cx = intrinsics.get('cx', intrinsics.get('cx_', img_w/2))
    cy = intrinsics.get('cy', intrinsics.get('cy_', img_h/2))
    
    if point_3d[2] <= 0:
        return None
    
    u = int(fx * point_3d[0] / point_3d[2] + cx)
    v = int(fy * point_3d[1] / point_3d[2] + cy)
    
    if 0 <= u < img_w and 0 <= v < img_h:
        return (u, v)
    return None


def draw_overlay(rgb, center=None, radius=None, tip=None, axis_unit=None, intrinsics=None, points_3d=None, bbox=None, conf=None):
    """Draw optional sphere center, tip, axis, points, radius, and bbox on image."""
    h, w = rgb.shape[:2]
    overlay = rgb.copy()
    
    # 1. Draw detection bbox and confidence (if provided)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if conf is not None:
            cv2.putText(overlay, f"Peach {float(conf):.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 2. Draw 3D points (if provided and intrinsics available)
    if points_3d is not None and intrinsics is not None:
        for point in points_3d:
            proj = project_to_image(point, intrinsics, h, w)
            if proj:
                cv2.circle(overlay, proj, 2, (0, 100, 0), -1)
    
    # 3. Project and draw center (if provided and intrinsics available)
    center_proj = None
    if center is not None and intrinsics is not None:
        center_proj = project_to_image(center, intrinsics, h, w)
        if center_proj:
            cv2.circle(overlay, center_proj, 8, (255, 0, 0), -1)
            cv2.putText(overlay, f"C({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})",
                       (center_proj[0] + 10, center_proj[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # 4. Project and draw tip (if provided and intrinsics available)
    tip_proj = None
    if tip is not None and intrinsics is not None:
        tip_proj = project_to_image(tip, intrinsics, h, w)
        if tip_proj:
            cv2.circle(overlay, tip_proj, 6, (0, 0, 255), -1)
            cv2.putText(overlay, "Tip", (tip_proj[0] + 10, tip_proj[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # 5. Draw axis line from tip towards center (if both projections exist)
    if center_proj is not None and tip_proj is not None:
        cv2.arrowedLine(overlay, tip_proj, center_proj, (255, 255, 0), 2, tipLength=0.3)
    
    # 6. Draw radius in top right corner (if provided)
    if radius is not None:
        radius_text = f"R: {radius:.4f}m"
        text_size = cv2.getTextSize(radius_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = w - text_size[0] - 10
        text_y = 30
        cv2.rectangle(overlay, (text_x - 5, text_y - text_size[1] - 5), 
                      (w - 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(overlay, radius_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return overlay


# === MAIN ===
def process_session():
    metadata_file = os.path.join(SESSION_DIR, "metadata.jsonl")
    
    if not os.path.exists(metadata_file):
        print(f"Metadata not found: {metadata_file}")
        return
    
    network = dn.load_net_custom(cfg_file.encode(), weights_file.encode(), 0, 1)
    cls_names = open(names_file).read().splitlines()
    net_w, net_h = dn.network_width(network), dn.network_height(network)
    
    ## Tip NN
    # Load Tip Detection Model
    tip_model = torch.load("./tip_detector.pth")
    tip_model.eval()
    
    # Create output directories
    overlay_dir = os.path.join(SESSION_DIR, "overlay_frames")
    os.makedirs(overlay_dir, exist_ok=True)
    
    results = []
    frame_list = []
    
    with open(metadata_file) as f:
        for line in f:
            record = json.loads(line)
            frame_idx = record["frame_idx"]
            rgb_path = os.path.join(SESSION_DIR, record["rgb_file"])
            disp_path = os.path.join(SESSION_DIR, record["disparity_file"])
            intrinsics = record["intrinsics"]
            
            # Load RGB & disparity
            rgb = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            disp_data = np.load(disp_path)
            disparity = disp_data['disparity']
            
            h, w = rgb.shape[:2]
            
            # === DARKNET DETECTION ===
            rgb_resized = cv2.resize(rgb, (net_w, net_h))
            dn_img = dn.make_image(net_w, net_h, 3)
            dn.copy_image_from_bytes(dn_img, cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR).tobytes())
            dets = dn.detect_image(network, cls_names, dn_img, thresh=THRESH)
            dn.free_image(dn_img)
            
            if not dets:
                print(f"Frame {frame_idx}: No detections")
                continue
            
            # Get peach detection (assume first/highest conf)
            label, conf, (x, y, bw, bh) = max(dets, key=lambda d: d[1])
            x1 = max(0, int((x - bw/2) * w / net_w))
            y1 = max(0, int((y - bh/2) * h / net_h))
            x2 = min(w, int((x + bw/2) * w / net_w))
            y2 = min(h, int((y + bh/2) * h / net_h))
            
            # === GRABCUT ===
            mask = np.zeros((h, w), np.uint8)
            # Fixed: Changed cv2.cv2 to cv2 AND added 'D' to the end of GC_PR_FGD
            mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(rgb, mask, (x1, y1, x2-x1, y2-y1), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
            
            # === 3D POINTS FROM DISPARITY ===
            # OAK-D disparity is in subpixel units (32x precision), convert to pixels
            disparity_px = disparity.astype(np.float32) / 32.0
            disparity_px = cv2.medianBlur(disparity_px.astype(np.uint16), 5).astype(np.float32)
            
            points_3d = []
            fx = intrinsics.get('fx', intrinsics.get('fx_', 1.0))
            fy = intrinsics.get('fy', intrinsics.get('fy_', 1.0))
            cx = intrinsics.get('cx', intrinsics.get('cx_', w/2))
            cy = intrinsics.get('cy', intrinsics.get('cy_', h/2))
            baseline = intrinsics.get('baseline_m', intrinsics.get('baseline', 0.075))
            
            for v in range(h):
                for u in range(w):
                    if mask[v, u] == 0:
                        continue
                    d = disparity_px[v, u]
                    if d <= 0:
                        continue
                    z = (fx * baseline) / d
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points_3d.append([x, y, z])
                    
            
            if len(points_3d) < 10:
                print(f"Frame {frame_idx}: Insufficient points")
                continue
            
            points_3d = np.array(points_3d)
            
            # DEBUG: Point cloud bounds
            print(f"Point cloud bounds (m):")
            print(f"  X: {points_3d[:, 0].min():.4f} - {points_3d[:, 0].max():.4f} (span: {np.ptp(points_3d[:, 0]):.4f})")
            print(f"  Y: {points_3d[:, 1].min():.4f} - {points_3d[:, 1].max():.4f} (span: {np.ptp(points_3d[:, 1]):.4f})")
            print(f"  Z: {points_3d[:, 2].min():.4f} - {points_3d[:, 2].max():.4f} (span: {np.ptp(points_3d[:, 2]):.4f})")
            
            # === SPHERE FITTING ===
            center, radius = fit_sphere(points_3d)
            if center is None:
                print(f"Frame {frame_idx}: Sphere fit failed")
                continue
                
            points_3d = filter_sphere_outliers(points_3d, center, radius, threshold=0.02)
            
            # === FIND TIP (BOTTOM) ===
            tip = find_tip_blob_detector_clahe(rgb, mask, disparity, intrinsics, frame_idx) # find_tip(points_3d, center, radius, 'directional_outlier')
            
            if tip is None:
                print(f"Frame {frame_idx}: PCA failed")
                continue
            
            # === BOTTOM-TO-CENTER AXIS ===
            bottom_to_center = center - tip
            axis_norm = np.linalg.norm(bottom_to_center)
            axis_unit = bottom_to_center / axis_norm if axis_norm > 0 else np.array([0, 0, 1])
            
            # Draw overlay
            bbox = (x1, y1, x2, y2)
            overlay_img = draw_overlay(rgb, center, radius, tip, axis_unit, intrinsics, 
                                      points_3d=None, bbox=bbox, conf=conf)
            
            # Save overlay frame
            overlay_path = os.path.join(overlay_dir, f"{frame_idx:06d}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
            frame_list.append((frame_idx, overlay_path, overlay_img))
            
            result = {
                "frame_idx": frame_idx,
                "center": center.tolist(),
                "radius": float(radius),
                "tip": tip.tolist(),
                "bottom_to_center_axis": axis_unit.tolist(),
                "axis_magnitude": float(axis_norm),
                "confidence": float(conf)
            }
            results.append(result)
            print(f"Frame {frame_idx}: center={center}, radius={radius:.3f}, axis={axis_unit}")
    
    dn.free_network_ptr(network)
    
    # Save results
    output_file = os.path.join(SESSION_DIR, "peach_analysis.jsonl")
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"Results saved to {output_file}")
    
    # Create video from overlay frames
    if frame_list:
        _, _, sample_img = frame_list[0]
        h, w = sample_img.shape[:2]
        video_path = os.path.join(SESSION_DIR, "peach_analysis.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
        
        for _, _, overlay_img in sorted(frame_list, key=lambda x: x[0]):
            frame_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved to {video_path}")
    
    return results


if __name__ == "__main__":
    process_session()
