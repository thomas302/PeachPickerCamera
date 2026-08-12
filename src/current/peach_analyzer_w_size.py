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
SESSION_DIR = "/path/to/session_folder"  # Set this to your session folder
cfg_file = "new_sample.cfg"
names_file = "new_sample.names"
weights_file = "new_sample.weights"
THRESH = 0.3

# === SPHERE FITTING ===
def fit_sphere(points):
    """Fit sphere to 3D points. Returns (center, radius)."""
    if len(points) < 4:
        return None, None
    
    center_init = points.mean(axis=0)
    
    def residuals(p):
        center = p[:3]
        radius = p[3]
        dist = np.linalg.norm(points - center, axis=1)
        return dist - radius
    
    p0 = np.append(center_init, np.linalg.norm(points - center_init).mean())
    result = least_squares(residuals, p0)
    center = result.x[:3]
    radius = result.x[3]
    return center, radius

def find_tip_pca(pcd):
    """Find tip point using PCA."""
    points = np.asarray(pcd.points)
    if len(points) < 3:
        return None
    
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered)
    primary_axis = vh[0]
    projections = np.dot(centered, primary_axis)
    tip_idx = np.argmax(projections)
    return points[tip_idx]

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

def draw_overlay(rgb, center, radius, tip, axis_unit, intrinsics, points_3d=None, bbox=None, conf=None):
    """Draw sphere center, tip, axis, points, and radius on image."""
    h, w = rgb.shape[:2]
    overlay = rgb.copy()
    
    # Draw detected points in dark green
    if points_3d is not None:
        for point in points_3d:
            proj = project_to_image(point, intrinsics, h, w)
            if proj:
                cv2.circle(overlay, proj, 2, (0, 100, 0), -1)
    
    # Draw detection bbox
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if conf:
            cv2.putText(overlay, f"Peach {conf:.2f}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Project center
    center_proj = project_to_image(center, intrinsics, h, w)
    if center_proj:
        cv2.circle(overlay, center_proj, 8, (255, 0, 0), -1)
        cv2.putText(overlay, f"C({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})",
                   (center_proj[0]+10, center_proj[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Project tip
    tip_proj = project_to_image(tip, intrinsics, h, w)
    if tip_proj:
        cv2.circle(overlay, tip_proj, 6, (0, 0, 255), -1)
        cv2.putText(overlay, "Tip", (tip_proj[0]+10, tip_proj[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Draw axis line (from tip towards center)
    if center_proj and tip_proj:
        cv2.arrowedLine(overlay, tip_proj, center_proj, (255, 255, 0), 2, tipLength=0.3)
    
    # Draw radius in top right corner
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
            mask[y1:y2, x1:x2] = cv2.GC_PR_FG
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(rgb, mask, (x1, y1, x2-x1, y2-y1), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
            
            # === 3D POINTS FROM DISPARITY ===
            points_3d = []
            fx = intrinsics.get('fx', intrinsics.get('fx_', 1.0))
            fy = intrinsics.get('fy', intrinsics.get('fy_', 1.0))
            cx = intrinsics.get('cx', intrinsics.get('cx_', w/2))
            cy = intrinsics.get('cy', intrinsics.get('cy_', h/2))
            baseline = intrinsics.get('baseline', 0.075)
            
            for v in range(h):
                for u in range(w):
                    if mask[v, u] == 0:
                        continue
                    d = disparity[v, u]
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
            
            # === SPHERE FITTING ===
            center, radius = fit_sphere(points_3d)
            if center is None:
                print(f"Frame {frame_idx}: Sphere fit failed")
                continue
            
            # === FIND TIP (BOTTOM) ===
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            tip = find_tip_pca(pcd)
            
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
                                      points_3d=points_3d, bbox=bbox, conf=conf)
            
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
