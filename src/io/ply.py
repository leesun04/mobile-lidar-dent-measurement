import numpy as np
import open3d as o3d
from ultralytics import YOLO

from src.detect.yolo_seg import run_inference, select_best_polygon
from src.io.read_depth import load_xyz
from src.depth.masks import create_masks
from src.depth.run_ransac import fit_plane_to_inliers, calculate_dent_depth_yolo
from src.depth.depth_from_plane import (
    poly_bbox, RANSAC_DIST, RANSAC_N, RANSAC_ITERS, Z_BAND_M,
)
from src.viz.plane_viz import create_plane_surface_pcd, create_depth_line_pcd


#mask에 해당하는 3D 점만 뽑기
def extract_dent_3d_points(world_xyz_arr, seg_mask):
    pts = world_xyz_arr[seg_mask == 255]
    valid = ~np.isnan(pts).any(axis=1) & np.any(pts != 0, axis=1)
    return pts[valid]

#점들을 ply 파일로 저장
def save_pointcloud_ply(points_np, output_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.io.write_point_cloud(output_path, pcd)
    return len(points_np)

#측정 결과를 색칠한 PLY 하나로 합침
def build_combined(image_path, depth_path, weights):
    """이미지 + depth → 색칠된 합본 PointCloud, 최대깊이, 평균깊이.

    measure_one()과 같은 계산을 하되, 중간 산출물(평면 inlier, 덴트 점,
    평면 격자, 깊이선)을 버리지 않고 색을 입혀 하나로 합친다.
    결과를 3D 뷰어로 열어 평면이 제대로 잡혔는지 눈으로 확인하는 용도.
    """
    # 1) YOLO로 덴트 polygon 추출
    model = YOLO(weights)
    result = run_inference(model, image_path)

    picked = select_best_polygon(result)
    if picked is None:
        raise RuntimeError(f"{image_path}: 덴트를 찾지 못함")
    poly_n = picked[0]           # (N, 2) 정규화 좌표

    # 2) ARKit JSON → 3D 점 배열 (H, W, 3)
    world_xyz = load_xyz(depth_path)

    # 3) 덴트 마스크 + 주변 정상면 마스크 생성
    bbox_n = poly_bbox(poly_n)
    H, W, _ = world_xyz.shape
    seg_mask, inlier_mask = create_masks(poly_n, bbox_n, H, W)

    # 4) 마스크로 점 추출. 측정 실패 픽셀(NaN)은 제외
    dent_pts = world_xyz[seg_mask == 255]
    plane_pts = world_xyz[inlier_mask == 255]
    dent_pts = dent_pts[~np.isnan(dent_pts).any(axis=1)]
    plane_pts = plane_pts[~np.isnan(plane_pts).any(axis=1)]

    # 5) z-band 필터: 확장 bbox가 배경까지 잡는 경우를 걸러낸다.
    #    덴트의 median z에서 Z_BAND_M 이내인 점만 평면 후보로 남김
    if Z_BAND_M is not None:
        dz_med = float(np.median(dent_pts[:, 2]))
        plane_pts = plane_pts[np.abs(plane_pts[:, 2] - dz_med) < Z_BAND_M]

    # 6) 정상면에 RANSAC 평면 피팅
    plane_model, inlier_cloud = fit_plane_to_inliers(
        plane_pts,
        distance_threshold=RANSAC_DIST,
        ransac_n=RANSAC_N,
        num_iterations=RANSAC_ITERS,
    )
    if plane_model is None:
        raise RuntimeError(f"{image_path}: RANSAC 실패")

    # 7) 덴트 점들의 평면까지 수직거리 → P99 최대 깊이
    max_d, mean_d, dent_cloud, max_depth_point = calculate_dent_depth_yolo(
        plane_model, dent_pts
    )

    # 8) 시각화용 요소 생성 (측정값에는 영향 없음)
    plane_surface = create_plane_surface_pcd(plane_model, inlier_cloud)
    depth_line = create_depth_line_pcd(plane_model, max_depth_point, num_points=200)

    # 9) 색 입히기
    inlier_cloud.paint_uniform_color([0.0, 0.0, 1.0])   # 파랑 = 정상면(평면 inlier)
    dent_cloud.paint_uniform_color([1.0, 0.0, 0.0])     # 빨강 = 덴트
    plane_surface.paint_uniform_color([0.5, 0.5, 0.5])  # 회색 = RANSAC이 찾은 평면
    depth_line.paint_uniform_color([1.0, 1.0, 0.0])     # 노랑 = 최대깊이 지점→평면

    # 10) 네 덩어리를 하나의 PointCloud로 합침
    combined = inlier_cloud + dent_cloud + plane_surface + depth_line
    return combined, max_d, mean_d
