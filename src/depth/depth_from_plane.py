"""
덴트 깊이 측정의 본체.

    [2] 마스크 생성  ->  [4] RANSAC 평면  ->  [5] 점-평면 거리

앞뒤 단계를 엮어 "polygon + 3D 점 배열"을 넣으면 "깊이 mm"가 나오게 한다.
파이프라인에서 실제 숫자가 만들어지는 곳이며, 논문의 40.754mm도 여기서 나왔다.

측정 원리:
    덴트 주변의 멀쩡한 면으로 기준 평면을 만들고,
    덴트의 각 점이 그 평면에서 얼마나 떨어졌는지를 잰다.
    가장 먼 점까지의 거리가 곧 최대 깊이다.
"""
import numpy as np

from src.depth.masks import create_masks
from src.depth.run_ransac import fit_plane_to_inliers, calculate_dent_depth_yolo

# ── 파라미터 ──
# 논문 실험(50cm -> 40.754mm)에 쓰인 설정. 바꾸면 결과가 달라진다.

# RANSAC이 "이 점은 평면 위에 있다"고 인정할 거리. 단위 meter (=15mm).
# 차체는 완전한 평면이 아니라 완만한 곡면이므로 어느 정도 여유를 준다.
RANSAC_DIST = 0.015

# 평면 하나를 결정하는 데 필요한 최소 점 개수. 세 점이면 평면이 정해진다.
RANSAC_N = 3

# RANSAC 반복 횟수. 많을수록 좋은 평면을 찾을 확률이 올라간다.
RANSAC_ITERS = 100000

# 평면 후보 점을 걸러낼 깊이 범위. 단위 meter (=50mm).
# 넓힌 bbox가 차체를 벗어나 배경(바닥, 벽)까지 잡는 경우가 있는데,
# 배경은 덴트보다 훨씬 멀기 때문에 이 필터로 제거된다.
# None으로 두면 필터를 끈다.
Z_BAND_M = 0.05


#polygon을 감싸는 사각형 구하기
def poly_bbox(poly_n):
    """polygon 점들의 최소/최대 좌표로 bbox를 만든다.

    Args:
        poly_n (np.ndarray): (N, 2) 정규화 polygon 좌표

    Returns:
        tuple: (x0, y0, x1, y1) 정규화 bbox
    """
    return (
        float(np.min(poly_n[:, 0])),
        float(np.min(poly_n[:, 1])),
        float(np.max(poly_n[:, 0])),
        float(np.max(poly_n[:, 1])),
    )


#이미지 한 장의 덴트 깊이 측정
def measure_one(poly_n, world_xyz):
    """polygon + 3D 점 배열 -> (status, 최대깊이mm, 평균깊이mm).

    실패해도 예외를 던지지 않고 status 문자열로 알린다.
    폴더를 일괄 처리할 때 한 장이 실패해도 나머지가 계속 돌아야 하기 때문이다.

    status 값:
        "ok"                정상 측정
        "no_points"         덴트나 정상면에 유효한 점이 없음 (측정 실패 픽셀 과다)
        "no_plane_in_band"  z-band 필터 후 평면 후보가 부족 (배경만 잡힌 경우)
        "ransac_fail"       평면을 찾지 못함

    Args:
        poly_n (np.ndarray): (N, 2) 정규화 polygon 좌표
        world_xyz (np.ndarray): (H, W, 3) 3D 점 배열. load_xyz()의 결과

    Returns:
        tuple: (status, max_depth_mm, mean_depth_mm)
               실패 시 깊이는 NaN
    """
    # ── 1) 덴트 / 정상면 마스크 만들기 ──
    bbox_n = poly_bbox(poly_n)
    H, W, _ = world_xyz.shape
    seg_mask, inlier_mask = create_masks(poly_n, bbox_n, H, W)

    # ── 2) 마스크에 해당하는 3D 점만 꺼내기 ──
    dent_pts = world_xyz[seg_mask == 255]       # 깊이를 잴 점들
    plane_pts = world_xyz[inlier_mask == 255]   # 평면을 만들 점들

    # LiDAR가 거리를 못 잰 픽셀은 NaN이므로 버린다.
    # 남겨두면 평균/평면 계산이 전부 NaN이 된다.
    dent_pts = dent_pts[~np.isnan(dent_pts).any(axis=1)]
    plane_pts = plane_pts[~np.isnan(plane_pts).any(axis=1)]

    if len(dent_pts) == 0 or len(plane_pts) < RANSAC_N:
        return "no_points", float("nan"), float("nan")

    # ── 3) z-band 필터: 배경 제거 ──
    # 덴트가 있는 깊이대(median z) 근처의 점만 평면 후보로 남긴다.
    # 넓힌 bbox가 차체 밖으로 삐져나가 바닥/벽을 잡았을 때 그것들이 걸러진다.
    if Z_BAND_M is not None:
        dz_med = float(np.median(dent_pts[:, 2]))
        plane_pts = plane_pts[np.abs(plane_pts[:, 2] - dz_med) < Z_BAND_M]
        if len(plane_pts) < RANSAC_N:
            return "no_plane_in_band", float("nan"), float("nan")

    # ── 4) 정상면에 기준 평면 맞추기 ──
    plane_model, _ = fit_plane_to_inliers(
        plane_pts,
        distance_threshold=RANSAC_DIST,
        ransac_n=RANSAC_N,
        num_iterations=RANSAC_ITERS,
    )
    if plane_model is None:
        return "ransac_fail", float("nan"), float("nan")

    # ── 5) 덴트 점들이 평면에서 얼마나 떨어졌는지 계산 ──
    max_d, mean_d, _, _ = calculate_dent_depth_yolo(plane_model, dent_pts)
    return "ok", float(max_d), float(mean_d)
