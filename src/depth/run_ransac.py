"""
RANSAC 평면 피팅과 점-평면 거리 기반 깊이 계산.

파이프라인의 [4]~[5]단계. 이 프로젝트에서 실제 숫자를 만드는 핵심 모듈이다.

RANSAC(Random Sample Consensus)이란:
    점 3개를 무작위로 뽑아 평면을 하나 만들고, 그 평면에 가까운 점이
    몇 개인지 센다. 이를 수만 번 반복해 가장 많은 점이 동의하는 평면을 고른다.
    일부 점이 튀어도(덴트 경계, 노이즈) 다수결로 눌러버리므로 강건하다.

깊이 계산:
    평면 ax+by+cz+d=0 과 점 p의 수직거리 = |a*px + b*py + c*pz + d| / |(a,b,c)|
    덴트의 모든 점에 대해 이 거리를 구한 뒤 P99를 최대 깊이로 삼는다.
"""
import open3d as o3d
import numpy as np


def fit_plane_to_inliers(plane_points_np,
                         distance_threshold=0.005,
                         ransac_n=3,
                         num_iterations=1000,
                         probability=0.9999,
                         apply_sor=False,
                         sor_nb_neighbors=20,
                         sor_std_ratio=3.0,
                         seed=None):
    """
    정상 표면(Inliers) 점군에 RANSAC 평면 피팅을 수행합니다.

    apply_sor=False (기본): RANSAC 자체가 robust estimator이므로 전처리 SOR 미적용.
                            차량 엣지/곡면 정상 포인트 보존.
    apply_sor=True: Ablation study용. SOR 적용 시 std_ratio=3.0으로 관대하게.
    """
    print(f"Called fit_plane_to_inliers(distance_threshold={distance_threshold}, "
          f"ransac_n={ransac_n}, num_iterations={num_iterations}, apply_sor={apply_sor})")

    if len(plane_points_np) < ransac_n:
        print(f"RANSAC 오류: 평면 피팅에 필요한 포인트({ransac_n}개)보다 "
              f"인라이어 포인트({len(plane_points_np)}개)가 적습니다.")
        return None, o3d.geometry.PointCloud()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(plane_points_np)

    if apply_sor:
        print(f"[Ablation] RANSAC 전 SOR 적용... (원본: {len(pcd.points)}개)")
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=sor_nb_neighbors,
            std_ratio=sor_std_ratio,
        )
        print(f"[Ablation] SOR 적용 후: {len(pcd.points)}개")

    if len(pcd.points) < ransac_n:
        print(f"RANSAC 오류: 포인트({len(pcd.points)}개)가 부족합니다.")
        return None, o3d.geometry.PointCloud()

    print(f"\n=== RANSAC 평면 피팅 실행 중 ({len(pcd.points)}개 대상)... ===")

    # seed=None (기본): 매 호출마다 다른 랜덤 샘플링 → stability 실험에서
    #   100회 반복의 표준편차 측정 가능.
    # seed=<int>: 재현성이 필요한 단일 시각화/디버깅 시에만 지정.
    # probability는 RANSAC 내부 수렴 신뢰도 (seed와 무관).
    ransac_kwargs = dict(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        probability=probability,
    )
    if seed is not None:
        try:
            plane_model, inliers_indices = pcd.segment_plane(**ransac_kwargs, seed=seed)
        except TypeError:
            np.random.seed(seed)
            ransac_kwargs.pop("probability", None)
            plane_model, inliers_indices = pcd.segment_plane(**ransac_kwargs)
    else:
        try:
            plane_model, inliers_indices = pcd.segment_plane(**ransac_kwargs)
        except TypeError:
            ransac_kwargs.pop("probability", None)
            plane_model, inliers_indices = pcd.segment_plane(**ransac_kwargs)

    a, b, c, d = plane_model

    # 법선 방향 통일: 법선이 +Z(카메라에서 멀어지는 쪽)를 향하도록 부호를 맞춘다.
    # RANSAC은 같은 평면에도 법선을 +/- 두 방향 중 아무거나 줄 수 있어서,
    # 통일해두지 않으면 파손 방향(안쪽/바깥쪽) 판정이 실행마다 뒤집힌다.
    # 깊이 크기 자체는 절댓값을 쓰므로 이 부호와 무관하다.
    normal = np.array([a, b, c])
    camera_view_dir = np.array([0.0, 0.0, 1.0])
    if np.dot(normal, camera_view_dir) < 0:
        plane_model = [-a, -b, -c, -d]
        a, b, c, d = plane_model

    print(f"\n평면 방정식: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
    inlier_cloud = pcd.select_by_index(inliers_indices)

    # RANSAC 피팅 품질 지표
    inlier_points = np.asarray(inlier_cloud.points)
    normal_unit = np.array([a, b, c])
    residuals = np.abs(inlier_points @ normal_unit + d) / np.linalg.norm(normal_unit)
    rmse_mm = np.sqrt(np.mean(residuals ** 2)) * 1000
    inlier_ratio = len(inliers_indices) / len(pcd.points)
    print(f"RANSAC 최종 평면 포인트: {len(inliers_indices)} 개 (inlier ratio: {inlier_ratio:.3f})")
    print(f"RANSAC 평면 피팅 RMSE: {rmse_mm:.3f} mm")

    return plane_model, inlier_cloud


def calculate_dent_depth_yolo(plane_model,
                              dent_points_np,
                              remove_outliers=True,
                              nb_neighbors=20,
                              std_ratio=2.0,
                              depth_percentile=99.0):
    """
    덴트 깊이를 계산합니다.

    반환되는 max depth는 depth_percentile 기준의 robust max입니다.
    진짜 최댓값을 쓰면 단일 노이즈 포인트에 지배당하므로 P99를 기본값으로 사용.
    시각화되는 지점과 반환값은 동일한 포인트를 가리킵니다.
    """
    if len(dent_points_np) == 0:
        print("덴트(아웃라이어) 포인트가 없습니다.")
        return 0, 0, o3d.geometry.PointCloud(), None

    original_count = len(dent_points_np)
    if remove_outliers and len(dent_points_np) > nb_neighbors:
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(dent_points_np)
        pcd_cleaned, _ = pcd_temp.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        dent_points_np = np.asarray(pcd_cleaned.points)
        removed_count = original_count - len(dent_points_np)
        if removed_count > 0:
            print(f"   -> 덴트 이상치 제거: {original_count}개 → {len(dent_points_np)}개 "
                  f"({removed_count}개 제거, {removed_count/original_count*100:.1f}%)")

    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)

    # 부호 있는 거리: 평면 아래(덴트 안쪽)면 음수, 위쪽(범프)이면 양수.
    # 깊이 크기는 절댓값으로 보고하되 방향 정보는 보존.
    signed_distances = (dent_points_np @ normal + d) / normal_norm
    distances_mm = np.abs(signed_distances) * 1000

    # P99 기준 robust max: 값과 지점을 동일 인덱스로 매칭
    sorted_idx = np.argsort(distances_mm)
    p99_rank = int(len(distances_mm) * (depth_percentile / 100.0))
    p99_rank = min(p99_rank, len(distances_mm) - 1)
    max_depth_idx = sorted_idx[p99_rank]

    max_depth_mm = distances_mm[max_depth_idx]
    max_depth_point = dent_points_np[max_depth_idx]
    mean_depth_mm = np.mean(distances_mm)
    p95_depth_mm = np.percentile(distances_mm, 95)
    true_max_mm = np.max(distances_mm)

    # 주의: 위에서 법선을 +Z로 통일했기 때문에, 카메라에서 더 먼 덴트 점은
    # 부호가 양수가 되어 아래 라벨이 "바깥쪽(bump)"으로 찍힌다.
    # 표시 문구일 뿐이며 max/mean 깊이 값에는 영향이 없다.
    dent_direction = "안쪽(dent)" if signed_distances[max_depth_idx] < 0 else "바깥쪽(bump)"

    print("\n=== 덴트 깊이 분석 (YOLO 아웃라이어 대상) ===")
    print(f"최대 덴트 깊이 (P{depth_percentile:.0f} robust): {max_depth_mm:.2f} mm")
    print(f"P95 깊이: {p95_depth_mm:.2f} mm")
    print(f"평균 깊이: {mean_depth_mm:.2f} mm")
    print(f"참고 - 순수 최댓값 (노이즈 포함): {true_max_mm:.2f} mm")
    print(f"파손 방향: {dent_direction}")

    outlier_cloud = o3d.geometry.PointCloud()
    outlier_cloud.points = o3d.utility.Vector3dVector(dent_points_np)
    return max_depth_mm, mean_depth_mm, outlier_cloud, max_depth_point
