"""
측정 결과를 3D로 확인하기 위한 시각화 요소 생성.

숫자만 봐서는 RANSAC이 엉뚱한 평면을 잡았는지 알 수 없다.
찾아낸 평면과 최대 깊이 지점을 점으로 그려 PLY에 함께 담으면,
3D 뷰어로 열어 한눈에 검증할 수 있다.

    파랑 = 정상면        빨강 = 덴트
    회색 = 찾은 평면      노랑 = 최대 깊이 선   <- 이 파일이 만드는 것

여기서 만드는 것은 전부 "보기 위한" 점이며, 측정값 계산에는 쓰이지 않는다.
이 파일을 통째로 지워도 깊이 숫자는 바뀌지 않는다.
"""
import numpy as np
import open3d as o3d

#평면을 눈에 보이는 격자 점으로 그리기
def create_plane_surface_pcd(plane_model, inlier_cloud, grid_size=50):
    """평면 방정식을 grid_size x grid_size 개의 회색 점으로 그린다.

    평면 방정식 ax+by+cz+d=0 자체는 무한히 넓은 면이라 그릴 수 없다.
    실제 정상면 점들이 분포한 범위(bounding box)만큼만 잘라서 격자를 만든다.

    Args:
        plane_model (list): [a, b, c, d] 평면 계수
        inlier_cloud: 평면 위 점들. 그릴 범위를 정하는 데 쓴다
        grid_size (int): 한 변당 점 개수. 50이면 총 2,500개

    Returns:
        o3d.geometry.PointCloud: 회색으로 칠해진 평면 격자
    """
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])

    # 평면이 수직인 경우(c=0) 등을 처리하기 위해 가장 큰 축을 기준으로 그림
    dominant_axis = np.argmax(np.abs(normal))
    
    # 평면 점들이 분포한 영역을 기준으로 그리드 생성
    aabb = inlier_cloud.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    points = []
    
    try:
        if dominant_axis == 0: # X축이 법선 (Y-Z 평면)
            y_range = np.linspace(min_bound[1], max_bound[1], grid_size)
            z_range = np.linspace(min_bound[2], max_bound[2], grid_size)
            yv, zv = np.meshgrid(y_range, z_range)
            xv = (-b * yv - c * zv - d) / a
            points = np.stack([xv, yv, zv], axis=-1)
            
        elif dominant_axis == 1: # Y축이 법선 (X-Z 평면)
            x_range = np.linspace(min_bound[0], max_bound[0], grid_size)
            z_range = np.linspace(min_bound[2], max_bound[2], grid_size)
            xv, zv = np.meshgrid(x_range, z_range)
            yv = (-a * xv - c * zv - d) / b
            points = np.stack([xv, yv, zv], axis=-1)
            
        else: # Z축이 법선 (X-Y 평면) - 가장 일반적
            x_range = np.linspace(min_bound[0], max_bound[0], grid_size)
            y_range = np.linspace(min_bound[1], max_bound[1], grid_size)
            xv, yv = np.meshgrid(x_range, y_range)
            zv = (-a * xv - b * yv - d) / c
            points = np.stack([xv, yv, zv], axis=-1)
            
        points = points.reshape(-1, 3)
    except ZeroDivisionError:
        print("평면 그리드 생성 중 0으로 나누기 오류 발생 (드문 경우).")
        return o3d.geometry.PointCloud()

    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(points)
    plane_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # 회색으로 표시
    return plane_pcd

#최대 깊이 지점에서 평면까지 선 긋기
def create_depth_line_pcd(plane_model, max_depth_point, num_points=100):
    """가장 깊은 점 -> 평면까지의 수직선을 노란 점들로 그린다.

    "이만큼 파였다"를 눈으로 보여주는 막대. 점을 평면에 수직으로 내린
    발(투영점)을 구한 뒤, 두 점 사이를 균등하게 나눠 채운다.

    Args:
        plane_model (list): [a, b, c, d] 평면 계수
        max_depth_point (np.ndarray): 가장 깊은 지점의 3D 좌표
        num_points (int): 선을 이룰 점 개수. 많을수록 촘촘한 선

    Returns:
        o3d.geometry.PointCloud: 노란색 선. 지점이 없으면 빈 클라우드
    """
    if max_depth_point is None:
        return o3d.geometry.PointCloud()

    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    
    # 평면에 투영된 점 계산
    # p_proj = p - ( (p.n + d) / (n.n) ) * n
    dist = np.dot(max_depth_point, normal) + d
    projected_point = max_depth_point - (dist / np.dot(normal, normal)) * normal

    # 두 점 사이에 선형 보간으로 점들 생성
    line_points = np.linspace(max_depth_point, projected_point, num_points)

    line_pcd = o3d.geometry.PointCloud()
    line_pcd.points = o3d.utility.Vector3dVector(line_points)
    line_pcd.paint_uniform_color([1, 1, 0])  # ⭐️ 노란색으로 표시
    return line_pcd