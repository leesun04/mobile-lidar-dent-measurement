import json
import numpy as np


def read_json(fpath):
    """JSON 파일을 읽어서 딕셔너리로 반환."""
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_xyz(pc_path, rgb_h=1920, return_valid_mask=False):
    """
    ARKit LiDAR JSON → 3D 카메라 좌표 포인트 클라우드.

    핀홀 역투영만 수행한다:
        x = (j - W/2) * z / (fl_x / ratio)
        y = (i - H/2) * z / (fl_y / ratio)
    ratio = rgb_h / W. ARKit이 주는 fl은 RGB 해상도 기준이므로
    depth 해상도(W×H)에 맞게 축소해서 사용한다.

    camera-to-world 변환(m00~m33, Pos)은 적용하지 않는다.
    RANSAC 평면 피팅과 점-평면 수직거리는 강체변환에 불변이라
    깊이 측정 결과가 달라지지 않는다.

    Args:
        pc_path (str): LiDAR JSON 파일 경로
        rgb_h (int): RGB 세로 해상도. fl 스케일 보정용
        return_valid_mask (bool): True면 (xyz_arr, valid_mask) 튜플 반환

    Returns:
        np.ndarray: (W, H, 3) 카메라 좌표 [x, y, z] 배열 (단위: meter)
    """
    pc = read_json(pc_path)

    # depth 맵 해상도 (ARKit LiDAR는 보통 256 x 192)
    W = pc["Width"]
    H = pc["Height"]

    # 초점거리. 촬영마다 미세하게 달라지므로 JSON에서 읽는다 (1446~1454 관측)
    fl_x = pc["fl"]["x"]
    fl_y = pc["fl"]["y"]

    # fl은 RGB(1920 기준) 해상도라서 depth 해상도로 축소해야 한다.
    # 1920 / 256 = 7.5배 차이
    ratio = rgb_h / W

    # 거리값 49,152개(1차원) → (H, W) 2차원 배열로 펼침.
    # 측정 실패 픽셀은 NaN으로 들어오며 그대로 둔다.
    # (0으로 바꾸면 원점에 가짜 점 뭉치가 생김)
    z = np.asarray(pc["Depth"], dtype=np.float64).reshape(H, W)

    # 픽셀 좌표표 생성.
    #   j = 열 번호표, i = 행 번호표 (둘 다 (H, W) 크기)
    # 원래 이중 for문에서 세던 j, i를 한꺼번에 만들어두는 것
    j, i = np.meshgrid(np.arange(W), np.arange(H))

    # 핀홀 역투영: 픽셀 위치 + 거리 → 실제 3D 좌표(meter).
    # 이미지 중심(W/2, H/2)을 원점으로 놓고, 초점거리로 나눠 각도를 거리로 환산.
    # j, i, z 모두 (H, W) 배열이라 numpy가 전체 픽셀을 한 번에 계산한다.
    x = (j - W / 2) * z / (fl_x / ratio)
    y = (i - H / 2) * z / (fl_y / ratio)

    # (H, W) 3장을 겹쳐 (H, W, 3) 만들기 → 각 픽셀이 [x, y, z] 좌표를 가짐
    xyz_arr = np.stack([x, y, z], axis=-1)

    # depth 배열이 RGB 이미지와 90도 틀어져 있어 방향을 맞춘다.
    # 좌표 변환이 아니라 배열 회전. 빼면 polygon 마스크가 엉뚱한 곳을 가리킨다.
    xyz_arr = np.rot90(xyz_arr, k=-1)

    if return_valid_mask:
        valid_mask = ~np.isnan(xyz_arr).any(axis=-1)
        return xyz_arr, valid_mask
    return xyz_arr