"""
polygon으로부터 두 개의 마스크를 만든다.

    seg_mask    = 덴트 영역        → 깊이를 잴 대상
    inlier_mask = 주변 정상 표면   → 기준 평면을 만들 재료

덴트 깊이란 "정상 표면이었다면 있었을 위치"에서 얼마나 파였는지이므로,
덴트만이 아니라 그 주변의 멀쩡한 면도 함께 필요하다.

    ┌──────────────────────────┐  ← 확장 bbox
    │      inlier_mask         │
    │    ┌──────────┐          │
    │    │ seg_mask │          │
    │    └──────────┘          │
    └──────────────────────────┘

파이프라인의 [2]단계. 마스크는 depth 해상도(192x256)로 만들어지며,
polygon의 0~1 좌표에 (W, H)를 곱해 격자 번호로 바꿔 그린다.
"""
import numpy as np
import cv2

# ── 파라미터 ──
# 이 값들이 논문 실험(50cm -> 40.754mm)에 쓰인 설정이다. 바꾸면 결과가 달라진다.

# bbox를 좌우/상하로 얼마나 넓혀 정상면을 확보할지.
# 0.80이면 덴트 크기의 80%만큼 사방으로 확장한다.
# 너무 작으면 평면 피팅에 쓸 점이 부족하고, 너무 크면 배경까지 들어온다.
PADDING_RATIO = 0.80

# seg_mask를 몇 번 부풀려 inlier에서 뺄지.
# 덴트 경계의 애매한 픽셀이 "정상면"으로 새어 들어가 평면을 기울이는 것을 막는다.
DILATION_ITER = 1

# polygon을 convex hull(볼록 껍질)로 채울지.
# 학습 모델이 덴트를 도넛 모양(테두리만)으로 잡는 경우가 있어,
# hull로 채워야 안쪽 깊은 부분이 마스크에 포함된다.
FILL_WITH_HULL = True

# 덴트 영역을 hull에 bbox까지 합쳐서 정의할지.
# polygon이 실제 덴트 테두리에서 살짝 어긋나 가장 깊은 지점을 놓치는 경우를 보강한다.
DENT_INCLUDE_BBOX = True


#bbox를 사방으로 넓히기
def expand_bbox(bbox, ratio):
    """정규화 bbox를 ratio만큼 확장한다. 이미지 밖으로 나가면 0~1로 자른다.

    Args:
        bbox (tuple): (x0, y0, x1, y1) 정규화 좌표
        ratio (float): 확장 비율. 0.8이면 폭/높이의 80%씩 사방으로

    Returns:
        tuple: 확장된 (x0, y0, x1, y1)
    """
    x0, y0, x1, y1 = bbox
    bw = x1 - x0        # 폭
    bh = y1 - y0        # 높이
    return (
        np.clip(x0 - bw * ratio, 0.0, 1.0),
        np.clip(y0 - bh * ratio, 0.0, 1.0),
        np.clip(x1 + bw * ratio, 0.0, 1.0),
        np.clip(y1 + bh * ratio, 0.0, 1.0),
    )


#덴트 마스크 + 정상면 마스크 생성
def create_masks(poly_n, bbox_n, H, W):
    """polygon → (seg_mask, inlier_mask). 둘 다 (H, W) uint8, 값은 0 또는 255.

    Args:
        poly_n (np.ndarray): (N, 2) 정규화 polygon 좌표
        bbox_n (tuple): polygon을 감싸는 정규화 bbox
        H, W (int): 만들 마스크 크기. depth 배열 크기를 그대로 쓴다

    Returns:
        (seg_mask, inlier_mask): 덴트 영역, 주변 정상면 영역
    """
    # 0~1 정규화 좌표 -> 실제 격자 번호.
    # 예: x=0.3125 이고 W=192면 -> 60번 칸
    scale = np.array([W, H])
    seg_idx = (poly_n * scale).astype(np.int32)

    # ── 1) 덴트 마스크 ──
    seg_mask = np.zeros((H, W), dtype=np.uint8)
    if FILL_WITH_HULL:
        # 볼록 껍질로 감싸 안쪽을 통째로 채운다 (도넛 모양 검출 보정)
        hull = cv2.convexHull(seg_idx)
        cv2.fillPoly(seg_mask, [hull], 255)
    else:
        cv2.fillPoly(seg_mask, [seg_idx], 255)

    # polygon이 어긋나 깊은 부분을 놓치는 경우를 대비해 bbox까지 덴트로 포함
    if DENT_INCLUDE_BBOX:
        tx0, ty0, tx1, ty1 = bbox_n
        cv2.rectangle(
            seg_mask,
            (int(tx0 * W), int(ty0 * H)),
            (int(tx1 * W), int(ty1 * H)),
            255, -1,        # -1 = 내부를 채움
        )

    # ── 2) 정상면 마스크 ──
    # 넓힌 bbox 전체를 칠한 뒤
    px0, py0, px1, py1 = expand_bbox(bbox_n, PADDING_RATIO)
    bbox_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(
        bbox_mask,
        (int(px0 * W), int(py0 * H)),
        (int(px1 * W), int(py1 * H)),
        255, -1,
    )

    # 덴트를 살짝 부풀려서 빼낸다.
    # 그냥 seg_mask를 빼면 경계선상의 애매한 픽셀이 정상면에 남아 평면을 기울인다.
    kernel = np.ones((3, 3), np.uint8)
    seg_dilated = cv2.dilate(seg_mask, kernel, iterations=DILATION_ITER)

    # 정상면 = 넓힌 bbox 안 - 부풀린 덴트
    inlier_mask = np.zeros((H, W), dtype=np.uint8)
    inlier_mask[(bbox_mask == 255) & (seg_dilated == 0)] = 255

    return seg_mask, inlier_mask
