"""
YOLO 세그멘테이션으로 이미지에서 덴트 영역을 찾는다.

파이프라인의 [1]단계. RGB 이미지 한 장을 받아 덴트 테두리를 이루는
polygon 좌표를 돌려준다. 이후 단계는 이 polygon만 있으면 되고
이미지 자체는 더 쓰지 않는다.

모델은 CARDD 데이터셋으로 학습한 YOLOv11m-seg를 쓴다 (scripts/train.py 참고).
"""

# YOLO 추론 시 입력 이미지를 이 크기로 리사이즈한다.
# 내부적으로 비율을 유지하며 여백(letterbox)을 채우지만, 결과 좌표는
# 원본 기준 0~1로 정규화되어 나오므로 여백을 따로 보정할 필요가 없다.
IMG_SIZE = 640


#이미지 한 장 추론
def run_inference(model, image_path):
    """YOLO 모델로 이미지를 추론해 결과 객체를 반환한다.

    save/save_txt를 끈 이유는 ultralytics가 자동으로 runs/ 폴더에 파일을
    떨구는 것을 막기 위해서다. 결과는 메모리로만 받아 다음 단계로 넘긴다.

    Args:
        model: 로드된 YOLO 모델
        image_path (str): RGB 이미지 경로

    Returns:
        ultralytics Results: 이미지 1장에 대한 추론 결과
    """
    results = model.predict(
        source=image_path, imgsz=IMG_SIZE,
        save=False, save_txt=False, verbose=False,
    )
    return results[0]      # 이미지 1장만 넣었으므로 결과도 1개


#가장 확실한 덴트 하나 고르기
def select_best_polygon(result):
    """추론 결과에서 confidence가 가장 높은 덴트의 polygon을 고른다.

    한 이미지에 덴트가 여러 개 잡힐 수 있지만, 본 시스템은 촬영자가
    측정하려는 덴트 하나를 화면 중앙에 두고 찍는 것을 전제로 한다.
    따라서 가장 확실한 검출 하나만 쓰고 나머지는 버린다.

    Args:
        result: run_inference()가 반환한 결과 객체

    Returns:
        tuple | None: (poly_n, best_idx, cls_id)
            poly_n (np.ndarray): (N, 2) 정규화 polygon 좌표 (0~1)
            best_idx (int): 선택된 검출의 인덱스
            cls_id (int): 클래스 번호
        덴트를 하나도 못 찾으면 None
    """
    # 마스크가 아예 없으면 덴트 미검출
    if result.masks is None or len(result.masks) == 0:
        return None

    best_idx = int(result.boxes.conf.argmax())      # confidence 최고
    cls_id = int(result.boxes.cls[best_idx])

    # xyn = 정규화된(normalized) polygon 좌표. 원본 이미지 크기로 나눈 0~1 값이라
    # depth 해상도로 옮길 때 비율만 곱하면 된다.
    return result.masks.xyn[best_idx], best_idx, cls_id
