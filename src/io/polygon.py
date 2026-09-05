"""
YOLO 세그멘테이션 polygon의 저장 / 불러오기.

polygon은 덴트 영역의 테두리를 이루는 점들이며, 좌표는 0~1로 정규화되어 있다.
정규화 좌표를 쓰는 이유는 RGB(1440x1920)와 depth(192x256)의 해상도가 달라
픽셀 번호로는 서로 대응시킬 수 없기 때문이다. 비율로 두면 어느 해상도에도
`좌표 * (W, H)` 한 번으로 옮길 수 있다.

sidecar란 이미지와 같은 이름의 .txt 파일을 뜻한다.
    0513164559.jpg / 0513164559.json / 0513164559.txt

YOLO를 다시 돌리면 세그멘테이션 결과가 미세하게 달라져 측정값이 흔들린다.
polygon을 txt로 한 번 저장해두면 나중에 몇 번을 다시 계산해도 같은 영역을
쓰게 되어 실험 재현성이 확보된다.
"""
import os
import numpy as np


#polygon을 txt 파일로 저장
def save_polygon_sidecar(poly_n, cls_id, txt_path):
    """polygon을 YOLO seg와 같은 포맷의 한 줄로 저장한다.

    포맷: "cls x1 y1 x2 y2 ... xN yN"  (좌표는 0~1 정규화, 소수점 6자리)

    Args:
        poly_n (np.ndarray): (N, 2) 정규화 polygon 좌표
        cls_id (int): 클래스 번호 (덴트 단일 클래스이므로 보통 0)
        txt_path (str): 저장할 .txt 경로
    """
    # (N, 2) 배열을 [x1, y1, x2, y2, ...] 한 줄로 펼친다
    flat = " ".join(f"{v:.6f}" for v in poly_n.flatten())
    with open(txt_path, "w") as f:
        f.write(f"{cls_id} {flat}\n")


#txt 파일에서 polygon 읽기
def load_polygon_sidecar(txt_path):
    """저장된 polygon txt를 (N, 2) 정규화 좌표 배열로 되돌린다.

    파일이 없거나 내용이 온전하지 않으면 None을 반환한다.
    폴더를 일괄 처리할 때 일부 파일만 없을 수 있으므로, 예외를 던지지 않고
    None을 돌려 호출부가 건너뛰도록 한다.

    Args:
        txt_path (str): polygon이 저장된 .txt 경로

    Returns:
        np.ndarray | None: (N, 2) 정규화 좌표. 읽지 못하면 None
    """
    if not os.path.exists(txt_path):
        return None

    with open(txt_path, "r") as f:
        line = f.readline().strip()
    if not line:
        return None

    parts = line.split()
    # 최소 조건: 클래스 번호 1개 + 점 3개(=좌표 6개). 삼각형보다 작으면 면이 아니다
    if len(parts) < 7:
        return None

    coords = list(map(float, parts[1:]))    # 맨 앞 클래스 번호는 버린다
    # x, y 쌍으로 떨어지지 않으면 파일이 깨진 것
    if len(coords) % 2 != 0:
        return None

    return np.array(coords).reshape(-1, 2)
