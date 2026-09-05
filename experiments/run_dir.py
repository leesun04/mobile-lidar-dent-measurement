"""
실험 결과를 실행 시각별 폴더로 분리 저장.

    from run_dir import make_run_dir, save_run_info

    out_dir = make_run_dir("zband003")
    # → experiments/results/20260903_1615_zband003/

    save_run_info(out_dir, z_band_m=0.03, weights="best.pt")
    csv_path = os.path.join(out_dir, "depth_results.csv")

실행마다 폴더가 갈리므로 이전 결과를 덮어쓰지 않는다.
어떤 설정으로 돌린 결과인지는 run_info.txt에 함께 남긴다.
"""
import os
from datetime import datetime

# experiments/results/
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


#실행 시각으로 결과 폴더 생성
def make_run_dir(tag=""):
    """results/YYYYMMDD_HHMM[_tag]/ 를 만들고 경로를 반환.

    tag를 주면 폴더명만 보고도 무엇을 시도한 실행인지 알 수 있다.
    (예: make_run_dir("zband003") → 20260903_1615_zband003)
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    if tag:
        stamp = f"{stamp}_{tag}"

    path = os.path.join(RESULTS_ROOT, stamp)

    # 같은 분에 두 번 돌린 경우를 대비해 뒤에 번호를 붙인다
    if os.path.exists(path):
        n = 2
        while os.path.exists(f"{path}_{n}"):
            n += 1
        path = f"{path}_{n}"

    os.makedirs(path)
    return path


#이 실행에 쓴 설정을 기록
def save_run_info(run_dir, **info):
    """run_info.txt로 파라미터를 남긴다.

    결과 폴더만 쌓아두면 나중에 "이 숫자가 어느 설정이었지"를 알 수 없다.
    측정에 영향을 주는 값(마스크/RANSAC 파라미터, 가중치, 입력 경로)을
    넘겨두면 폴더끼리 비교가 가능해진다.
    """
    path = os.path.join(run_dir, "run_info.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    return path
