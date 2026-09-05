"""
iPhone LiDAR 덴트 깊이 측정.

입력: RGB 이미지(.jpg) + 같은 이름의 ARKit depth(.json)

    # 이미지 1장
    python -m scripts.measure --input a.jpg --weights weights/best.pt

    # 폴더 일괄 + CSV 저장
    python -m scripts.measure --input ./data --weights weights/best.pt --csv out.csv
"""
import os
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# 리포 루트를 import 경로에 추가 (python scripts/measure.py 로도 실행 가능하게)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detect.yolo_seg import run_inference, select_best_polygon
from src.io.read_depth import load_xyz
from src.depth.depth_from_plane import measure_one


#입력 경로 -> (이미지, depth json) 쌍 목록
def collect_pairs(input_path, depth_path=None):
    """파일이면 1쌍, 폴더면 안의 jpg 전체. json은 같은 이름으로 찾는다."""
    p = Path(input_path)

    if p.is_file():
        js = Path(depth_path) if depth_path else p.with_suffix(".json")
        if not js.exists():
            print(f"[오류] depth json을 찾을 수 없음: {js}")
            return []
        return [(str(p), str(js))]

    if not p.is_dir():
        print(f"[오류] 경로가 없음: {p}")
        return []

    pairs = []
    for img in sorted(p.glob("*.jpg")):
        js = img.with_suffix(".json")
        if js.exists():
            pairs.append((str(img), str(js)))
        else:
            print(f"[건너뜀] {img.name} — 짝이 되는 json 없음")
    return pairs


#이미지 1장 측정: YOLO -> polygon -> 3D 점 -> 깊이
def measure_file(model, image_path, json_path, rgb_h=1920):
    """(status, max_depth_mm, mean_depth_mm) 반환."""
    result = run_inference(model, image_path)

    picked = select_best_polygon(result)
    if picked is None:
        return "no_detection", float("nan"), float("nan")
    poly_n = picked[0]

    world_xyz = load_xyz(json_path, rgb_h=rgb_h)
    return measure_one(poly_n, world_xyz)


def parse_args():
    ap = argparse.ArgumentParser(description="iPhone LiDAR 덴트 깊이 측정")
    ap.add_argument("--input", required=True,
                    help="jpg 파일 또는 jpg가 든 폴더")
    ap.add_argument("--depth", default=None,
                    help="depth json 경로. 생략하면 이미지와 같은 이름으로 찾음")
    ap.add_argument("--weights", required=True,
                    help="YOLO 세그멘테이션 가중치 (.pt)")
    ap.add_argument("--csv", default=None,
                    help="결과를 CSV로 저장할 경로")
    ap.add_argument("--rgb-h", type=int, default=1920,
                    help="RGB 세로 해상도. 초점거리 스케일 보정용 (기본 1920)")
    ap.add_argument("--device", default="",
                    help="사용할 GPU 번호 (예: 0). 생략하면 자동")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    pairs = collect_pairs(args.input, args.depth)
    if not pairs:
        print("처리할 파일이 없습니다.")
        return

    if not os.path.exists(args.weights):
        print(f"[오류] 가중치 파일이 없습니다: {args.weights}")
        return

    model = YOLO(args.weights)

    rows = []
    for img_path, json_path in pairs:
        stem = Path(img_path).stem
        try:
            status, max_d, mean_d = measure_file(
                model, img_path, json_path, rgb_h=args.rgb_h
            )
        except Exception as e:
            status, max_d, mean_d = f"error:{type(e).__name__}", float("nan"), float("nan")
            print(f"{stem}  예외: {e}")

        if status == "ok":
            print(f"{stem}  max={max_d:.2f}mm  mean={mean_d:.2f}mm")
        else:
            print(f"{stem}  {status}")

        rows.append(dict(
            file=stem,
            status=status,
            max_depth_mm="" if np.isnan(max_d) else f"{max_d:.3f}",
            mean_depth_mm="" if np.isnan(mean_d) else f"{mean_d:.3f}",
        ))

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n측정 완료: {n_ok}/{len(rows)}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "status",
                                              "max_depth_mm", "mean_depth_mm"])
            w.writeheader()
            w.writerows(rows)
        print(f"CSV 저장: {args.csv}")


if __name__ == "__main__":
    main()
