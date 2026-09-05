"""
측정 결과를 색상 PLY로 저장.

    파랑 = 정상면 (RANSAC 평면 inlier)
    빨강 = 덴트
    회색 = 찾아낸 평면
    노랑 = 최대 깊이 지점에서 평면까지의 선

    python -m scripts.export_ply --image a.jpg --depth a.json \
        --weights weights/best.pt --out result.ply

저장된 .ply는 MeshLab, CloudCompare, Open3D 뷰어 등으로 열어볼 수 있다.
"""
import os
import sys
import argparse

import open3d as o3d

# 리포 루트를 import 경로에 추가 (python scripts/export_ply.py 로도 실행 가능하게)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io.ply import build_combined


#터미널에서 받을 인자 등록
def parse_args():
    ap = argparse.ArgumentParser(description="측정 결과를 색상 PLY로 저장")
    ap.add_argument("--image", required=True, help="RGB 이미지 (.jpg)")
    ap.add_argument("--depth", required=True, help="ARKit depth (.json)")
    ap.add_argument("--weights", required=True, help="YOLO 가중치 (.pt)")
    ap.add_argument("--out", default="result.ply", help="저장할 PLY 경로")
    ap.add_argument("--device", default="", help="GPU 번호 (예: 0)")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 측정 + 시각화 요소를 한 번에 만든다
    cloud, max_d, mean_d = build_combined(args.image, args.depth, args.weights)

    o3d.io.write_point_cloud(args.out, cloud)
    print(f"max={max_d:.2f}mm  mean={mean_d:.2f}mm  → {args.out}")


if __name__ == "__main__":
    main()
