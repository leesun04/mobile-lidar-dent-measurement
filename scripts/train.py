"""
덴트 세그멘테이션 모델 학습.

    python -m scripts.train --data data_cardd_v3.yaml
    python -m scripts.train --data my.yaml --epochs 100 --device 0,1 --name test

데이터셋 yaml은 리포에 포함되지 않으므로 직접 만들어 --data로 지정해야 한다.
"""
import argparse
from ultralytics import YOLO


#터미널에서 받을 인자 등록
def parse_args():
    ap = argparse.ArgumentParser(description="덴트 세그멘테이션 모델 학습")

    # 모델 구조 yaml은 리포 안(configs/)에 있어 거의 바뀌지 않으므로 기본값 지정
    ap.add_argument("--model", default="configs/seg-model.yaml",
                    help="모델 구조 yaml (기본: configs/seg-model.yaml)")

    # 데이터셋 경로는 사용자마다 다르고 리포에 포함되지 않아 기본값을 줄 수 없음
    ap.add_argument("--data", required=True,
                    help="데이터셋 yaml (필수)")

    ap.add_argument("--project", default="runs",
                    help="학습 결과 저장 폴더 (기본: runs)")
    ap.add_argument("--name", default="train",
                    help="결과 하위 폴더명. 실험마다 바꾸면 구분됨")

    # 아래 3개는 논문 실험에 쓴 값이 기본값.
    # 옵션 없이 실행하면 논문과 동일한 조건으로 학습된다.
    ap.add_argument("--epochs", type=int, default=400,
                    help="학습 에포크 수 (기본: 400)")
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="입력 이미지 크기 (기본: 1024)")
    ap.add_argument("--batch", type=int, default=14,
                    help="배치 크기. GPU 메모리 부족하면 줄일 것 (기본: 14)")

    # "0,1"처럼 여러 GPU를 넘길 수 있어야 하므로 int가 아닌 문자열로 받는다
    ap.add_argument("--device", default="0",
                    help="GPU 번호 (예: '0' 또는 '0,1')")

    return ap.parse_args()


def main():
    args = parse_args()

    # yaml에서 모델 구조를 읽어 새 모델 생성 (사전학습 가중치 없이 처음부터 학습)
    model = YOLO(args.model)
    print("모델 정보")
    model.info()

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,

        # ─── 아래는 논문 실험 설정. 바꾸면 결과가 달라지므로 인자로 빼지 않음 ───
        overlap_mask=True,          # 겹치는 인스턴스 마스크 분리
        lr0=0.001,                  # 초기 학습률
        patience=None,              # 조기 종료 없이 끝까지 학습
        cos_lr=False,               # 코사인 스케줄러 대신 선형 감소
        optimizer="auto",           # SGD 자동 선택
        weight_decay=0.0005,        # L2 정규화 (과적합 방지)
        auto_augment="randaugment", # 자동 증강 정책
        erasing=0.4,                # 랜덤 지우기 확률
        mosaic=1.0,                 # 모자이크 증강 (작은 덴트 검출에 유리)
        fliplr=0.5,                 # 좌우 반전 (차체 좌/우 대칭 학습)
        augment=False,              # 추론 시 TTA 비활성
        workers=8,                  # 데이터 로딩 워커 수
        save=True,
        verbose=True,
        seed=0,                     # 재현성 확보
        deterministic=True,
    )


if __name__ == "__main__":
    main()
