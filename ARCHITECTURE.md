# 코드 구조와 동작 흐름

이 문서는 폴더 구조, 모든 함수의 역할, 그리고 실행할 때 어떤 순서로
호출되는지를 정리한 것이다.

---

## 1. 한 줄 요약

> 아이폰으로 찍은 **사진 + 거리 데이터**를 받아, 덴트가 몇 mm 파였는지 계산한다.

```
입력: a.jpg (RGB 사진)  +  a.json (ARKit LiDAR 거리값)
출력: 40.75 mm
```

---

## 2. 처리 흐름 5단계

```
 a.jpg ─────────────┐
                    │
        [1] YOLO 세그멘테이션          src/detect/yolo_seg.py
            덴트 테두리 polygon 추출
                    │
                    ▼
        [2] 마스크 생성                src/depth/masks.py
            덴트 영역 / 주변 정상면을 나눔
                    │
 a.json ────────────┤
                    │
        [3] 3D 변환                    src/io/read_depth.py
            거리값 -> 3차원 점 좌표
                    │
                    ▼
        [4] RANSAC 평면 피팅           src/depth/run_ransac.py
            정상면으로 "원래 면" 추정
                    │
                    ▼
        [5] 점-평면 거리 계산          src/depth/run_ransac.py
            덴트가 평면에서 얼마나 떨어졌나
                    │
                    ▼
              40.75 mm
```

전체를 엮는 것은 `src/depth/depth_from_plane.py`의 `measure_one()`이고,
사람이 실행하는 입구는 `scripts/measure.py`다.

---

## 3. 왜 이렇게 재는가

덴트 깊이란 **"원래 매끈했던 면에서 얼마나 들어갔는가"** 다.
그런데 원래 면은 이미 찌그러져서 볼 수 없다. 그래서:

1. 덴트 **주변**의 멀쩡한 면을 본다
2. 그 면들로 평면을 하나 만든다 (= 원래 면이었을 자리)
3. 덴트의 각 점이 그 평면에서 얼마나 떨어졌는지 잰다

```
        정상면            정상면
    ────────────┐      ┌────────────     <- 실제 표면
                 \    /
                  \__/                    <- 덴트
    ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈    <- RANSAC이 만든 기준 평면
                  ↕
                이 거리 = 깊이
```

---

## 4. 폴더 구조

```
mobile-lidar-dent-measurement/
├── README.md               작업 문서 (공개 전 교체 예정)
├── ARCHITECTURE.md         이 문서
├── requirements.txt        의존 패키지
├── .gitignore
│
├── configs/
│   ├── default.yaml        측정 파라미터 모음
│   └── seg-model.yaml      YOLOv11m-seg 구조 정의
│
├── src/                    라이브러리 (직접 실행하지 않음)
│   ├── io/                 데이터 읽기/쓰기
│   │   ├── read_depth.py   ARKit JSON -> 3D 점
│   │   ├── polygon.py      polygon 저장/불러오기
│   │   └── ply.py          PLY 파일 + 색상 합성
│   ├── detect/
│   │   └── yolo_seg.py     YOLO 추론
│   ├── depth/              측정 핵심
│   │   ├── masks.py        마스크 생성
│   │   ├── run_ransac.py   평면 피팅 + 깊이 계산
│   │   └── depth_from_plane.py  전체를 엮는 곳
│   └── viz/
│       └── plane_viz.py    3D 시각화 요소
│
├── scripts/                사람이 실행하는 곳
│   ├── measure.py          깊이 측정
│   ├── export_ply.py       결과를 3D로 저장
│   └── train.py            모델 학습
│
└── experiments/            논문 실험
    ├── measure_depth_all.py
    ├── run_dir.py          결과 폴더 관리
    ├── results/            실행마다 쌓임 (깃 제외)
    └── results_published/  확정 결과 (깃 포함)
```

**`src/` 와 `scripts/` 를 나눈 이유**: `src/`는 부품이고 `scripts/`는 조립품이다.
부품을 한 벌만 두면, 알고리즘을 고칠 때 한 곳만 고쳐도 측정/시각화/실험이
모두 같이 바뀐다. 정리 전에는 같은 함수가 3~4벌씩 복사돼 있어서
한 곳만 고치면 스크립트마다 결과가 달라지는 문제가 있었다.

---

## 5. 함수 전체 목록

### `src/io/read_depth.py` — ARKit JSON을 3D 점으로

| 함수 | 하는 일 |
|---|---|
| `read_json(fpath)` | JSON 파일을 열어 딕셔너리로 반환 |
| `load_xyz(pc_path, rgb_h=1920, return_valid_mask=False)` | **거리값 배열 -> 3D 좌표 배열.** 핀홀 역투영 후 `rot90`으로 방향을 RGB에 맞춘다 |

JSON에서 읽는 값은 4개뿐이다.

| 키 | 내용 |
|---|---|
| `Width`, `Height` | depth 해상도 (256 x 192) |
| `Depth` | 거리값 49,152개 (1차원 배열, meter) |
| `fl` | 초점거리. 촬영마다 다름 (1446~1454 관측) |

`m00~m33`(카메라 자세 행렬), `Rot`, `Pos`는 **읽지 않는다.**
회전/이동은 점과 평면 사이 거리를 바꾸지 않으므로 깊이 결과에 영향이 없다.

### `src/io/polygon.py` — polygon 저장/불러오기

| 함수 | 하는 일 |
|---|---|
| `save_polygon_sidecar(poly_n, cls_id, txt_path)` | polygon을 `"cls x1 y1 x2 y2 ..."` 한 줄로 저장 |
| `load_polygon_sidecar(txt_path)` | 저장된 txt -> (N, 2) 좌표 배열. 실패 시 None |

YOLO를 다시 돌리면 세그멘테이션이 미세하게 달라져 측정값이 흔들린다.
polygon을 한 번 저장해두면 몇 번을 재계산해도 같은 영역을 쓴다.
실험 재현성을 위한 장치이며, 현재 `experiments/`에서만 쓴다.

### `src/detect/yolo_seg.py` — 덴트 찾기

| 함수 | 하는 일 |
|---|---|
| `run_inference(model, image_path)` | YOLO로 이미지 1장 추론 |
| `select_best_polygon(result)` | confidence 최고인 덴트 하나의 polygon 반환. 없으면 None |

반환되는 `masks.xyn`은 **0~1로 정규화된 좌표**다. 원본 해상도로 나눈 값이라
RGB(1440x1920)와 depth(192x256)의 크기 차이를 신경 쓰지 않아도 된다.

### `src/depth/masks.py` — 영역 나누기

| 함수 | 하는 일 |
|---|---|
| `expand_bbox(bbox, ratio)` | bbox를 사방으로 넓힘. 0~1 범위로 자름 |
| `create_masks(poly_n, bbox_n, H, W)` | **덴트 마스크 + 정상면 마스크** 생성 |

```
    ┌──────────────────────────┐  <- bbox를 80% 넓힌 영역
    │      inlier_mask         │     (= 평면을 만들 재료)
    │    ┌──────────┐          │
    │    │ seg_mask │          │     (= 깊이를 잴 대상)
    │    └──────────┘          │
    └──────────────────────────┘
```

정상면 = **넓힌 bbox - 부풀린 덴트**. 덴트를 살짝 부풀려서 빼는 이유는,
경계선상의 애매한 픽셀이 정상면에 섞이면 평면이 기울기 때문이다.

### `src/depth/run_ransac.py` — 평면과 깊이

| 함수 | 하는 일 |
|---|---|
| `fit_plane_to_inliers(plane_points_np, ...)` | **RANSAC 평면 피팅.** `ax+by+cz+d=0` 계수와 평면 위 점들을 반환 |
| `calculate_dent_depth_yolo(plane_model, dent_points_np, ...)` | **점-평면 거리 -> 깊이 mm.** P99 최대값, 평균, 덴트 점 구름, 최대 지점 반환 |

**P99를 쓰는 이유**: 진짜 최댓값을 쓰면 노이즈 점 하나에 결과가 지배당한다.
99번째 백분위수를 최대 깊이로 삼아 튀는 점의 영향을 없앤다.

### `src/depth/depth_from_plane.py` — 전체를 엮는 곳

| 함수 | 하는 일 |
|---|---|
| `poly_bbox(poly_n)` | polygon 점들의 min/max로 bbox 계산 |
| `measure_one(poly_n, world_xyz)` | **측정 본체.** 마스크 -> NaN 제거 -> z-band 필터 -> RANSAC -> 깊이 |

`measure_one()`은 예외 대신 status 문자열을 돌려준다. 폴더를 일괄 처리할 때
한 장이 실패해도 나머지가 계속 돌아야 하기 때문이다.

| status | 뜻 |
|---|---|
| `ok` | 정상 측정 |
| `no_points` | 유효한 점이 없음 (LiDAR 측정 실패 픽셀 과다) |
| `no_plane_in_band` | z-band 필터 후 평면 후보 부족 (배경만 잡힘) |
| `ransac_fail` | 평면을 찾지 못함 |

### `src/viz/plane_viz.py` — 3D 시각화

| 함수 | 하는 일 |
|---|---|
| `create_plane_surface_pcd(plane_model, inlier_cloud, grid_size=50)` | 평면을 50x50 회색 격자 점으로 |
| `create_depth_line_pcd(plane_model, max_depth_point, num_points=100)` | 최대 깊이 지점 -> 평면까지 노란 선 |

**측정값에는 전혀 관여하지 않는다.** 이 파일을 지워도 깊이 숫자는 같다.

### `src/io/ply.py` — PLY 파일

| 함수 | 하는 일 |
|---|---|
| `extract_dent_3d_points(world_xyz_arr, seg_mask)` | 마스크 영역의 3D 점만 추출 |
| `save_pointcloud_ply(points_np, output_path)` | 점 배열 -> PLY 파일 |
| `build_combined(image_path, depth_path, weights)` | **측정 + 시각화를 한 번에.** 4색 합본 PointCloud 반환 |

`build_combined()`는 `measure_one()`과 같은 계산을 하되, 중간 산출물을
버리지 않고 색을 입혀 합친다.

| 색 | 무엇 |
|---|---|
| 파랑 | 정상면 (RANSAC이 평면으로 인정한 점) |
| 빨강 | 덴트 |
| 회색 | 찾아낸 평면 |
| 노랑 | 최대 깊이 지점에서 평면까지의 선 |

### `scripts/measure.py` — 실행 진입점

| 함수 | 하는 일 |
|---|---|
| `collect_pairs(input_path, depth_path=None)` | 경로 -> `(jpg, json)` 쌍 목록. 파일이면 1쌍, 폴더면 전체 |
| `measure_file(model, image_path, json_path, rgb_h)` | 이미지 1장 측정 |
| `parse_args()` | 터미널 인자 등록 |
| `main()` | 전체 실행 + 출력 + CSV 저장 |

### `scripts/export_ply.py` / `scripts/train.py`

| 함수 | 하는 일 |
|---|---|
| `export_ply.main()` | `build_combined()` 호출 후 PLY 저장 |
| `train.main()` | YOLO 모델 생성 후 학습 |

### `experiments/run_dir.py`

| 함수 | 하는 일 |
|---|---|
| `make_run_dir(tag="")` | `results/20260903_1615_태그/` 생성 |
| `save_run_info(run_dir, **info)` | 그 실행에 쓴 파라미터를 `run_info.txt`로 기록 |

---

## 6. 실행할 때 무슨 일이 일어나는가

### `python -m scripts.measure --input a.jpg --weights best.pt`

```
main()
 │
 ├─ parse_args()                          인자 읽기
 ├─ collect_pairs("a.jpg")                -> [("a.jpg", "a.json")]
 ├─ YOLO(weights)                         모델 로드 (한 번만)
 │
 └─ measure_file(model, "a.jpg", "a.json")
     │
     ├─ run_inference(model, "a.jpg")             [1] YOLO 추론
     ├─ select_best_polygon(result)               -> poly_n (N,2)
     ├─ load_xyz("a.json")                        [3] -> world_xyz (256,192,3)
     │
     └─ measure_one(poly_n, world_xyz)
         │
         ├─ poly_bbox(poly_n)                     -> bbox_n
         ├─ create_masks(poly_n, bbox_n, H, W)    [2]
         │   └─ expand_bbox(bbox_n, 0.80)
         │                                        -> seg_mask, inlier_mask
         ├─ (마스크로 점 추출 + NaN 제거)
         ├─ (z-band 필터로 배경 제거)
         ├─ fit_plane_to_inliers(plane_pts)       [4] -> plane_model
         └─ calculate_dent_depth_yolo(...)        [5] -> 40.75, 17.47
```

폴더를 넣으면 `measure_file()`부터가 파일 수만큼 반복된다.
모델 로드는 한 번만 하므로 여러 장을 처리해도 로딩 시간은 한 번이다.

### `python -m scripts.export_ply --image a.jpg --depth a.json --weights best.pt`

```
main()
 └─ build_combined("a.jpg", "a.json", "best.pt")
     ├─ run_inference / select_best_polygon       [1]
     ├─ load_xyz                                  [3]
     ├─ poly_bbox / create_masks                  [2]
     ├─ fit_plane_to_inliers                      [4]
     ├─ calculate_dent_depth_yolo                 [5]
     ├─ create_plane_surface_pcd                  회색 격자
     ├─ create_depth_line_pcd                     노란 선
     └─ (색칠 + 합치기)
 └─ o3d.io.write_point_cloud(out, cloud)
```

`measure.py`와 계산 내용은 같고, 중간 결과를 버리지 않는 점만 다르다.

### `python -m scripts.train --data data.yaml`

```
main()
 ├─ parse_args()
 ├─ YOLO("configs/seg-model.yaml")        구조 yaml로 새 모델 생성
 ├─ model.info()
 └─ model.train(...)                      400 epoch 학습
```

---

## 7. 데이터가 어떻게 변하는가

```
a.json
  │  read_json()
  ▼
{"Width": 256, "Height": 192, "Depth": [0.77, 0.79, ...], "fl": {...}}
  │  reshape(192, 256)
  ▼
z (192, 256)                     거리값만 있는 2차원 배열
  │  meshgrid로 픽셀 좌표표 만들고 핀홀 역투영
  ▼
x, y, z 각각 (192, 256)
  │  np.stack
  ▼
(192, 256, 3)                    픽셀마다 [x, y, z]
  │  np.rot90(k=-1)              RGB와 방향 맞추기
  ▼
world_xyz (256, 192, 3)          <- 여기부터 측정에 쓰임


a.jpg
  │  YOLO
  ▼
poly_n (N, 2)                    0~1 정규화 좌표
  │  * (W, H) = (192, 256)
  ▼
seg_mask, inlier_mask (256, 192) uint8, 값은 0 또는 255
  │  world_xyz[mask == 255]
  ▼
dent_pts (M, 3), plane_pts (K, 3)
  │  RANSAC
  ▼
plane_model [a, b, c, d]
  │  |a*px + b*py + c*pz + d| / |(a,b,c)|
  ▼
40.75 mm
```

### 해상도가 다른데 어떻게 맞추는가

RGB는 1440x1920, depth는 192x256으로 **7.5배 차이**가 난다.
픽셀 번호로는 대응시킬 수 없어 **비율(0~1)** 을 거친다.

```
덴트가 RGB 450번 픽셀에 있음
450 / 1440 = 0.3125        <- YOLO가 이미 나눠서 준다 (xyn)
0.3125 * 192 = 60          <- create_masks()에서 곱한다
-> depth 60번 칸
```

크기를 맞추는 게 아니라 **성긴 격자에 맞춰 읽는 것**이다.
depth 한 칸이 RGB 약 56픽셀(7.5x7.5)을 대표하므로, 덴트가 작거나
멀리 있으면 덮는 칸이 줄어 정확도가 떨어진다.

---

## 8. 모듈 의존 관계

```
scripts/measure.py    ──> detect/yolo_seg
                      ──> io/read_depth
                      ──> depth/depth_from_plane

scripts/export_ply.py ──> io/ply

scripts/train.py      ──> (ultralytics만)

io/ply.py             ──> detect/yolo_seg, io/read_depth,
                          depth/masks, depth/run_ransac,
                          depth/depth_from_plane, viz/plane_viz

depth/depth_from_plane ──> depth/masks, depth/run_ransac

depth/masks.py         ──> (numpy, cv2)
depth/run_ransac.py    ──> (numpy, open3d)
io/read_depth.py       ──> (numpy)
viz/plane_viz.py       ──> (numpy, open3d)
detect/yolo_seg.py     ──> (없음)
```

- 순환 참조 없음
- `scripts/`는 `src/`만 참조하고, 그 반대는 없음
- 아래쪽 5개 모듈은 다른 내부 모듈에 의존하지 않아 단독으로 테스트 가능

`__init__.py`는 두지 않았다. Python 3.3+의 namespace package 기능으로
없어도 `from src.depth.masks import ...`가 동작한다.

---

## 9. 파라미터

측정 결과에 영향을 주는 값들. `configs/default.yaml`에도 같은 값이 있지만
현재 코드는 각 파일의 상수를 읽는다 (검증 후 config 연결 예정).

| 값 | 위치 | 뜻 |
|---|---|---|
| `IMG_SIZE = 640` | `detect/yolo_seg.py` | YOLO 추론 입력 크기 |
| `PADDING_RATIO = 0.80` | `depth/masks.py` | bbox를 얼마나 넓혀 정상면을 확보할지 |
| `DILATION_ITER = 1` | `depth/masks.py` | 덴트를 몇 번 부풀려 정상면에서 뺄지 |
| `FILL_WITH_HULL = True` | `depth/masks.py` | polygon을 볼록 껍질로 채울지 |
| `DENT_INCLUDE_BBOX = True` | `depth/masks.py` | 덴트 영역에 bbox까지 포함할지 |
| `RANSAC_DIST = 0.015` | `depth/depth_from_plane.py` | 평면 위로 인정할 거리 (15mm) |
| `RANSAC_N = 3` | `depth/depth_from_plane.py` | 평면 결정에 필요한 점 수 |
| `RANSAC_ITERS = 100000` | `depth/depth_from_plane.py` | RANSAC 반복 횟수 |
| `Z_BAND_M = 0.05` | `depth/depth_from_plane.py` | 배경 제거용 깊이 범위 (50mm) |

**이 값들이 논문 결과(50cm -> 40.754mm)를 만든 설정이다. 바꾸면 숫자가 달라진다.**

---

## 10. 아직 쓰이지 않는 함수

| 함수 | 언제 쓰이나 |
|---|---|
| `io/polygon.save_polygon_sidecar()` | `experiments/convert_iphone.py` 이관 시 |
| `io/ply.extract_dent_3d_points()` | 동상 |
| `io/ply.save_pointcloud_ply()` | 동상 |

실험 스크립트가 polygon을 미리 저장해두고 재사용하는 구조라서,
`experiments/` 정비가 끝나면 연결된다.

---

## 11. 알아두면 좋은 것

**출력에 "파손 방향: 바깥쪽(bump)"이 뜨는 이유**
`run_ransac.py`에서 평면 법선을 +Z(카메라에서 멀어지는 쪽)로 통일하는데,
덴트 점은 카메라에서 더 멀어 부호가 양수가 되고 라벨이 뒤집혀 찍힌다.
표시 문구일 뿐이며 깊이 값은 절댓값을 쓰므로 영향이 없다.

**거리에 따른 정확도**
덴트가 덮는 depth 칸 수가 거리에 따라 급격히 줄어든다.

| 거리 | 덴트가 덮는 칸 | 측정 깊이 (GT 43mm) |
|---|---|---|
| 50cm | 1,026 | 40.75 mm |
| 100cm | 292 | 23.30 mm |
| 200cm | 89 | 4.22 mm |

칸이 적으면 가장 깊은 지점이 격자 사이로 빠져나간다.
**50~70cm에서 촬영해야 한다.**
