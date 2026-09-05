> ⚠️ **작업 문서입니다.** 깃허브 공개 전에 이 파일은 프로젝트 소개 README로 교체하고,
> 아래 내용은 `MIGRATION.md`로 옮기거나 삭제하세요.

# iPhone LiDAR 덴트 깊이 측정 — 코드 이관 작업 문서

흩어져 있던 차량 덴트 측정 코드를 이 폴더로 모아 **포트폴리오용 리포**로 정리하는 작업.

- **범위**: iPhone LiDAR **깊이 측정만**
- **제외**: RealSense 비교 · 치수(너비/높이) 측정 · 실험 코드 · 데이터 · 가중치 · 논문 문서
- **목적**: 논문 아카이브가 아니라 "동작하는 프로그램"으로 보이게

---

## 1. 처리 흐름

```
입력: a.jpg (RGB) + a.json (ARKit depth 256x192)
  |
  v
[1] YOLO 세그멘테이션 -> 덴트 polygon          src/detect/yolo_seg.py
[2] polygon -> seg_mask + inlier_mask          src/depth/masks.py
[3] Depth + fl -> 인트린식 역투영 -> 3D 점     src/io/read_depth.py
[4] z-band 필터 -> RANSAC 평면 피팅            src/depth/run_ransac.py
[5] 점-평면 수직거리 -> P99 max depth          src/depth/depth_from_plane.py
  |
  v
출력: 40.75 mm   (선택: 색상 PLY)
```

---

## 2. 폴더 구조 / 진행 상태

```
mobile-lidar-dent-measurement/
├── README.md                     [작업중] 이 문서
├── requirements.txt              [완료]
├── .gitignore                    [완료]
├── configs/
│   ├── default.yaml              [빈 파일]
│   └── seg-model.yaml            [빈 파일]
├── src/
│   ├── io/
│   │   ├── read_depth.py         [없음]
│   │   ├── polygon.py            [없음]
│   │   └── ply.py                [없음]
│   ├── detect/
│   │   └── yolo_seg.py           [빈 파일]
│   ├── depth/
│   │   ├── masks.py              [없음]
│   │   ├── run_ransac.py         [완료]
│   │   └── depth_from_plane.py   [없음]
│   └── viz/
│       └── plane_viz.py          [완료]
└── scripts/
    ├── measure.py                [없음]
    ├── export_ply.py             [없음]
    └── train.py                  [없음]
```

py 9개 · `__init__.py` 안 씀 (PEP 420) · **2 / 9 완료**

---

## 3. 어디서 무엇을 가져오는가

원본은 `/mnt/nas4/lsj/` 아래 3개 폴더에 흩어져 있음. 총 11개 파일.

### 3-1. `test-dent/depth_dent/`

#### `run_ransac.py` -> `src/depth/run_ransac.py` [완료 · 그대로 복사]

RANSAC으로 평면을 찾고 깊이를 재는 핵심 파일.

| 함수 | 기능 |
|---|---|
| `fit_plane_to_inliers()` | 정상면 점군 -> 평면 방정식 `ax+by+cz+d=0`. open3d `segment_plane` 사용. 법선을 카메라 +Z 기준으로 통일해 덴트가 항상 평면 아래로 오게 함. inlier ratio / RMSE 출력 |
| `calculate_dent_depth_yolo()` | 덴트 점들의 평면까지 부호거리 -> **P99 robust max depth** + mean depth (mm). 진짜 최댓값은 노이즈 1개에 지배당하므로 P99 사용 |

> TODO: `print()` 15줄 있음. 검증 끝난 뒤 정리.

#### `create_plane.py` -> `src/viz/plane_viz.py` [완료 · 그대로 복사]

결과를 3D로 보기 위한 시각화. 측정값에는 영향 없음.

| 함수 | 기능 |
|---|---|
| `create_plane_surface_pcd()` | 평면식 + inlier bbox -> 50x50 회색 격자 |
| `create_depth_line_pcd()` | 최대 깊이점 -> 평면 투영점 노란 작대기 (100점 보간) |

### 3-2. `test-dent/point_count/`

#### `convert_iphone.py` -> 함수를 4개 파일로 분산

원래는 "ARKit JSON -> PLY 일괄 변환" 실험 스크립트. 안의 재사용 함수만 추출.

| 함수 | 기능 | 목적지 |
|---|---|---|
| `run_inference()` | YOLO 모델 추론 | `src/detect/yolo_seg.py` |
| `select_best_polygon()` | 여러 detection 중 confidence 최고 하나 선택 | `src/detect/yolo_seg.py` |
| `save_polygon_sidecar()` | polygon을 txt로 저장 (YOLO 재실행 방지) | `src/io/polygon.py` |
| `create_seg_mask()` | polygon -> 마스크 | `src/depth/masks.py` |
| `extract_dent_3d_points()` | 마스크 영역의 3D 점만 추출 | `src/io/ply.py` |
| `save_pointcloud_ply()` | numpy -> PLY 저장 | `src/io/ply.py` |
| `load_depth_iphone()` | ARKit JSON 읽기 | 제외 (중복) |
| 거리별 폴더 순회, `main()` | 실험 로직 | 제외 |

#### `measure_depth_all.py` -> 함수를 3개 파일로 분산

**40.75mm를 실제로 만들어낸 파일.** 가장 중요.

| 함수 | 기능 | 목적지 |
|---|---|---|
| `measure_one()` | 깊이 측정 본체. 마스크 -> NaN 제거 -> z-band 필터 -> RANSAC -> 깊이 | `src/depth/depth_from_plane.py` |
| `poly_bbox()` | polygon -> bbox | `src/depth/depth_from_plane.py` |
| `load_polygon_sidecar()` | 저장된 polygon txt 읽기 | `src/io/polygon.py` |
| `run()` | 폴더 순회 + CSV 저장 | `scripts/measure.py` |
| `expand_bbox()`, `create_masks()` | 마스크 생성 | 제외 (중복) |
| `load_world_iphone()` | ARKit JSON 읽기 | 제외 (중복) |
| `iter_jobs()` 센서 분기, 거리 정규식 | 실험 로직 | 제외 |

#### `extract_best_depth_ply.py` -> `src/io/ply.py` + `scripts/export_ply.py`

| 함수 | 기능 | 목적지 |
|---|---|---|
| `build_combined()` | 4색 합성 PLY — 파랑(정상면) / 빨강(덴트) / 회색(평면) / 노랑(깊이선) | `src/io/ply.py` |
| `pick_best_per_group()` | GT 43mm 최근접 파일 선택 | 제외 (실험 로직) |

### 3-3. `test-dent/` (루트)

#### `test_dent.py` -> `src/depth/masks.py`

단일 이미지 측정 스크립트. 마스크 생성 함수만 추출.

| 함수 | 기능 |
|---|---|
| `expand_bbox()` | 덴트 bbox를 80% 확장 -> 주변 정상면 확보 |
| `create_masks()` | **seg_mask**(덴트) = polygon 채움, convex hull 옵션.<br>**inlier_mask**(정상면) = 확장 bbox − dilate된 seg_mask. dilate는 경계 누수 방지 |

> 이 두 함수가 `test_dent.py` / `measure_depth_all.py` / `bench_pipeline.py` 3곳에 복붙돼 있음.
> 하나로 합치는 것이 이번 정리의 핵심 중 하나.

### 3-4. `yolo-dent/measure_dent/`

#### `read_depth.py` -> `src/io/read_depth.py` [로직 수정 있음]

아이폰 JSON을 3D 점으로 바꾸는 파일. 이게 없으면 RANSAC에 넣을 데이터가 없음.

| 함수 | 기능 |
|---|---|
| `read_json()` | JSON 로드 |
| `get_world_xyz_arr()` | `Depth`(거리값 49,152개) + `fl`(초점거리) -> 역투영 -> 3D 점 배열 |

**수정 내용**

| 원본 | 처리 |
|---|---|
| 인트린식 역투영 `(j-W/2)*z/(fl/7.5)` | 유지 (7.5 = 1920/256) |
| `m00~m33` + `Pos` 월드 좌표 변환 | **제거** — 강체변환이라 점-평면 거리 불변 |
| `np.rot90(k=-1)` | **유지** — 배열 방향 맞추기. 빼면 마스크가 엉뚱한 곳을 가리킴 |
| for문 49,152회 | numpy 벡터화 |
| `fl`을 인자로 받음 | JSON에서 직접 읽음 (촬영마다 다름: 1446.73 ~ 1454.25) |

읽는 JSON 키: `Width`, `Height`, `Depth`, `fl`
안 읽는 키: `m00~m33`, `Rot`, `Pos`

### 3-5. `yolo-dent/depth_dent/`

#### `read_depth.py` -> `src/io/polygon.py`

이름은 같지만 3-4와 다른 파일. polygon 문자열 처리 함수만 가져옴.

| 함수 | 기능 |
|---|---|
| `get_normalized_coords()` | `"0 0.31 0.45 0.33 ..."` 문자열 -> (N,2) 좌표 배열 + bbox |

### 3-6. `yolo-dent/`

| 원본 | 목적지 | 내용 |
|---|---|---|
| `train.py` | `scripts/train.py` | CARDD 학습 (400ep / 1024px / batch 14 / SGD lr 0.001). 절대경로 3개를 `--model`, `--data`, `--project` 인자로 변경 |
| `models/seg-model.yaml` | `configs/seg-model.yaml` | YOLOv11m-seg 구조. 그대로 복사 |

### 3-7. `yolo26/` [완료]

| 원본 | 목적지 | 내용 |
|---|---|---|
| `requirements.txt` | `requirements.txt` | 78줄 -> 실제 쓰는 7개만. `open3d` 추가 |

---

## 4. 원본 -> 목적지 요약

| # | 원본 파일 | 목적지 | 상태 |
|---|---|---|---|
| 1 | `test-dent/depth_dent/run_ransac.py` | `src/depth/run_ransac.py` | 완료 |
| 2 | `test-dent/depth_dent/create_plane.py` | `src/viz/plane_viz.py` | 완료 |
| 3 | `yolo-dent/measure_dent/read_depth.py` | `src/io/read_depth.py` | 변환행렬 제거 |
| 4 | `yolo-dent/depth_dent/read_depth.py` | `src/io/polygon.py` (일부) | 함수 1개 |
| 5 | `test-dent/point_count/convert_iphone.py` | `yolo_seg.py`+`polygon.py`+`ply.py`+`masks.py` | 함수 6개 분산 |
| 6 | `test-dent/point_count/measure_depth_all.py` | `depth_from_plane.py`+`polygon.py`+`measure.py` | 함수 4개 분산 |
| 7 | `test-dent/point_count/extract_best_depth_ply.py` | `src/io/ply.py`+`scripts/export_ply.py` | 함수 1개 |
| 8 | `test-dent/test_dent.py` | `src/depth/masks.py` | 함수 2개 |
| 9 | `yolo-dent/train.py` | `scripts/train.py` | 경로만 수정 |
| 10 | `yolo-dent/models/seg-model.yaml` | `configs/seg-model.yaml` | 그대로 복사 |
| 11 | `yolo26/requirements.txt` | `requirements.txt` | 완료 |

원본 파일 하나가 목적지 여러 곳으로 쪼개지는 이유: `convert_iphone.py`와 `measure_depth_all.py`가
"실험용 배치 스크립트 안에 재사용 가능한 함수가 섞여 있는" 구조이기 때문.
재사용 함수는 `src/`로, 배치 로직은 `scripts/`로 분리한다.

---

## 5. 안 옮기는 것

| 분류 | 대상 |
|---|---|
| RealSense 경로 | `test-dent/depth_dent/read_depth.py`, `test-dent/measure_dent/read_depth.py`, `convert_seosor.py`, 모든 sensor 분기 |
| 치수(너비/높이) 측정 | `test-dent/measure_dent/` 전체, `measure_size_all.py` |
| 실험 · 비교 코드 | `convert_weather.py`, `measure_weather_all.py`, `count_iphone_points.py`, `compare_points.py`, `bench_pipeline.py`, `yolo26/` 나머지 |
| 구버전 · 잔재 | `yolo-dent/test_dent.py`, `test_dent_v1.py`, `test_measure.py`, `val.py`, `test/3D/test.ipynb`, `.ipynb_checkpoints/` |
| 데이터 | raw/converted 1.5GB, `captured_data/` — 샘플도 넣지 않음 |
| 모델 가중치 | 905MB -> GitHub Release |
| 결과 CSV | 숫자는 최종 README 표로만 |
| 논문 · 발표 | `METHODS.md`, `CODE_GUIDE.md`, `RESEARCH_SUMMARY.md`, presentation 25개, docx 2개 |
| 별도 리포 | `side_dent_page/` — `.env`에 카카오 API 키 있으니 주의 |

---

## 6. 고쳐야 할 하드코딩

| 원본 파일 | 줄 | 변경 |
|---|---|---|
| `test-dent/test_dent.py` | 4, 32-35, 70 | 절대경로 4개 + docstring 경로 + `CUDA_VISIBLE_DEVICES` -> CLI 인자 |
| `point_count/convert_iphone.py` | 26-31, 201 | `RAW_ROOT`/`OUT_ROOT`/`MODEL_PATH` -> 인자, `DIST_PATTERN` 삭제 |
| `point_count/measure_depth_all.py` | 30-31, 39-47 | `sys.path.insert` 삭제, RAW_ROOT 4개·거리 정규식 -> `--input` `--csv` |
| `point_count/extract_best_depth_ply.py` | 27-29, 45-49 | `sys.path` 2줄 삭제, `SENSORS`·`GT_DEPTH_MM` 삭제 |
| `yolo-dent/train.py` | 6, 13, 17 | `--model` `--data` `--project` |

**목표 실행 형태**

```bash
python -m scripts.measure    --input a.jpg --depth a.json --weights weights/best.pt
python -m scripts.measure    --input ./data --csv out.csv
python -m scripts.export_ply --input a.jpg --depth a.json --out result.ply
python -m scripts.train      --data data.yaml --model configs/seg-model.yaml
```

---

## 7. configs/default.yaml (채울 내용)

```yaml
camera:
  rgb_h: 1920          # fl 스케일 보정용 (fl, 해상도는 JSON에서 읽음)

mask:
  padding_ratio: 0.80
  dilation_iter: 1
  fill_with_hull: true
  dent_include_bbox: true

ransac:
  distance_threshold: 0.015
  ransac_n: 3
  num_iterations: 100000
  z_band_m: 0.05
```

**이 값들이 40.75mm를 만든 설정. 바꾸면 결과가 달라진다.**

---

## 8. 환경

| 구분 | 내용 |
|---|---|
| 실행 환경 | **다른 서버** · conda env `lsj_yolo` · Python 3.10.19 |
| 정리 환경 | 이 서버 (NAS 마운트). open3d 없어 실행 불가, 파일 작업만 |
| 패키지 | ultralytics 8.4.6 / torch 2.9.1 / open3d 0.19.0 / opencv 4.13.0.90 / numpy 2.2.6 / PyYAML 6.0.3 |

---

## 9. 검증 기준

이관 후 원본 데이터로 실행해 아래 값이 재현되어야 성공.

```
depth_results_summary.csv
iphone,50,24,24,40.754,17.468,6.124
                 ^^^^^^ avg_max_depth_mm
```

검증은 실행 환경(`lsj_yolo`)이 있는 다른 서버에서 수행.

---

## 10. 남은 작업

- [x] `requirements.txt`
- [x] `.gitignore`
- [x] `src/depth/run_ransac.py`
- [x] `src/viz/plane_viz.py`
- [ ] `src/io/read_depth.py`  ← 유일하게 로직 변경
- [ ] `src/io/polygon.py`
- [ ] `src/io/ply.py`
- [ ] `src/detect/yolo_seg.py`
- [ ] `src/depth/masks.py`  ← 3벌 중복 통합
- [ ] `src/depth/depth_from_plane.py`
- [ ] `scripts/measure.py`
- [ ] `scripts/export_ply.py`
- [ ] `scripts/train.py`
- [ ] `configs/default.yaml`
- [ ] `configs/seg-model.yaml`
- [ ] 다른 서버에서 40.754mm 검증
- [ ] README를 공개용 프로젝트 소개로 교체
- [ ] (검증 후) `run_ransac.py` print 정리
