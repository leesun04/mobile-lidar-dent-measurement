
#CSV 저장 함수
def run(distances_filter=None, limit_per_group=None, csv_path=CSV_OUT, quiet=False):
    rows = []
    counts = {}  # (sensor, distance) -> processed_count

    for sensor, distance, img_path, sidecar_path, depth_loader in iter_jobs(distances_filter):
        key = (sensor, distance)
        if limit_per_group and counts.get(key, 0) >= limit_per_group:
            continue
        counts[key] = counts.get(key, 0) + 1

        stem = Path(img_path).stem

        # 1) Sidecar 확인 (convert가 segmentation에 성공한 것만)
        poly_n = load_polygon_sidecar(sidecar_path)
        if poly_n is None:
            rows.append(dict(sensor=sensor, distance=distance, file=stem,
                             status="no_sidecar", max_depth_mm="", mean_depth_mm=""))
            if not quiet:
                print(f"[{sensor} {distance}cm] {stem}  no_sidecar (convert 미성공)")
            continue

        # 2) depth 로딩
        try:
            world_xyz = depth_loader()
        except Exception as e:
            rows.append(dict(sensor=sensor, distance=distance, file=stem,
                             status=f"depth_err:{type(e).__name__}",
                             max_depth_mm="", mean_depth_mm=""))
            print(f"[{sensor} {distance}cm] {stem}  depth load 실패: {e}")
            continue

        # 3) 깊이 측정
        try:
            status, max_d, mean_d = measure_one(poly_n, world_xyz)
        except Exception as e:
            status, max_d, mean_d = f"error:{type(e).__name__}", float("nan"), float("nan")
            if not quiet:
                print(f"[{sensor} {distance}cm] {stem}  예외: {e}")

        rows.append(dict(
            sensor=sensor, distance=distance, file=stem, status=status,
            max_depth_mm=("" if np.isnan(max_d) else f"{max_d:.3f}"),
            mean_depth_mm=("" if np.isnan(mean_d) else f"{mean_d:.3f}"),
        ))
        if not quiet:
            tag = status if status != "ok" else f"max={max_d:.2f}mm mean={mean_d:.2f}mm"
            print(f"[{sensor} {distance}cm] {stem}  {tag}")

    # CSV 저장
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sensor", "distance", "file", "status",
                                          "max_depth_mm", "mean_depth_mm"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV 저장: {csv_path}  ({len(rows)} rows)")

    # (sensor, distance) 요약
    print("\n=== (센서, 거리) 평균 ===")
    print(f"참고: 실측 dent 크기 = {GT_LENGTH_CM}cm (= {GT_LENGTH_CM*10:.0f}mm).\n"
          f"      아래는 RANSAC 평면 기준 P99 max depth와 mean depth(둘 다 mm).\n")
    print(f"{'Sensor':<8}{'Dist':>5}  {'n_ok/n':>7}  "
          f"{'mean(max_d) mm':>15}  {'mean(mean_d) mm':>17}  {'std(max_d)':>11}")
    print("-" * 75)

    groups = {}
    for r in rows:
        key = (r["sensor"], r["distance"])
        groups.setdefault(key, []).append(r)

    summary = []
    for key in sorted(groups.keys(), key=lambda k: (k[0], k[1])):
        sensor, distance = key
        grp = groups[key]
        oks = [g for g in grp if g["status"] == "ok"]
        n_total = len(grp)
        n_ok = len(oks)
        if oks:
            maxs = [float(g["max_depth_mm"]) for g in oks]
            means = [float(g["mean_depth_mm"]) for g in oks]
            avg_max = statistics.mean(maxs)
            avg_mean = statistics.mean(means)
            std_max = statistics.stdev(maxs) if len(maxs) > 1 else 0.0
            print(f"{sensor:<8}{distance:>5}  {n_ok:>3}/{n_total:<3}  "
                  f"{avg_max:>15.3f}  {avg_mean:>17.3f}  {std_max:>11.3f}")
            summary.append(dict(sensor=sensor, distance=distance,
                                n_ok=n_ok, n_total=n_total,
                                avg_max=avg_max, avg_mean=avg_mean, std_max=std_max))
        else:
            print(f"{sensor:<8}{distance:>5}  {n_ok:>3}/{n_total:<3}  "
                  f"{'-':>15}  {'-':>17}  {'-':>11}")
            summary.append(dict(sensor=sensor, distance=distance,
                                n_ok=0, n_total=n_total,
                                avg_max=None, avg_mean=None, std_max=None))

    # 요약 CSV
    summary_path = csv_path.replace(".csv", "_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sensor", "distance", "n_ok", "n_total",
                                          "avg_max_depth_mm", "avg_mean_depth_mm",
                                          "std_max_depth_mm"])
        w.writeheader()
        for s in summary:
            w.writerow({
                "sensor": s["sensor"],
                "distance": s["distance"],
                "n_ok": s["n_ok"],
                "n_total": s["n_total"],
                "avg_max_depth_mm": "" if s["avg_max"] is None else f"{s['avg_max']:.3f}",
                "avg_mean_depth_mm": "" if s["avg_mean"] is None else f"{s['avg_mean']:.3f}",
                "std_max_depth_mm": "" if s["std_max"] is None else f"{s['std_max']:.3f}",
            })
    print(f"요약 CSV 저장: {summary_path}")