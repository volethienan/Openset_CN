"""Check completion gates for the five Access audit amendments; no model is trained."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'exp/reports/access'
    failures, checks = [], []

    def check(name, ok, detail):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))
        if not ok:
            failures.append(name)

    for name in ('manifest.json', 'simargl_raw_audit.json', 'feature_availability.json', 'benchmark_k5.json'):
        check(f'artifact:{name}', (out / name).exists(), str(out / name))
    if failures:
        raise RuntimeError('Missing required artifacts: ' + ', '.join(failures))
    manifest = json.loads((out / 'manifest.json').read_text())
    check('complete_main_runs', manifest['collected_runs'] == 280 and not manifest['missing'], manifest['collected_runs'])
    check('source_hashes_match', all(hashlib.sha256((root / 'exp' / name).read_bytes()).hexdigest() == expected
                                   for name, expected in manifest['code_hashes'].items()), 'source files match evidence export')
    hashes = json.loads((out / 'result_hashes.json').read_text())
    unchanged = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
                    for pair in hashes for name, expected in pair.items())
    check('saved_results_preserved', unchanged, f'{sum(map(len, hashes))} saved NPZ/JSON hashes')
    raw = json.loads((out / 'simargl_raw_audit.json').read_text())
    check('raw_full_coverage', raw['counts']['rows'] == raw['parquet_rows'], raw['counts'])
    check('bounded_exclusion_equivalent', raw['masks_identical'], raw['counts'])
    check('suspected_not_verified_labels', raw['policy']['status'] == 'suspected_label_anomaly_not_verified_ground_truth', raw['policy'])
    bench = json.loads((out / 'benchmark_k5.json').read_text())
    check('benchmark_K5', bench['n_prior'] == 5 and all(r['cfg']['n_prior'] == 5 and r['models_trained'] == 5 for r in bench['rows']), bench['n_prior'])
    check('benchmark_three_datasets', {r['dataset'] for r in bench['rows']} == {'CIDDS_ctx', 'ToN_ctx', 'CIC18'}, len(bench['rows']))
    check('benchmark_repeat_stability', all(r['max_repeat_score_diff'] == 0 and len(r['score_seconds']) == bench['repeats'] for r in bench['rows']), 'all repeats identical scores')
    avail = json.loads((out / 'feature_availability.json').read_text())
    check('feature_scope_honest', avail['protocol'] == 'retrospective_complete_flow_start_order' and not avail['packet_causal_verified'], avail['protocol'])
    rows = pd.read_csv(out / 'main_runs.csv')
    metrics = ['F1', 'MCC', 'AUROC', 'zd_recall_at1fpr', 'FPR', 'macro_zd_first_period_at1fpr']
    rows.groupby(['dataset', 'method'])[metrics].agg(['mean', 'std', 'count']).to_csv(out / 'dataset_seed_variability.csv')
    check('unique_result_keys', not rows.duplicated(['dataset', 'method', 'seed']).any(), len(rows))
    check('all_cohort_methods_have_five_seeds', rows.groupby(['dataset', 'method']).seed.nunique().eq(5).all(), len(rows.groupby(['dataset', 'method'])))
    check('main_metrics_finite', np.isfinite(rows[['F1', 'MCC', 'AUROC', 'zd_recall_at1fpr']].to_numpy()).all(), 'finite primary metrics')
    check('metric_scope_explicit', rows.zero_day_scope.str.contains('including_post_feedback').all() and rows.roc_threshold_source.str.contains('diagnostic').all(), 'whole-stream and ROC definitions')
    tests = pd.read_csv(out / 'dataset_tests.csv')
    check('dataset_inference_units', tests.groupby('cohort')['n'].first().to_dict() == {
        'historical_filtered_7': 7., 'published_labels_7': 7., 'without_uncertain_SIMARGL_6': 6.}, '7/7/6 units, never 35 seed units')
    check('correction_never_reduces_p', (tests.p_two_sided_holm + 1e-12 >= tests.p_two_sided).all(), 'Holm within metric')
    hist = tests[(tests.cohort == 'historical_filtered_7') & (tests.baseline == 'ssf_causal')].set_index('metric')
    check('historical_statistics_reproduced', np.isclose(hist.loc['F1', 'p_one_sided'], .109375)
          and np.isclose(hist.loc['AUROC', 'p_one_sided'], .0078125), 'F1=.109375; AUROC=.0078125')
    contract = (root / 'ACCESS_PROTOCOL.md').read_text(encoding='utf-8')
    check('limitations_written', all(x in contract for x in ('ToN', 'CIC17_full', 'camouflage', 'post-feedback', 'production readiness')), 'claim boundaries present')
    result = dict(status='PASS' if not failures else 'FAIL', meaning='audit_amendments_complete_not_journal_acceptance_or_packet_causal_validation', checks=checks, failures=failures)
    (out / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not failures:
        lines = ['# Xử lý 5 điểm trong audit IEEE Access', '', '## Material Passport', '',
                 '- Mode: run + validate.',
                 '- Evidence: raw SIMARGL, saved predictions, source, three new initial-fit K5 benchmarks.',
                 '- Verification: các gate trong verification.json; không xác nhận lại toàn bộ campaign huấn luyện.',
                 '- Scope: retrospective complete-flow replay; chưa xác nhận feature-level packet causality.', '',
                 '## 1. SIMARGL', '',
                 f'Đã duyệt {raw["counts"]["rows"]:,} raw flow. Quy tắc bounded theo host, khoảng thời gian và nhãn gốc Normal flow '
                 f'chọn đúng {raw["counts"]["bounded_excluded"]:,} flow như quy tắc cũ; chênh lệch bằng 0. '
                 'Đã sửa predicate, kiểm thử hai cận thời gian và bảo toàn các nhãn attack. '
                 'Vì mask không đổi, không cần rebuild/rerun các stream hiện có cho thay đổi này.',
                 'Raw protocol/flag/packet/port evidence hỗ trợ nghi vấn nhãn, không được xem là ground truth attack đã xác nhận. '
                 'Báo cáo đồng thời filtered, published labels và bỏ SIMARGL.', '',
                 '## 2. Thống kê', '',
                 'evaluate.py, check_criteria.py và real_report.py dùng chung inference.py. '
                 'Ghép cùng seed trước, lấy trung bình trong dataset; không dùng dataset×seed làm đơn vị độc lập. '
                 'Xuất hai hướng kiểm định, Holm trong metric và sensitivity trên 20 so sánh. '
                 'Có kiểm thử nhân đôi seed không tăng n hoặc giảm p, thiếu seed không làm lệch pairing, và các scenario cùng dataset không tăng n.', '',
                 '## 3. Benchmark đúng K5', '',
                 '| Dataset | Số GBM | Threads | Batch | Fit đầu (s) | Median flow/s |',
                 '|---|---:|---:|---:|---:|---:|']
        for row in bench['rows']:
            lines.append(f'| {row["dataset"]} | {row["models_trained"]} | {bench["threads"]} | {row["batch_size"]:,} | '
                         f'{row["initial_fit_s"]:.2f} | {row["median_flows_per_s"]:,.0f} |')
        lines += ['', 'Đo 5 lần sau warmup, dùng median, lặp lại cho điểm số giống hệt. '
                  'Chỉ đo scoring feature vectors, không phải tốc độ end-to-end, preprocessing, clustering hay analyst. '
                  'Không dùng benchmark K1 cũ để nói về K5.', '', '## 4. Thời điểm khả dụng của feature', '',
                  'Đã ghi contract trong source, metadata và ACCESS_PROTOCOL.md; audit duration/overlap được lưu riêng. '
                  'Kết quả hiện tại được giới hạn thành replay flow hoàn chỉnh theo start timestamp với predict-then-update. '
                  '**Chưa chuyển thành campaign theo completion/export time; không tuyên bố packet-causal hoặc phát hiện ngay lúc flow bắt đầu.** '
                  'Một campaign strict availability-time mới cần dựng lại context từ full raw data và rerun tất cả phương pháp bị ảnh hưởng, '
                  'không thể được thay bằng đổi tên cột hay sắp xếp lại sampled stream.', '',
                  '## 5. Phạm vi kết luận', '',
                  'Đã tách ROC@1% diagnostic khỏi operating FPR; bổ sung macro theo họ và first-update-period recall, '
                  'đồng thời ghi rõ whole-stream zero-day gồm post-feedback. '
                  'Giữ giới hạn ToN, CIC17_full, group camouflage, số quyết định so với flow lan nhãn và chi phí analyst chưa đo. '
                  'Đã sửa mô tả thuật toán: fresh prior mỗi refit, cập nhật P/A qua feedback; không nhận recursive-kernel update '
                  'hoặc locality của GBM scores là bảo đảm.', '',
                  '## Kết quả tổng hợp theo ba cohort', '']
        cohorts = pd.read_csv(out / 'cohort_means.csv').query('method == "ernp_K5"')
        lines += ['| Cohort | F1 | AUROC | Pooled ZD@1% | Macro ZD first-period@1% |', '|---|---:|---:|---:|---:|']
        for _, row in cohorts.iterrows():
            lines.append(f'| {row["cohort"]} | {row["F1"]:.4f} | {row["AUROC"]:.4f} | {row["zd_recall_at1fpr"]:.4f} | '
                         f'{row["macro_zd_first_period_at1fpr"]:.4f} |')
        lines += ['', 'Các cohort là phân tích độ nhạy hậu kiểm và phải trình bày cùng nhau, không chọn cohort có kết quả tốt nhất. '
                  'Hoàn tất audit amendments không đồng nghĩa bài chắc chắn được IEEE Access nhận hoặc novelty đã được chứng nhận. '
                  'Bản thảo không bị chỉnh sửa trong đợt này.', '']
        (out / 'RESOLUTION.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if failures:
        raise RuntimeError(', '.join(failures))


if __name__ == '__main__':
    main()
