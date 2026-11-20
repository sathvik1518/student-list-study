"""Run the whole study:   python run_study.py
Reads data/students_raw.csv (never changes it), writes everything else into outputs/."""
import sys
from pathlib import Path

from study.pipeline import run

ROOT = Path(__file__).parent

if __name__ == "__main__":
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "students_raw.csv"
    s = run(raw, ROOT / "outputs")
    f, c = s["fit"], s["corr"]
    print(f"Raw rows {s['raw_rows']}  ->  analysis rows {s['analysis_rows']}  (removed {s['removed_rows']}, all listed in outputs/removed_rows.csv)")
    print(f"Hours vs final: Pearson {c['pearson']:.2f}  Spearman {c['spearman']:.2f}  (n={c['n']})")
    print(f"Tests run: {s['tests_run']}.  Gradient descent vs lstsq: max difference {f['max_difference']:.2e}")
    print(f"Test RMSE {f['test_rmse']:.2f} vs 'always guess the average' {f['baseline_rmse']:.2f}")
    print("Raw file: unchanged (sha256 checked).")
    print("Open outputs/report.md")
