# Study the Student List

Question from the teacher: **"Do extra study hours help, and does the section matter?"**
The answer must fit on one page and be safe to trust. It is in `outputs/report.md` (top of the file).

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python run_study.py
python -m pytest
```

Mac / Linux: `source .venv/bin/activate` instead of the Activate line.
Then open `outputs/report.md` (in VS Code: right click → Open Preview) and the three pictures in `outputs/charts/`.

## What each part does

| File | Job |
|---|---|
| `make_raw_data.py` | Step 0. Makes the messy `data/students_raw.csv` (same every time). Run it only if you want to rebuild the file. |
| `data/students_raw.csv` | The raw file. **Nothing in this project ever changes it.** `data/students_raw.sha256` is its fingerprint. |
| `study/load.py` | Reads the file as text and checks the fingerprint did not change. |
| `study/clean.py` | Cleans in memory. Counts every fix. Lists every removed row with a reason. |
| `study/stats.py` | Section table with 95% intervals, one pre-chosen pair test, ANOVA, Pearson, Spearman, Holm correction. |
| `study/model.py` | Train/test split, gap filling from train only, standardising, gradient descent from scratch. |
| `study/charts.py`, `study/report.py` | The pictures and `report.md`. |
| `run_study.py` | Runs everything. |
| `tests/` | 97 tests. |
| `break_it.py` | Breaks the code 29 ways on purpose and checks the tests notice. |

## Outputs (all in `outputs/`)

`report.md` · `section_table.csv` · `removed_rows.csv` · `cleaning_log.csv` · `clean_students.csv` · `train_filled.csv` · `test_filled.csv` · `tests_run.csv` · `summary.json` · `charts/*.png`

## Use your own file

If your lesson has its own `students_raw.csv`, copy it over `data/students_raw.csv` and run `python run_study.py data/students_raw.csv`.
It needs the columns `student_id, section, hours, prior, final`. Then update the fingerprint:
`python -c "import hashlib,pathlib;p=pathlib.Path('data/students_raw.csv');pathlib.Path('data/students_raw.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n')"`
(The tests that compare counts to this stand-in file will then fail. That is expected.)

## Windows note

`.gitattributes` stops Git from changing line endings in the raw file. Without it, Git on Windows could change the file and break the fingerprint check.
