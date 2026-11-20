"""Step 0: make data/students_raw.csv, a deliberately MESSY file, the same every time (fixed seed).

Run it once. After that the raw file is never changed by anything in this project: the study only READS it,
and a fingerprint (sha256) in data/students_raw.sha256 proves it.

The mess on purpose:
  * repeated rows (same student twice, one of them with different numbers)
  * numbers written as text: "12 hrs", "78%", "ten"
  * section spelled many ways: "A", "a", " Sec A", "section b", "B-"
  * impossible values: 940 study hours, negative hours, a score of 130
  * gaps: empty, "NA", "n/a"
"""
import csv
import hashlib
import random
from pathlib import Path

ROOT = Path(__file__).parent
SEED = 51
N_STUDENTS = 240

WORDS = {3: "three", 5: "five", 8: "eight", 10: "ten", 12: "twelve", 15: "fifteen"}
SECTION_SPELLINGS = {
    "A": ["A", "A", "A", "a", " A ", "Sec A", "section a"],
    "B": ["B", "B", "B", "b", "B ", "Sec B", "section b", "B-"],
    "C": ["C", "C", "C", "c", " C", "Sec C", "section c"],
}
SECTION_BONUS = {"A": 0.0, "B": 1.5, "C": -1.0}  # a small real difference, easy to miss on purpose


def make_rows(rng: random.Random) -> list[dict]:
    rows = []
    for i in range(1, N_STUDENTS + 1):
        section = rng.choice("ABC")
        prior = min(100.0, max(20.0, rng.gauss(62, 14)))
        hours = min(25.0, max(0.0, rng.gauss(8 + 0.04 * (prior - 62), 3.5)))   # students with a higher prior study a bit more
        final = 28 + 0.40 * prior + 1.6 * hours + SECTION_BONUS[section] + rng.gauss(0, 7)
        final = min(100.0, max(0.0, final))
        rows.append({"student_id": f"S{i:03d}", "section": section, "hours": round(hours, 1), "prior": round(prior), "final": round(final)})
    return rows


def dirty(rows: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        r["section"] = rng.choice(SECTION_SPELLINGS[r["section"]])
        out.append(r)

    def pick(k, exclude=()):
        return rng.sample([i for i in range(len(out)) if i not in exclude], k)

    used: set[int] = set()
    for i in pick(9, used):           # hours as text, "12 hrs" / "8h" / a word
        used.add(i)
        h = out[i]["hours"]
        out[i]["hours"] = WORDS[min(WORDS, key=lambda w: abs(w - h))] if rng.random() < 0.4 else f"{round(h)} {rng.choice(['hrs', 'h', 'hours'])}"
    for i in pick(7, used):           # a final score written with a percent sign
        used.add(i)
        out[i]["final"] = f"{out[i]['final']}%"
    for i in pick(12, used):          # gaps
        used.add(i)
        out[i][rng.choice(["hours", "prior", "final"])] = rng.choice(["", "NA", "n/a"])
    for i in pick(2, used):           # a student with no final score at all
        used.add(i)
        out[i]["final"] = ""
    # impossible values
    j = pick(1, used)[0]; used.add(j); out[j]["hours"] = 940
    j = pick(1, used)[0]; used.add(j); out[j]["hours"] = -3
    j = pick(1, used)[0]; used.add(j); out[j]["hours"] = 120
    j = pick(1, used)[0]; used.add(j); out[j]["final"] = 130
    j = pick(1, used)[0]; used.add(j); out[j]["prior"] = -5
    # repeated rows: 6 exact copies and 1 copy with a different final score
    for i in pick(6):
        out.append(dict(out[i]))
    k = pick(1)[0]
    clash = dict(out[k])
    try:
        clash["final"] = int(float(str(clash["final"]).rstrip("%"))) + 9
    except ValueError:
        clash["final"] = 77
    out.append(clash)
    rng.shuffle(out)
    return out


def main() -> None:
    rng = random.Random(SEED)
    rows = dirty(make_rows(rng), rng)
    path = ROOT / "data" / "students_raw.csv"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["student_id", "section", "hours", "prior", "final"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "data" / "students_raw.sha256").write_text(digest + "\n")
    print(f"wrote {path} with {len(rows)} rows. sha256 {digest[:16]}...")


if __name__ == "__main__":
    main()
