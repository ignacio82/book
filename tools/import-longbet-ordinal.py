"""Import compact, verified results from the completed ordinal experiment.

Usage: python tools/import-longbet-ordinal.py /path/to/ordinal-comparison
Requires NumPy. This exports existing fits; it does not select or refit panels.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("experiment", type=Path)
args = parser.parse_args()
source = args.experiment.resolve()
target = Path(__file__).resolve().parents[1] / "data" / "longbet-ordinal"
target.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = json.loads((source / "manifest.json").read_text())
for name, expected in manifest.items():
    assert sha(source / name) == expected, f"Changed source artifact: {name}"
summary = json.loads((source / "summary.json").read_text())
assert summary["seeds"] == list(range(91000, 91020))
assert summary["scenarios"] == ["rare_top", "dispersion"]
panels = list(csv.DictReader((source / "panel-results.csv").open()))
assert len(panels) == 180
rare_longbet = [r for r in panels if r["scenario"] == "rare_top" and "longbet" in r["method"]]
assert len(rare_longbet) == 40
rare_failures = {m: sum(int(r["diagnostic_failures"]) > 0 for r in rare_longbet if r["method"] == m)
                 for m in ("ordinal_longbet", "binary_longbet")}
shutil.copyfile(source / "panel-results.csv", target / "panel-results.csv")

# Use the first prespecified seed, exposure 4, preserving chain-major order.
draw_file = source / "results" / "rare_top_91000" / "ordinal_longbet.npz"
draws = np.load(draw_file)["att_draws"]
n_draws = draws.shape[-1]
assert draws.shape[:2] == (6, 5) and n_draws % 4 == 0
per_chain = n_draws // 4
assert np.max(np.abs(draws.sum(axis=1))) < 1e-10
stream = io.StringIO()
writer = csv.writer(stream, lineterminator="\n")
writer.writerow(["chain", "iteration", "top_att", "lowest_att"])
for d in range(n_draws):
    writer.writerow([d // per_chain + 1, d % per_chain + 1,
                     format(draws[3, 4, d], ".17g"), format(draws[3, 0, d], ".17g")])
with (target / "decision-draws.csv.gz").open("wb") as f:
    with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:
        g.write(stream.getvalue().encode())

shares = []
for seed in summary["seeds"]:
    counts = json.loads((source / "results" / f"rare_top_{seed}" / "metadata.json").read_text())["category_counts"]
    shares.append(counts[-1] / sum(counts))
provenance = {
    "experiment_date": summary.get("experiment_date", "2026-09-14"),
    "source_manifest_sha256": sha(source / "manifest.json"),
    "python_source_sha256": summary["source_sha256"],
    "source_panel_results_sha256": sha(source / "panel-results.csv"),
    "source_decision_draws_sha256": sha(draw_file),
    "versions": summary["versions"],
    "settings": json.loads((source / "run-settings.json").read_text()),
    "primary": summary["primary"][0],
    "mean_training_top_share": float(np.mean(shares)),
    "rare_top_failed_fits": rare_failures,
    "decision_example": {"seed": 91000, "exposure": 4, "chains": 4, "retained_per_chain": per_chain,
                         "top_gain": 0.03, "lowest_reduction": 0.20,
                         "thresholds_are_illustrative": True},
}
(target / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
files = ["panel-results.csv", "decision-draws.csv.gz", "provenance.json"]
with (target / "checksums.csv").open("w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["file", "sha256"])
    writer.writerows((name, sha(target / name)) for name in files)
print(f"Verified {len(manifest)} source hashes; exported 40 panels and {n_draws:,} paired decision draws.")
