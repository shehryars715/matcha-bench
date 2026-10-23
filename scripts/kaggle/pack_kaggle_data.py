"""Pack the files the Kaggle GPU stage needs into one zip for a private Kaggle dataset.

Included: data/dapt (corpus cache), data/tasks (all tasks), lexicons/lexicon_*.txt.
Not included: raw MATCHA, data/processed, mined candidates and annotation files.
"""

import argparse
import zipfile

import _bootstrap  # noqa: F401
from matcha_bench.config import REPO_ROOT, resolve_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT.parent / "matcha-bench-data.zip"))
    args = ap.parse_args()

    files = []
    for folder in ("data/dapt", "data/tasks"):
        root = REPO_ROOT / folder
        if not root.exists():
            raise SystemExit(f"{root} is missing; build it first (see docs/KAGGLE.md)")
        files += sorted(p for p in root.rglob("*") if p.is_file())
    files += sorted((REPO_ROOT / "lexicons").glob("lexicon_*.txt"))

    out = resolve_path(args.out)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, path.relative_to(REPO_ROOT).as_posix())
    size = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files ({size / 1e6:.0f} MB) -> {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
