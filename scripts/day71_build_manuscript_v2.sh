#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PAPER_DIR="$ROOT/paper-v2"
OUTPUT_DIR="$ROOT/output/pdf"
AUDIT_DIR="$PAPER_DIR/audits"
SOURCE_COMMIT=${1:-$(git -C "$ROOT" rev-parse HEAD)}

if ! git -C "$ROOT" cat-file -e "$SOURCE_COMMIT^{commit}"; then
  echo "Unknown source commit: $SOURCE_COMMIT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$AUDIT_DIR" "$ROOT/tmp"
BUILD_DIR=$(mktemp -d "$ROOT/tmp/day71-manuscript-v2.XXXXXX")

cleanup() {
  rm -rf "$BUILD_DIR"
}
trap cleanup EXIT INT TERM

SOURCE_DATE_EPOCH=$(git -C "$ROOT" show -s --format=%ct "$SOURCE_COMMIT")
export SOURCE_DATE_EPOCH
export TZ=UTC
export LC_ALL=C

python3 "$ROOT/scripts/day71_build_manuscript_v2_figures.py"

cd "$PAPER_DIR"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" manuscript.tex >"$BUILD_DIR/pass-1.stdout"
cd "$BUILD_DIR"
BIBINPUTS="$PAPER_DIR:${BIBINPUTS:-}" bibtex manuscript >"$BUILD_DIR/bibtex.stdout"
cd "$PAPER_DIR"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" manuscript.tex >"$BUILD_DIR/pass-2.stdout"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" manuscript.tex >"$BUILD_DIR/pass-3.stdout"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" manuscript.tex >"$BUILD_DIR/pass-4.stdout"

FINAL_PDF="$OUTPUT_DIR/neural-chameleon-causal-mechanisms-v2.pdf"
cp "$BUILD_DIR/manuscript.pdf" "$FINAL_PDF"
cp "$BUILD_DIR/manuscript.log" "$AUDIT_DIR/manuscript-v2-build.log"
cp "$BUILD_DIR/manuscript.blg" "$AUDIT_DIR/manuscript-v2-bibtex.log"
MANUSCRIPT_BUILD_DIR="$BUILD_DIR" perl -pi -e 's/\Q$ENV{MANUSCRIPT_BUILD_DIR}\E/<BUILD_DIR>/g; s/[ \t]+$//' \
  "$AUDIT_DIR/manuscript-v2-build.log" "$AUDIT_DIR/manuscript-v2-bibtex.log"

if grep -E "undefined references|Citation .* undefined|Reference .* undefined" "$BUILD_DIR/manuscript.log" >/dev/null 2>&1; then
  echo "Unresolved references remain in manuscript build" >&2
  exit 1
fi

RENDER_DIR="$AUDIT_DIR/rendered-pages"
mkdir -p "$RENDER_DIR"
find "$RENDER_DIR" -maxdepth 1 -type f -name 'page-*.png' -delete
pdftoppm -png -r 120 "$FINAL_PDF" "$RENDER_DIR/page" >/dev/null 2>&1

python3 "$ROOT/scripts/day71_audit_manuscript_v2.py"
PDF_SHA=$(shasum -a 256 "$FINAL_PDF" | awk '{print $1}')
printf 'source_commit=%s\npdf=%s\nsha256=%s\n' "$SOURCE_COMMIT" "$FINAL_PDF" "$PDF_SHA"
