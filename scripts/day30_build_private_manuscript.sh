#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PAPER_DIR="$ROOT/paper"
OUTPUT_DIR="$ROOT/output/pdf"
AUDIT_DIR="$PAPER_DIR/audits"
SOURCE_COMMIT=${1:-$(git -C "$ROOT" rev-parse HEAD)}

if ! git -C "$ROOT" cat-file -e "$SOURCE_COMMIT^{commit}"; then
  echo "Unknown source commit: $SOURCE_COMMIT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$AUDIT_DIR" "$ROOT/tmp"
BUILD_DIR=$(mktemp -d "$ROOT/tmp/day30-manuscript.XXXXXX")

cleanup() {
  find "$BUILD_DIR" -mindepth 1 -maxdepth 1 -type f -delete 2>/dev/null || true
  rmdir "$BUILD_DIR" 2>/dev/null || true
  rmdir "$ROOT/tmp" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

SOURCE_DATE_EPOCH=$(git -C "$ROOT" show -s --format=%ct "$SOURCE_COMMIT")
export SOURCE_DATE_EPOCH
export TZ=UTC
export LC_ALL=C

cd "$PAPER_DIR"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" manuscript.tex >"$BUILD_DIR/pass-1.stdout"
BIBINPUTS=".:${BIBINPUTS:-}" bibtex "$BUILD_DIR/manuscript" >"$BUILD_DIR/bibtex.stdout"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" manuscript.tex >"$BUILD_DIR/pass-2.stdout"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" manuscript.tex >"$BUILD_DIR/pass-3.stdout"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" manuscript.tex >"$BUILD_DIR/pass-4.stdout"

FINAL_PDF="$OUTPUT_DIR/neural-chameleon-causal-mechanisms-private.pdf"
cp "$BUILD_DIR/manuscript.pdf" "$FINAL_PDF"
cp "$BUILD_DIR/manuscript.log" "$AUDIT_DIR/day30-manuscript-build.log"
cp "$BUILD_DIR/manuscript.blg" "$AUDIT_DIR/day30-manuscript-bibtex.log"

PDF_SHA=$(shasum -a 256 "$FINAL_PDF" | awk '{print $1}')
printf 'source_commit=%s\npdf=%s\nsha256=%s\n' "$SOURCE_COMMIT" "$FINAL_PDF" "$PDF_SHA"
