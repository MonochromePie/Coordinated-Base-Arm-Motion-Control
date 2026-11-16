#!/usr/bin/env bash
# Build a report: pdfLaTeX, BibTeX (if the .tex uses \bibliography), pdfLaTeX twice.
# The PDF is written next to the .tex; aux/log/bbl files go to <report>/build/.
#
#   ./report/build.sh report/final_draft/main.tex    # one report
#   ./report/build.sh --all                          # every report/*/main.tex
set -uo pipefail

build() {
  local tex="$1"
  local dir name
  dir="$(cd "$(dirname "$tex")" && pwd)" || return 1
  name="$(basename "$tex" .tex)"
  [[ -f "$dir/$name.tex" ]] || { echo "build: $dir/$name.tex not found"; return 1; }
  mkdir -p "$dir/build"

  pass() {
    (cd "$dir" && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$name.tex" >/dev/null) \
      || { echo "build: LaTeX error, see $dir/build/$name.log"; grep -m5 -A3 '^!' "$dir/build/$name.log"; return 1; }
  }

  pass || return 1
  if grep -q '^[^%]*\\bibliography{' "$dir/$name.tex"; then
    (cd "$dir" && BIBINPUTS="$dir:" bibtex "build/$name" >"build/$name.bibtex.out") \
      || { echo "build: BibTeX error, see $dir/build/$name.blg"; grep -m5 -i 'error\|warning' "$dir/build/$name.blg"; return 1; }
  fi
  pass && pass || return 1
  # report undefined citations/references and missing files, if any
  grep -m5 'LaTeX Warning: \(Citation\|Reference\|File\)' "$dir/build/$name.log"
  mv -f "$dir/build/$name.pdf" "$dir/$name.pdf" && echo "build: wrote $dir/$name.pdf"
}

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <path/to/file.tex> | --all"
  exit 1
fi

if [[ "$1" == "--all" ]]; then
  status=0
  for tex in "$(dirname "$0")"/*/main.tex; do
    build "$tex" || status=1
  done
  exit $status
fi
build "$1"
