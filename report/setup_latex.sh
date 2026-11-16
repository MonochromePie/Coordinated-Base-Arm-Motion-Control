#!/usr/bin/env bash
# Install a local LaTeX toolchain (pdfLaTeX + BibTeX) with every package the reports use,
# then check that each package is found.
#
#   ./report/setup_latex.sh            # install (asks for sudo) and check
#   ./report/setup_latex.sh --check    # only check what is installed
set -euo pipefail

# Debian/Ubuntu packages -> LaTeX packages they provide (checked on Ubuntu 24.04)
APT_PACKAGES=(
  texlive-binaries            # pdflatex, bibtex
  texlive-base                # amssymb
  texlive-latex-base          # amsmath, fontenc, inputenc, geometry, graphicx, hyperref
  texlive-latex-recommended   # booktabs, float, setspace, subcaption, xcolor
  texlive-latex-extra         # enumitem, titlesec
  texlive-science             # algorithm, algpseudocode
  texlive-pictures            # tikz / pgf
  lmodern                     # Latin Modern fonts
)

# LaTeX packages used by report/*/main.tex
STY=(algorithm algpseudocode amsmath amssymb booktabs enumitem float fontenc geometry graphicx
     hyperref inputenc lmodern setspace subcaption tikz titlesec xcolor)

check() {
  local missing=0
  for bin in pdflatex bibtex kpsewhich; do
    if ! command -v "$bin" >/dev/null; then echo "missing program: $bin"; missing=1; fi
  done
  if command -v kpsewhich >/dev/null; then
    for p in "${STY[@]}"; do
      [[ -n "$(kpsewhich "$p.sty")" ]] || { echo "missing LaTeX package: $p"; missing=1; }
    done
  fi
  if (( missing )); then echo "LaTeX setup incomplete"; return 1; fi
  echo "LaTeX setup OK: $(pdflatex --version | head -1)"
}

if [[ "${1:-}" == "--check" ]]; then
  check
  exit $?
fi

if command -v apt-get >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
elif command -v dnf >/dev/null; then
  sudo dnf install -y texlive-scheme-basic texlive-collection-latexrecommended \
    texlive-collection-latexextra texlive-collection-pictures texlive-collection-mathscience \
    texlive-lm texlive-bibtex
elif command -v pacman >/dev/null; then
  sudo pacman -S --needed texlive-basic texlive-latex texlive-latexrecommended texlive-latexextra \
    texlive-pictures texlive-mathscience texlive-fontsrecommended texlive-binextra
elif command -v brew >/dev/null; then
  brew install --cask mactex-no-gui
  echo "Open a new terminal so /Library/TeX/texbin is on PATH, then rerun with --check."
  exit 0
else
  echo "No supported package manager found. Install TeX Live (https://tug.org/texlive/) with"
  echo "the packages: ${STY[*]}"
  exit 1
fi

check
