# Coordinated-Base-Arm-Motion-Control
Capstone Project: Coordinated base–arm grasp planning for mobile manipulators, comparing classical whole-body control with Vision-Language-Action approaches on the Walkie platform.

## Setup

uv setup, if don't have uv, good luck

```bash
uv sync
```

## Repo layout

| Path | What it is |
|---|---|
| [run_ik_gui.py](run_ik_gui.py) | **Bimanual IK demo**: MuJoCo viewer + Tkinter target GUI, IK drives both arms (and optionally the base) |
| [benchmark/](benchmark/) | `run_benchmark.py`: numbers and plots comparing every IK version → `benchmark/results/` |
| [ik_sim/](ik_sim/) | `run_ik_sim.py`: any IK version on any test scenario, played back in MuJoCo → `ik_sim/results/` |
| [walker_robot/](walker_robot/) | The IK library (`IK/`, `bimanual_ik.py`), the IK versions + scenarios, sim helpers, `legacy/` |
| [simulation/](simulation/) | MuJoCo models (MJCF) and meshes |
| [report/](report/) | LaTeX reports (proposal, progress reports, final draft), see [Reports](#reports-latex) |

## Running (from the repo root)

```bash
uv run run_ik_gui.py                                 # bimanual IK with target GUI

uv run benchmark/run_benchmark.py                    # experiments A (singularity), B (rest weight), C (random targets)
uv run benchmark/run_benchmark.py --only B           # one experiment   (--help for options)

uv run ik_sim/run_ik_sim.py A1                       # scenario A1/A2/B in the viewer, its default IK versions in turn
uv run ik_sim/run_ik_sim.py B --methods rest_both current
uv run ik_sim/run_ik_sim.py B --record               # side-by-side GIF + keyframe strip for the report

uv run simulation/walker_launch.py                   # pure simulation with MuJoCo's manual control panel
uv run walker_robot/legacy/control_example.py        # older Walker sim/control framework example
```

IK versions (`--methods`): `pinv`, `dls`, `rest_left`, `rest_both`, `current`, `current_arm`
(see [walker_robot/ik_versions.py](walker_robot/ik_versions.py)).

How the IK works, how to use it in your own code and how to tune it: [walker_robot/README.md](walker_robot/README.md).

## Reports (LaTeX)

The reports are in `report/<name>/main.tex` and build with pdfLaTeX + BibTeX.

### Install the LaTeX toolchain

```bash
./report/setup_latex.sh            # installs TeX Live (asks for sudo), then checks every package
./report/setup_latex.sh --check    # only check an existing install
```

The script uses `apt` on Ubuntu/Debian; it also handles `dnf`, `pacman` and Homebrew (MacTeX). On Ubuntu it installs:

| apt package | Provides |
|---|---|
| `texlive-binaries` | `pdflatex`, `bibtex` |
| `texlive-base` | `amssymb` |
| `texlive-latex-base` | `amsmath`, `fontenc`, `inputenc`, `geometry`, `graphicx`, `hyperref` |
| `texlive-latex-recommended` | `booktabs`, `float`, `setspace`, `subcaption`, `xcolor` |
| `texlive-latex-extra` | `enumitem`, `titlesec` |
| `texlive-science` | `algorithm`, `algpseudocode` |
| `texlive-pictures` | `tikz` |
| `lmodern` | Latin Modern fonts |

Manual install on Ubuntu:

```bash
sudo apt-get install --no-install-recommends texlive-binaries texlive-base texlive-latex-base \
  texlive-latex-recommended texlive-latex-extra texlive-science texlive-pictures lmodern
```

### Build a report

```bash
./report/build.sh report/final_draft/main.tex   # one report -> report/final_draft/main.pdf
./report/build.sh --all                         # every report
```

`build.sh` runs pdfLaTeX, BibTeX (when the file uses `\bibliography`) and pdfLaTeX twice more, writes the PDF next to the `.tex`, and keeps the aux/log files in `report/<name>/build/` (git-ignored). It prints the first LaTeX error, and any undefined citation or missing file, if there is one.

Each report folder has the same layout:

```
report/<name>/
  main.tex
  citations.bib     # BibTeX references (the proposal has its references inline)
  figures/          # plots and screenshots
  images/           # logos and photos
  scripts/          # (final_draft) scripts that generate figures, e.g. plot_execution.py
```

`\graphicspath{{figures/}{images/}}` is set in every `main.tex`, so `\includegraphics{name.png}` needs no folder prefix.

