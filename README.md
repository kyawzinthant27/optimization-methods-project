# Phase 1: Code and Results

Read the summary files first. The raw records contain the original runs and timings.

| Report section | Folder under `revisions/` | Entry point |
| --- | --- | --- |
| 2–3 Image preparation and measured SSIM | `user-selected-v1` | `experiment.py` |
| 5–6 MILP and baseline GA | `part1-milp-ga-v1` | `experiment.py` |
| 5.3 MILP budget sensitivity | `part1-milp-budget-sensitivity-v1` | `run.py` |
| 5.3 Exact budget and browser stress test | `part1-4mbps-stress-v1` | `plan.py`, `run.mjs`, `summarize.py` |
| 6.4 GA mutation and elite sensitivity | `part1-ga-sensitivity-v1` | `run.py` |
| 6.4 GA generation limit | `part1-ga-generation-budget-v1` | `run.py` |
| 7 Browser benchmark | `part1-browser-benchmark-v1` | `run.mjs` |

The copies of the Python and JavaScript sources were formatted for reading. The recorded results and included configuration files are unchanged. The original project holds the exact source files used to produce the recorded runs, along with their integrity records and the WebP assets. Use that checkout and a new output directory to rerun the archived protocols. This submission folder is a code and results bundle, not a standalone replay of that archive.

The candidate manifest gives the measured file sizes and full-precision SSIM inputs. The benchmark used Playwright 1.64.1. Python package versions are listed in `requirements.lock.txt`.
