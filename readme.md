# Radiosensitivity biomarkers: RSI → GARD → RxRSI

A self-contained tutorial that computes three genome-based radiosensitivity biomarkers from public bulk RNA-seq data, then uses them for survival modelling and personalized dose estimation:

- **RSI** (Radiation Sensitivity Index) — a 10-gene expression score of intrinsic tumor radiosensitivity (low RSI = radiosensitive, high RSI = radioresistant).
- **GARD** (Genomically Adjusted Radiation Dose) — RSI combined with the delivered fractionation through the linear-quadratic model; the biological *effect* a given physical dose achieves.
- **RxRSI** — the inverse: the physical dose a patient would need to reach a target GARD.

## Dataset

We use **GSE190826**, pre-treatment rectal cancer biopsies treated with neoadjuvant chemoradiotherapy (**50.4 Gy / 28 fractions = 1.8 Gy/fraction**), with disease-free survival (DFS) and pathological-response (pCR) data.

- GEO record: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190826>
- Raw counts (downloaded automatically by `tutorial.py`): <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE190nnn/GSE190826/suppl/GSE190826_RAW.tar>
- Source publication (Cancer Cell, 2022): <https://www.cell.com/cancer-cell/fulltext/S1535-6108(22)00006-X>

After filtering to pre-CRT biopsies the cohort is **n = 105** (86 with DFS data; 26 pCR / 79 non-pCR).

## Background papers

| Biomarker / topic | Reference | DOI |
|---|---|---|
| Original GARD model (RSI → α → GARD) | Scott et al., *Lancet Oncol.* 2017 | [10.1016/S1470-2045(16)30648-9](https://doi.org/10.1016/S1470-2045(16)30648-9) |
| Pan-cancer validation of GARD | Scott et al., *Lancet Oncol.* 2021 | [10.1016/S1470-2045(21)00347-8](https://doi.org/10.1016/S1470-2045(21)00347-8) |
| RxRSI / personalized dose (NSCLC) | Scott et al., *J. Thorac. Oncol.* 2021 | [10.1016/j.jtho.2020.11.008](https://doi.org/10.1016/j.jtho.2020.11.008) |
| Maximally selected rank statistics | Lausen & Schumacher, *Biometrics* 1992 | [10.2307/2532740](https://doi.org/10.2307/2532740) |
| GSE190826 source (rectal cancer) | Cancer Cell, 2022 | [10.1016/j.ccell.2022.01.004](https://doi.org/10.1016/j.ccell.2022.01.004) |

## Setup

Python 3.10. With [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.10
uv pip install -r requirements.txt
```

(or `uv sync` to use `pyproject.toml` + `uv.lock`).

## Running

```bash
uv run python tutorial.py
```

On first run it downloads `GSE190826_RAW.tar` (once), parses per-sample `featureCounts`, and caches `data/gse190826_counts.csv` + `data/gse190826_meta.csv` so subsequent runs are fast. Outputs:

- `data/results.csv` — per-patient RSI, GARD, RxRSI, group labels, and survival columns.
- `data/gse190826_processed.h5ad` — the full processed `AnnData`.
- `figures/*.png` — all plots (RSI/GARD distributions, maxstat scan, Cox forest plots, Kaplan–Meier curves, RxRSI waterfall/spectrum, etc.).

## Repository layout

```text
radiosensitivity-biomarker/
├── tutorial.py                      # full RSI -> GARD -> RxRSI pipeline
├── requirements.txt                 # dependencies (Python 3.10)
├── data/                            # downloaded + cached (created on first run)
│   └── references/                  # the three Scott et al. papers (PDFs)
└── figures/                         # output plots
```
