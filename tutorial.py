"""
Radiosensitivity Biomarker Tutorial
RSI -> GARD -> RxRSI on GSE190826 (rectal cancer, 50.4 Gy / 28 fx = 1.8 Gy/fx)

Dataset: Fontenot et al. / Isella et al. -- 105 pre-treatment rectal cancer biopsies
Accession: GSE190826
Clinical: DFS time (months) + DFS event + pCR status deposited in GEO metadata

https://www.cell.com/cancer-cell/fulltext/S1535-6108(22)00006-X?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS153561082200006X%3Fshowall%3Dtrue


"""
# %%
import gzip
import urllib.request
import warnings
from pathlib import Path

import GEOparse
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from rnanorm import CPM
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR  = BASE_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.figsize":       (8, 5),
    "figure.dpi":           150,
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "axes.edgecolor":       "black",
    "axes.labelcolor":      "black",
    "xtick.color":          "black",
    "ytick.color":          "black",
    "text.color":           "black",
    "legend.facecolor":     "white",
    "legend.edgecolor":     "black",
    "legend.labelcolor":    "black",
    "savefig.facecolor":    "white",
    "savefig.edgecolor":    "white",
})

# ── constants that we can change lter ──────────────────────────────────────────────────────────────────
GEO_ID     = "GSE190826"
RAW_TAR_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE190nnn/GSE190826/suppl/GSE190826_RAW.tar"

N_FRACTIONS  = 28       # 50.4 Gy / 28 fx = 1.8 Gy per fraction
TOTAL_DOSE   = 50.4     # Gy
DOSE_PER_FX  = TOTAL_DOSE / N_FRACTIONS  # = 1.8 Gy

BETA    = 0.05          # fixed LQ beta (Gy^-2), Scott 2017
D_ALPHA = 2.0           # SF2-assay reference dose for alpha derivation (Gy)

# Even though this study only has a 1 arm delivered dose, we will establish a upper and lower interval just to see how we could evaluate RxRSI in a multi-arm study, if that was the case.
SOC_LOW  = 45.0         # Gy  -- lower edge of rectal neoadjuvant SOC
SOC_HIGH = 54.0         # Gy  -- upper edge
SOC_REF  = 50.4         # Gy  -- delivered to all patients in this cohort

# RSI 10-gene coefficients -- Check out the paper from Scott et al. 2017
RSI_COEFFS = {
    "AR":    -0.0098009,
    "JUN":    0.0128283,
    "STAT1":  0.0254552,
    "PRKCB": -0.0017589,
    "RELA":  -0.0038171,
    "ABL1":   0.1070213,
    "SUMO1": -0.0002509,
    "PAK2":  -0.0092431,
    "HDAC1": -0.0204469,
    "IRF1":  -0.0441683,
}
RSI_GENES = list(RSI_COEFFS.keys())

# GRCh38 Ensembl IDs for the 10 RSI genes (featureCounts files use Ensembl IDs). Since its not many genes, we can use something like https://www.syngoportal.org/convert to conver from Ensembl to HGNC symbols
RSI_ENSEMBL = {
    "AR":    "ENSG00000169083",
    "JUN":   "ENSG00000177606",
    "STAT1": "ENSG00000115415",
    "PRKCB": "ENSG00000166501",
    "RELA":  "ENSG00000173039",
    "ABL1":  "ENSG00000097007",
    "SUMO1": "ENSG00000116030",
    "PAK2":  "ENSG00000180370",
    "HDAC1": "ENSG00000116478",
    "IRF1":  "ENSG00000125347",
}
# Reverse map: Ensembl -> symbol
ENSEMBL_TO_SYMBOL = {v: k for k, v in RSI_ENSEMBL.items()}

# RxRSI group colours (Scott 2021)
RXRSI_COLORS = {
    "Radiosensitive": "#2ca02c",
    "Intermediate":   "#E6A817",
    "Radioresistant": "#d62728",
}
RXRSI_ORDER = ["Radiosensitive", "Intermediate", "Radioresistant"]

# %%
# ── 1. Getting the dataset GSE190826 ─────────────────────────────────────────────────────────
print("=== 1. Fetching GSE190826 ===")

counts_cache = DATA_DIR / "gse190826_counts.csv"
meta_cache   = DATA_DIR / "gse190826_meta.csv"

if counts_cache.exists() and meta_cache.exists():
    print("  Loading from cache ...")
    counts_df = pd.read_csv(counts_cache, index_col=0)
    meta_df   = pd.read_csv(meta_cache,   index_col=0)
else:
    # Download the RAW tar (per-sample raw counts .gz files)
    tar_path = DATA_DIR / "GSE190826_RAW.tar"
    if not tar_path.exists():
        print(f"  Downloading {RAW_TAR_URL} ...")
        urllib.request.urlretrieve(RAW_TAR_URL, tar_path)
        print(f"  Saved: {tar_path.name}  ({tar_path.stat().st_size // 1024 // 1024} MB)")

    import tarfile
    print("  Parsing count files from tar ...")
    count_cols = {}

    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".gz")]
        for m in members:
            gsm_id = m.name.split("_")[0]
            f = tar.extractfile(m)
            if f is None:
                continue
            with gzip.open(f) as gz:
                # featureCounts format: Geneid Chr Start End Strand Length <count>
                df = pd.read_csv(gz, sep="\t", comment="#")
            gene_col  = df.columns[0]   # "Geneid"
            count_col = df.columns[-1]  # sample-specific count column
            df[gene_col] = df[gene_col].str.split(".").str[0]  # strip Ensembl version
            count_cols[gsm_id] = df.set_index(gene_col)[count_col]

    counts_df = pd.DataFrame(count_cols).T
    counts_df.index.name = "sample_id"
    # Columns are Ensembl IDs; gotta rename the 10 RSI genes to HGNC symbols for convenience
    counts_df.rename(columns=ENSEMBL_TO_SYMBOL, inplace=True)

    # GEO metadata: DFS time/event + pCR + treatment
    print("  Loading GEO metadata ...")
    gse = GEOparse.get_GEO(geo=GEO_ID, destdir=str(DATA_DIR), silent=True)
    meta_rows = []
    for gsm_name, gsm in gse.gsms.items():
        row = {"sample_id": gsm_name}
        for ch in gsm.metadata.get("characteristics_ch1", []):
            if ":" in ch:
                k, v = ch.split(":", 1)
                row[k.strip().lower().replace(" ", "_")] = v.strip()
        meta_rows.append(row)
    meta_df = pd.DataFrame(meta_rows).set_index("sample_id")

    counts_df.to_csv(counts_cache)
    meta_df.to_csv(meta_cache)
    print(f"  Saved: {counts_cache.name}, {meta_cache.name}")

print(f"  Samples: {counts_df.shape[0]}   Genes: {counts_df.shape[1]}")
print(f"  Metadata columns: {list(meta_df.columns)}")


# %% 2. Filter to pre-CRT
# ── 2. Keep only pre-CRT biopsies ─────────────────────────────────────────────
print("\n=== 2. Filtering to pre-CRT samples ===")

# The dataset has both pre-CRT and post-CRT samples; RSI is computed on baseline (ITT - intent to treat samples)
pre_mask = meta_df["time"].str.strip().str.lower() == "pre-crt"
meta_df  = meta_df[pre_mask].copy()
shared   = counts_df.index.intersection(meta_df.index)
counts_df = counts_df.loc[shared].copy()
meta_df   = meta_df.loc[shared].copy()

print(f"  Pre-CRT samples: {len(meta_df)}")


# %% 3. Build AnnData
# ── 3. Build AnnData ───────────────────────────────────────────────────────────
print("\n=== 3. Building AnnData ===")

counts_df = counts_df.apply(pd.to_numeric, errors="coerce").fillna(0)
gene_sums = counts_df.sum(axis=0)
counts_df = counts_df.loc[:, gene_sums > 0]

adata = ad.AnnData(counts_df)
adata.obs = meta_df.copy()

# Rename clinical columns to standard names used throughout
adata.obs.rename(columns={
    "disease_free_survival_time_(months)": "dfs_months",
    "disease_free_survival_event":          "dfs_event",
    "treatment_response_pcr_vs._non-pcr":   "response",
}, inplace=True)

# Coerce survival columns to numeric (some rows have "n.a.")
adata.obs["dfs_months"] = pd.to_numeric(adata.obs["dfs_months"], errors="coerce")
adata.obs["dfs_event"]  = pd.to_numeric(adata.obs["dfs_event"],  errors="coerce")

# Protocol-uniform dose for all patients
adata.obs["total_dose"]  = TOTAL_DOSE
adata.obs["n_fractions"] = N_FRACTIONS
adata.obs["dose_per_fx"] = DOSE_PER_FX

adata.layers["raw_counts"] = adata.X.copy()

print(f"  AnnData: {adata.n_obs} samples x {adata.n_vars} genes")
print(f"  DFS: {adata.obs['dfs_months'].notna().sum()} samples have survival data")
print(f"  Response: {adata.obs['response'].value_counts().to_dict()}")

TIME_COL  = "dfs_months"
EVENT_COL = "dfs_event"


# %% 4. Normalize (CPM)
# ── 4. Normalize: raw counts -> CPM ───────────────────────────────────────────
print("\n=== 4. Normalizing (CPM) ===")

raw = adata.layers["raw_counts"].copy()
cpm = CPM().fit_transform(raw)
adata.layers["cpm"]     = cpm.copy()
adata.layers["log_cpm"] = np.log1p(cpm)

print("  Layers: raw_counts | cpm | log_cpm")


# %% 5. Compute RSI
# ── 5. Compute RSI ─────────────────────────────────────────────────────────────
print("\n=== 5. Computing RSI ===")

# Steps (Scott 2017 + reference implementation):
#   1. Order all genes by log-CPM within each sample (just to get a within-sample
#      ordering; this intermediate position is NOT the rank used in the formula).
#   2. Extract the 10 RSI genes and re-rank them AMONG THEMSELVES into relative
#      ranks 1..10, where the HIGHEST-expressed RSI gene = 10 and the lowest = 1
#      (matching Scott 2017: highest expression -> rank 10).
#   3. RSI = weighted sum of those relative ranks, clipped to [0, 1].

missing = [g for g in RSI_GENES if g not in adata.var_names]
present = [g for g in RSI_GENES if g in adata.var_names]
if missing:
    print(f"  WARNING: RSI genes absent from data: {missing}")

log_cpm   = adata.layers["log_cpm"]
var_names = np.array(adata.var_names)

rsi_scores = []
for i in range(adata.n_obs):
    sample_expr = log_cpm[i, :]

    # Intermediate within-sample ordering position. We assign position 1 to the
    # highest-expressed gene purely as a sorting key; this is NOT the rank that
    # enters RSI (that is computed in the re-ranking step below).
    global_order    = np.argsort(sample_expr)[::-1]
    global_rank_pos = np.empty(len(sample_expr), dtype=float)
    global_rank_pos[global_order] = np.arange(1, len(sample_expr) + 1)

    # Ordering position of each of the 10 RSI genes
    gene_positions = {g: global_rank_pos[np.where(var_names == g)[0][0]] for g in present}

    # Re-rank the RSI genes AMONG THEMSELVES (the relative rank from scott 2017). sorted_genes is ordered from highest
    # to lowest expression (smallest ordering position first), so the highest-
    # expressed gene (r=0) gets relative rank n_present (=10), and the lowest
    # (r=n_present-1) gets relative rank 1 -- i.e. highest expression -> rank 10
    sorted_genes   = sorted(gene_positions, key=lambda g: gene_positions[g])
    n_present      = len(sorted_genes)
    relative_ranks = {g: n_present - r for r, g in enumerate(sorted_genes)}

    rsi = sum(RSI_COEFFS[g] * relative_ranks[g] for g in present)
    rsi_scores.append(float(np.clip(rsi, 0, 1)))

adata.obs["RSI"] = rsi_scores
print(f"  RSI range: [{min(rsi_scores):.4f}, {max(rsi_scores):.4f}]  "
      f"median: {np.median(rsi_scores):.4f}")


# %% 6. RSI distribution plots
# ── 6. RSI distribution plots ─────────────────────────────────────────────────
print("\n=== 6. RSI distribution plots ===")

# Histogram
fig, ax = plt.subplots()
ax.hist(adata.obs["RSI"], bins=20, color="#4878d0", edgecolor="white", linewidth=0.5)
ax.set_xlabel("RSI")
ax.set_ylabel("Number of patients")
ax.set_title(f"RSI distribution -- GSE190826 rectal cancer (n={adata.n_obs})")
plt.tight_layout()
fig.savefig(FIG_DIR / "rsi_histogram.png")
plt.close()
print("  Saved: rsi_histogram.png")

# Boxplot by pathological response (pCR vs non-pCR)
fig, ax = plt.subplots()
groups = sorted(adata.obs["response"].dropna().unique())
plot_data = [adata.obs.loc[adata.obs["response"] == g, "RSI"].dropna().values for g in groups]
bp = ax.boxplot(plot_data, labels=groups, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#d0e4f7")
for med in bp["medians"]:
    med.set_color("black")
ax.set_xlabel("Response")
ax.set_ylabel("RSI")
ax.set_title("RSI by pathological response (pCR vs non-pCR)")
plt.tight_layout()
fig.savefig(FIG_DIR / "rsi_boxplot_by_response.png")
plt.close()
print("  Saved: rsi_boxplot_by_response.png")


# %% 7. Compute GARD
# ── 7. Compute GARD ───────────────────────────────────────────────────────────
print("\n=== 7. Computing GARD ===")

# alpha = (ln(RSI) + beta * d_alpha^2) / (-d_alpha)   [Scott 2017]
# GARD  = n * d * (alpha + beta * d)
rsi_arr = np.array(rsi_scores)
rsi_arr = np.where(rsi_arr <= 0, 1e-4, rsi_arr)

alpha = (np.log(rsi_arr) + BETA * D_ALPHA**2) / (-D_ALPHA)
gard  = N_FRACTIONS * DOSE_PER_FX * (alpha + BETA * DOSE_PER_FX)
gard  = np.clip(gard, 1, 100) # lets clip the values, although we can use them unclipped.

adata.obs["GARD"]  = gard
adata.obs["alpha"] = alpha
print(f"  GARD range: [{gard.min():.2f}, {gard.max():.2f}]  median: {np.median(gard):.2f}")


# %% 8. Dichotomize GARD (median + maxstat)
# ── 8. Dichotomize GARD: median split + maxstat ───────────────────────────────
print("\n=== 8. Dichotomizing GARD ===")

# Median split
gard_median = float(np.median(gard))
adata.obs["GARD_cat_median"] = np.where(gard >= gard_median, "High", "Low")
print(f"  Median cutpoint: {gard_median:.2f}  "
      f"groups: {adata.obs['GARD_cat_median'].value_counts().to_dict()}")


def maxstat_2g(gard_vals, time_vals, event_vals, pct_window=(33, 75)):
    """Scan log-rank chi2 over candidate GARD cutpoints and return the one that
    maximizes it, with BH-FDR correction across the scan. We restrict the search to
    the central `pct_window` percentiles of GARD so that neither group becomes a tiny
    extreme -- splitting near the tails inflates the statistic by chance (Lausen &
    Schumacher 1992)."""
    lo, hi = np.percentile(gard_vals, pct_window)
    candidates = np.unique(gard_vals[(gard_vals >= lo) & (gard_vals <= hi)])
    chi2_vals, pvals = [], []
    for cut in candidates:
        g_hi = gard_vals >= cut
        r = logrank_test(time_vals[g_hi], time_vals[~g_hi],
                         event_observed_A=event_vals[g_hi],
                         event_observed_B=event_vals[~g_hi])
        chi2_vals.append(r.test_statistic)
        pvals.append(r.p_value)
    chi2_arr = np.array(chi2_vals)
    _, bh_pvals, _, _ = multipletests(pvals, method="fdr_bh")
    best_idx = int(np.argmax(chi2_arr))
    return candidates[best_idx], chi2_arr, candidates, bh_pvals[best_idx]


obs_s = adata.obs[[TIME_COL, EVENT_COL, "GARD"]].dropna()
obs_s = obs_s[obs_s[TIME_COL] > 0]
maxstat_cut, chi2_arr, cut_arr, maxstat_bh_p = maxstat_2g(
    obs_s["GARD"].values, obs_s[TIME_COL].values, obs_s[EVENT_COL].values)
maxstat_pct = float(np.mean(gard < maxstat_cut) * 100)
print(f"  Maxstat cutpoint: {maxstat_cut:.2f} Gy "
      f"(~{maxstat_pct:.0f}th percentile of GARD)  BH-p: {maxstat_bh_p:.4f}")

fig, ax = plt.subplots()
ax.plot(cut_arr, chi2_arr, color="#4878d0")
ax.axvline(maxstat_cut, color="red", linestyle="--",
           label=f"Optimal cut = {maxstat_cut:.1f} (~{maxstat_pct:.0f}th pct)")
ax.set_xlabel("GARD cutpoint")
ax.set_ylabel("Log-rank chi2")
ax.set_title("Maxstat scan -- GARD vs DFS")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "maxstat_scan.png")
plt.close()

adata.obs["GARD_cat_maxstat"] = np.where(gard >= maxstat_cut, "High", "Low")
print(f"  Maxstat groups: {adata.obs['GARD_cat_maxstat'].value_counts().to_dict()}")

# GARD histogram
fig, ax = plt.subplots()
ax.hist(gard, bins=20, color="#ee854a", edgecolor="white", linewidth=0.5)
ax.axvline(gard_median, color="black", linestyle="--", label=f"Median ({gard_median:.1f})")
if not np.isnan(maxstat_cut) and abs(maxstat_cut - gard_median) > 0.5:
    ax.axvline(maxstat_cut, color="red", linestyle=":", label=f"Maxstat ({maxstat_cut:.1f})")
ax.set_xlabel("GARD")
ax.set_ylabel("Number of patients")
ax.set_title("GARD distribution")
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "gard_histogram.png")
plt.close()
print("  Saved: gard_histogram.png")

# GARD boxplot by response
fig, ax = plt.subplots()
groups = sorted(adata.obs["response"].dropna().unique())
plot_data = [adata.obs.loc[adata.obs["response"] == g, "GARD"].dropna().values for g in groups]
bp = ax.boxplot(plot_data, labels=groups, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#fde8d0")
for med in bp["medians"]:
    med.set_color("black")
ax.set_xlabel("Response")
ax.set_ylabel("GARD")
ax.set_title("GARD by pathological response")
plt.tight_layout()
fig.savefig(FIG_DIR / "gard_boxplot_by_response.png")
plt.close()
print("  Saved: gard_boxplot_by_response.png")

# GARD vs RSI scatter
fig, ax = plt.subplots()
for grp, color in [("High", "#2ca02c"), ("Low", "#d62728")]:
    mask = adata.obs["GARD_cat_median"] == grp
    ax.scatter(adata.obs.loc[mask, "RSI"], adata.obs.loc[mask, "GARD"],
               label=grp, alpha=0.7, color=color, s=40)
ax.set_xlabel("RSI")
ax.set_ylabel("GARD")
ax.set_title("GARD vs RSI (coloured by median group)")
ax.legend(title="GARD group")
plt.tight_layout()
fig.savefig(FIG_DIR / "gard_vs_rsi_scatter.png")
plt.close()
print("  Saved: gard_vs_rsi_scatter.png")


# %% 9. Cox modelling
# ── 9. Cox modelling ───────────────────────────────────────────────────────────
print("\n=== 9. Cox modelling ===")

# Build the modelling table: z-score the continuous predictors, binarize the
# categorical GARD splits, and encode pCR as the one clinical covariate we have.
obs_cox = adata.obs[adata.obs[TIME_COL] > 0].dropna(subset=[TIME_COL, EVENT_COL]).copy()
obs_cox["RSI_z"]                = zscore(obs_cox["RSI"],  nan_policy="omit")
obs_cox["GARD_z"]               = zscore(obs_cox["GARD"], nan_policy="omit")
obs_cox["GARD_cat_median_bin"]  = (obs_cox["GARD_cat_median"]  == "High").astype(int)
obs_cox["GARD_cat_maxstat_bin"] = (obs_cox["GARD_cat_maxstat"] == "High").astype(int)
obs_cox["pcr_bin"]              = (obs_cox["response"].str.strip().str.lower() == "pcr").astype(int)


def fit_cox(cols, label):
    """Fit a Cox model on `cols` (first column is the predictor of interest) and
    return its hazard ratio, 95% CI, p-value, and concordance."""
    sub = obs_cox[cols + [TIME_COL, EVENT_COL]]
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(sub, duration_col=TIME_COL, event_col=EVENT_COL)
    row = cph.summary.loc[cols[0]]
    return {
        "model": label, "predictor": cols[0],
        "HR":       round(float(np.exp(row["coef"])), 3),
        "HR_lower": round(float(np.exp(row["coef lower 95%"])), 3),
        "HR_upper": round(float(np.exp(row["coef upper 95%"])), 3),
        "p":        round(float(row["p"]), 4),
        "c_index":  round(float(cph.concordance_index_), 3),
    }


predictors = ["RSI_z", "GARD_z", "GARD_cat_median_bin", "GARD_cat_maxstat_bin"]
cox_results = []
for pred in predictors:                              # univariate
    cox_results.append(fit_cox([pred], f"Univariate: {pred}"))
for pred in predictors:                              # adjusted for pCR
    cox_results.append(fit_cox([pred, "pcr_bin"], f"Multivariate: {pred}"))

cox_df = pd.DataFrame(cox_results)
cox_df.to_csv(DATA_DIR / "cox_results.csv", index=False)
for r in cox_results:
    print(f"  {r['model']}: HR={r['HR']} [{r['HR_lower']}-{r['HR_upper']}]  p={r['p']}  C={r['c_index']}")


def forest_plot(df, title, outpath, color):
    fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.8)))
    y = range(len(df))
    ax.scatter(df["HR"], y, color=color, zorder=3, s=60)
    ax.hlines(y, df["HR_lower"], df["HR_upper"], color=color, linewidth=2)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(list(y)); ax.set_yticklabels(df["predictor"].tolist())
    ax.set_xlabel("Hazard Ratio (95% CI)"); ax.set_title(title)
    plt.tight_layout(); fig.savefig(outpath); plt.close()

forest_plot(cox_df[cox_df["model"].str.startswith("Univariate")],
            "Univariate Cox -- DFS", FIG_DIR / "forest_univariate.png", "#4878d0")
forest_plot(cox_df[cox_df["model"].str.startswith("Multivariate")],
            "Multivariate Cox -- DFS", FIG_DIR / "forest_multivariate.png", "#ee854a")
print("  Saved: cox_results.csv, forest_univariate.png, forest_multivariate.png")


# %% 10. Kaplan-Meier plots
# ── 10. Kaplan-Meier plots ────────────────────────────────────────────────────
print("\n=== 10. Kaplan-Meier plots ===")

def plot_km_2group(group_col, title, outpath, colors):
    """Plot Kaplan-Meier DFS curves for the two groups in `group_col`, annotated
    with the log-rank p-value between them."""
    df = adata.obs[[group_col, TIME_COL, EVENT_COL]].dropna()
    df = df[df[TIME_COL] > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    grp_a, grp_b = sorted(df[group_col].unique(), reverse=True)  # High/pCR first
    for grp in (grp_a, grp_b):
        mask = df[group_col] == grp
        kmf.fit(df.loc[mask, TIME_COL], df.loc[mask, EVENT_COL], label=grp)
        kmf.plot_survival_function(ax=ax, color=colors[grp], ci_show=False)

    lr = logrank_test(
        df.loc[df[group_col] == grp_a, TIME_COL], df.loc[df[group_col] == grp_b, TIME_COL],
        event_observed_A=df.loc[df[group_col] == grp_a, EVENT_COL],
        event_observed_B=df.loc[df[group_col] == grp_b, EVENT_COL])
    ax.text(0.05, 0.05, f"log-rank p = {lr.p_value:.4f}", transform=ax.transAxes, fontsize=10)

    ax.set_xlabel("Time (months)"); ax.set_ylabel("DFS probability")
    ax.set_title(title); ax.set_ylim(0, 1.05)
    plt.tight_layout(); fig.savefig(outpath); plt.close()


GARD_COLORS = {"High": "#2ca02c", "Low": "#d62728"}
plot_km_2group("GARD_cat_median", f"DFS -- GARD median split (cut={gard_median:.1f})",
               FIG_DIR / "km_dfs_gard_median.png", GARD_COLORS)
plot_km_2group("GARD_cat_maxstat", f"DFS -- GARD maxstat split (cut={maxstat_cut:.1f})",
               FIG_DIR / "km_dfs_gard_maxstat.png", GARD_COLORS)
plot_km_2group("response", "DFS -- pCR vs non-pCR",
               FIG_DIR / "km_dfs_response.png", {"pCR": "#2ca02c", "Non-pCR": "#d62728"})
print("  Saved: km_dfs_gard_median.png, km_dfs_gard_maxstat.png, km_dfs_response.png")


# %% 11. Compute RxRSI
# ── 11. RxRSI ─────────────────────────────────────────────────────────────────
print("\n=== 11. Computing RxRSI ===")

# GARD_T = maxstat cutpoint (or median fallback).
# RxRSI = the physical dose needed to reach the target GARD_T for each patient.
#   RxRSI = GARD_T / (alpha + beta * d)   [Scott 2021, eq 4]
# alpha is the patient-specific linear LQ parameter derived from RSI (computed in
# step 7); d is the dose per fraction. Lower alpha (radioresistant tumour) -> larger
# required dose. We guard against alpha + beta*d <= 0 (would give a non-physical dose).
GARD_T = maxstat_cut if not np.isnan(maxstat_cut) else gard_median
print(f"  Target GARD (GARD_T): {GARD_T:.2f}")

denom = alpha + BETA * DOSE_PER_FX
rxrsi = np.where(denom > 0, GARD_T / denom, np.inf)
rxrsi = np.where(np.isnan(rxrsi) | (rxrsi <= 0), np.inf, rxrsi)

adata.obs["RxRSI"] = rxrsi
finite = np.isfinite(rxrsi)
print(f"  RxRSI (finite): min={rxrsi[finite].min():.1f}  "
      f"max={rxrsi[finite].max():.1f}  median={np.median(rxrsi[finite]):.1f} Gy")


# %% 12. RxRSI group assignment
# ── 12. RxRSI group assignment ────────────────────────────────────────────────
print("\n=== 12. RxRSI dose classification ===")

rxrsi_group = np.where(
    rxrsi <= SOC_LOW, "Radiosensitive",
    np.where(rxrsi <= SOC_HIGH, "Intermediate", "Radioresistant"),
)
adata.obs["RxRSI_group"] = rxrsi_group
print(f"  {pd.Series(rxrsi_group).value_counts().to_dict()}")

# Scott 2021 +-10% delivered-dose match
delivered = adata.obs["total_dose"].values
tol = 0.10
dose_match = np.where(
    rxrsi > delivered * (1 + tol), "Under-dosed",
    np.where(rxrsi < delivered * (1 - tol), "Over-dosed", "Adequate"),
)
adata.obs["dose_match"] = dose_match
print(f"  Dose-match vs {TOTAL_DOSE} Gy: {pd.Series(dose_match).value_counts().to_dict()}")


# %% 13. RxRSI plots
# ── 13. RxRSI plots ───────────────────────────────────────────────────────────
print("\n=== 13. RxRSI plots ===")

obs_rx = adata.obs[np.isfinite(adata.obs["RxRSI"])].copy()
obs_rx = obs_rx.sort_values("RxRSI").reset_index(drop=True)

Y_MAX = 170  # cap; extreme outliers clamped

# 13a. Dose vs RSI scatter with analytic curves
fig, ax = plt.subplots(figsize=(10, 7))

rsi_grid   = np.linspace(0.01, 0.99, 300)
alpha_grid = (np.log(rsi_grid) + BETA * D_ALPHA**2) / (-D_ALPHA)
# RxRSI = GARD_T / (alpha + beta*d); the curve shape depends on the dose per
# fraction d, so we draw it for a few representative fractionations.
for d_fx, ls in [(1.5, "--"), (1.8, "-"), (2.0, ":")]:
    denom_c = alpha_grid + BETA * d_fx
    rx_c    = np.where(denom_c > 0, GARD_T / denom_c, np.nan)
    ax.plot(rsi_grid, np.clip(rx_c, 0, Y_MAX), color="#1f77b4",
            linestyle=ls, linewidth=2, label=f"RxRSI @ {d_fx} Gy/fx", zorder=1)

for grp in RXRSI_ORDER:
    sub = obs_rx[obs_rx["RxRSI_group"] == grp]
    if sub.empty:
        continue
    y_vals = sub["RxRSI"].values
    x_vals = sub["RSI"].values
    off    = y_vals > Y_MAX
    y_plot = np.where(off, Y_MAX, y_vals)
    if (~off).any():
        ax.scatter(x_vals[~off], y_plot[~off], c=RXRSI_COLORS[grp],
                   edgecolor="black", linewidth=0.5, s=70, marker="o", zorder=3)
    if off.any():
        ax.scatter(x_vals[off], y_plot[off], c=RXRSI_COLORS[grp],
                   edgecolor="black", linewidth=0.5, s=95, marker="v", zorder=3)

soc_low_pad, soc_high_pad = SOC_LOW * 0.9, SOC_HIGH * 1.1
ax.axhspan(soc_low_pad, SOC_LOW, color="grey", alpha=0.06, zorder=0)
ax.axhspan(SOC_HIGH, soc_high_pad, color="grey", alpha=0.06, zorder=0)
ax.axhline(SOC_LOW,  color="grey", linestyle="--", linewidth=1.5, zorder=2)
ax.axhline(SOC_HIGH, color="grey", linestyle="--", linewidth=1.5, zorder=2)
ax.axhline(soc_low_pad,  color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
ax.axhline(soc_high_pad, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
ax.text(0.99, SOC_LOW,  f" SOC low ({SOC_LOW} Gy)",  va="bottom", ha="right",
        fontsize=11, color="grey", transform=ax.get_yaxis_transform())
ax.text(0.99, SOC_HIGH, f" SOC high ({SOC_HIGH} Gy)", va="bottom", ha="right",
        fontsize=11, color="grey", transform=ax.get_yaxis_transform())

group_handles = [
    plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor=RXRSI_COLORS[g], markeredgecolor="black", label=g)
    for g in RXRSI_ORDER
]
ax.legend(handles=group_handles, title="Group", loc="upper left", frameon=False)
ax.set_xlabel("RSI")
ax.set_ylabel("Required dose (Gy)")
ax.set_title(f"Dose vs RSI -- target GARD = {GARD_T:.1f}")
ax.set_xlim(0, 1)
ax.set_ylim(0, Y_MAX)
n_clamped = int((obs_rx["RxRSI"] > Y_MAX).sum())
if n_clamped:
    fig.text(0.5, -0.02,
             f"{n_clamped} patient(s) with RxRSI > {Y_MAX} Gy clamped to top edge",
             ha="center", fontsize=10, color="dimgrey")
plt.tight_layout()
fig.savefig(FIG_DIR / "dose_vs_rsi_curves.png", bbox_inches="tight")
plt.close()
print("  Saved: dose_vs_rsi_curves.png")

# 13b. Waterfall (bars coloured by group, SOC band shaded)
fig, ax = plt.subplots(figsize=(11, 6))
heights   = np.where(obs_rx["RxRSI"] > Y_MAX, Y_MAX, obs_rx["RxRSI"].values)
off_mask  = obs_rx["RxRSI"].values > Y_MAX
bar_colors = [RXRSI_COLORS[g] for g in obs_rx["RxRSI_group"]]
ax.bar(np.arange(len(obs_rx)), heights, color=bar_colors, edgecolor="black", linewidth=0.3)
if off_mask.any():
    ax.scatter(np.where(off_mask)[0], np.full(off_mask.sum(), Y_MAX * 0.97),
               marker="^", c="black", s=25, zorder=4)
ax.axhspan(SOC_LOW * 0.9, SOC_LOW,   color="grey", alpha=0.08, zorder=0)
ax.axhspan(SOC_HIGH, SOC_HIGH * 1.1, color="grey", alpha=0.08, zorder=0)
ax.axhspan(SOC_LOW, SOC_HIGH,        color="grey", alpha=0.20, zorder=0)
ax.axhline(SOC_LOW,  color="grey", linestyle="--", linewidth=1.2)
ax.axhline(SOC_HIGH, color="grey", linestyle="--", linewidth=1.2)
ax.axhline(SOC_LOW * 0.9,  color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
ax.axhline(SOC_HIGH * 1.1, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
grp_counts = obs_rx["RxRSI_group"].value_counts()
in_soc = int(((obs_rx["RxRSI"] >= SOC_LOW) & (obs_rx["RxRSI"] <= SOC_HIGH)).sum())
in_ext = int(((obs_rx["RxRSI"] >= SOC_LOW * 0.9) & (obs_rx["RxRSI"] <= SOC_HIGH * 1.1)).sum())
legend_labels = {
    "Radiosensitive": f"Radiosensitive (de-escalation, n={grp_counts.get('Radiosensitive', 0)})",
    "Intermediate":   f"Intermediate (SOC [{SOC_LOW}-{SOC_HIGH} Gy], n={in_soc}; "
                      f"+-10% [{SOC_LOW*0.9:.1f}-{SOC_HIGH*1.1:.1f} Gy], n={in_ext})",
    "Radioresistant": f"Radioresistant (under-dosage, n={grp_counts.get('Radioresistant', 0)})",
}
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=RXRSI_COLORS[g], label=legend_labels[g])
    for g in RXRSI_ORDER
]
ax.legend(handles=legend_handles, loc="upper left",
          bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes,
          fontsize=9, frameon=True)
ax.set_xlabel("Patients (ranked by RxRSI)")
ax.set_ylabel("RxRSI (Gy)")
ax.set_title(f"RxRSI distribution -- target GARD = {GARD_T:.1f}")
ax.set_xlim(-1, len(obs_rx))
ax.set_ylim(0, Y_MAX)
plt.tight_layout()
fig.savefig(FIG_DIR / "rxrsi_waterfall.png", bbox_inches="tight")
plt.close()
print("  Saved: rxrsi_waterfall.png")

# 13c. Histogram coloured by group with SOC window
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(obs_rx["RxRSI"].min(), min(obs_rx["RxRSI"].max(), Y_MAX), 25)
for grp in RXRSI_ORDER:
    sub_vals = obs_rx.loc[obs_rx["RxRSI_group"] == grp, "RxRSI"].clip(upper=Y_MAX)
    ax.hist(sub_vals, bins=bins, color=RXRSI_COLORS[grp],
            edgecolor="white", linewidth=0.5, alpha=0.85, label=grp)
ax.axvspan(SOC_LOW, SOC_HIGH, alpha=0.12, color="grey",
           label=f"SOC window ({SOC_LOW}-{SOC_HIGH} Gy)")
ax.axvline(SOC_REF, color="black", linestyle="--", linewidth=1.2,
           label=f"Delivered ({SOC_REF} Gy)")
ax.set_xlabel("RxRSI (Gy)")
ax.set_ylabel("Number of patients")
ax.set_title(f"RxRSI distribution -- target GARD = {GARD_T:.1f}")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "rxrsi_histogram.png")
plt.close()
print("  Saved: rxrsi_histogram.png")

# 13d. RxRSI vs delivered dose with +-10% band
fig, ax = plt.subplots(figsize=(8, 6))
x        = obs_rx["total_dose"].values
y        = obs_rx["RxRSI"].values.clip(max=Y_MAX)
pt_colors = [RXRSI_COLORS[g] for g in obs_rx["RxRSI_group"]]
ax.scatter(x, y, c=pt_colors, edgecolor="black", linewidth=0.4, s=55, alpha=0.85, zorder=3)
x_line = np.linspace(0, 80, 100)
ax.plot(x_line, x_line, color="black", linewidth=1.2, label="RxRSI = delivered")
ax.fill_between(x_line, x_line * 0.9, x_line * 1.1, alpha=0.10,
                color="grey", label="+-10%")
for grp in RXRSI_ORDER:
    ax.scatter([], [], c=RXRSI_COLORS[grp], edgecolor="black", linewidth=0.4,
               s=55, label=grp)
ax.set_xlabel("Delivered dose (Gy)")
ax.set_ylabel("RxRSI (Gy)")
ax.set_title("RxRSI vs delivered dose -- Scott 2021 +-10% rule")
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
fig.savefig(FIG_DIR / "rxrsi_vs_delivered.png")
plt.close()
print("  Saved: rxrsi_vs_delivered.png")

# 13e. Cohort spectrum bar
fig, ax = plt.subplots(figsize=(10, 2.2))
total_n = len(obs_rx)
left = 0.0
for grp in RXRSI_ORDER:
    n_grp = int((obs_rx["RxRSI_group"] == grp).sum())
    if n_grp == 0:
        continue
    width = n_grp / total_n
    ax.barh(0, width, left=left, height=0.6, color=RXRSI_COLORS[grp], edgecolor="white")
    ax.text(left + width / 2, 0, f"{grp}\n(n={n_grp})",
            va="center", ha="center", fontsize=10, color="white", fontweight="bold")
    left += width
ax.set_xlim(0, 1)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_xlabel("Fraction of cohort")
ax.set_title(f"Cohort radiosensitivity spectrum -- target GARD = {GARD_T:.1f}")
plt.tight_layout()
fig.savefig(FIG_DIR / "rxrsi_spectrum_bar.png")
plt.close()
print("  Saved: rxrsi_spectrum_bar.png")


# %% 14. Save results
# ── 14. Save results.csv + .h5ad ──────────────────────────────────────────────
print("\n=== 14. Saving outputs ===")

save_cols = ["RSI", "GARD", "alpha", "GARD_cat_median", "GARD_cat_maxstat",
             "RxRSI", "RxRSI_group", "total_dose", "dose_match",
             "dfs_months", "dfs_event", "response"]
adata.obs[save_cols].to_csv(DATA_DIR / "results.csv")
print(f"  Saved: results.csv  ({len(adata.obs)} patients)")

adata.write_h5ad(DATA_DIR / "gse190826_processed.h5ad")
print(f"  Saved: gse190826_processed.h5ad")
print(f"  (obs: {list(adata.obs.columns)[:10]} ...)")
print(f"  (layers: {list(adata.layers.keys())})")

print("\n=== Done ===")
print(f"  Figures -> {FIG_DIR}")
print(f"  Data    -> {DATA_DIR}")

# %%
#TODO: also create a cut at 70th percentile from low and high like other papers have done and compare results with the median
# ALso make sure to add which percentile the maxstat cut belongs to (60,65, etc).   Maxstat cutpoint: 18.93 Gy  (sits at ~69th percentile of full GARD distribution; search window was [8.26, 25.39])  BH-p: 0.3320, so all gard values within a given percentile that meet the 25% threshold are being used, and NOT the gard values at each of the percentiles (e.g. 15th percentile would be gard 7.3, etc)