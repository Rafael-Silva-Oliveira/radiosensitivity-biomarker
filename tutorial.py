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

# Explicit light-theme style so plots look correct both when saved to file
# and when displayed interactively (VS Code, Jupyter). Without these, a dark
# IDE theme can inherit dark rcParams that make text/legends invisible.
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

# ── constants ──────────────────────────────────────────────────────────────────
GEO_ID     = "GSE190826"
RAW_TAR_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE190nnn/GSE190826/suppl/GSE190826_RAW.tar"

N_FRACTIONS  = 28       # 50.4 Gy / 28 fx = 1.8 Gy per fraction (standard long-course rectal CRT)
TOTAL_DOSE   = 50.4     # Gy
DOSE_PER_FX  = TOTAL_DOSE / N_FRACTIONS  # = 1.8 Gy

BETA    = 0.05          # fixed LQ beta (Gy^-2), Scott 2017
D_ALPHA = 2.0           # SF2-assay reference dose for alpha derivation (Gy)

SOC_LOW  = 45.0         # Gy  -- lower edge of rectal neoadjuvant SOC
SOC_HIGH = 54.0         # Gy  -- upper edge
SOC_REF  = 50.4         # Gy  -- delivered to all patients in this cohort

# RSI 10-gene coefficients -- Scott et al. 2017
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

# GRCh38 Ensembl IDs for the 10 RSI genes (featureCounts files use Ensembl IDs)
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
# ── 1. Fetch GSE190826 ─────────────────────────────────────────────────────────
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
    # Columns are Ensembl IDs; rename the 10 RSI genes to HGNC symbols for convenience
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

# The dataset has both pre-CRT and post-CRT samples; RSI is computed on baseline
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
survival_available = adata.obs[TIME_COL].notna().sum() >= 20


# %% 4. Normalize (CPM)
# ── 4. Normalize: raw counts -> CPM ───────────────────────────────────────────
print("\n=== 4. Normalizing (CPM) ===")

raw = adata.layers["raw_counts"]
cpm = CPM().fit_transform(raw)
adata.layers["cpm"]     = cpm
adata.layers["log_cpm"] = np.log1p(cpm)

print("  Layers: raw_counts | cpm | log_cpm")


# %% 5. Compute RSI
# ── 5. Compute RSI ─────────────────────────────────────────────────────────────
print("\n=== 5. Computing RSI ===")

# Steps (Scott 2017 + reference implementation):
#   1. Rank all genes by log-CPM within each sample (descending = rank 1 is highest expr)
#   2. Extract the 10 RSI genes and re-rank them among themselves (relative ranks 1-10)
#   3. RSI = weighted sum of relative ranks, clipped to [0, 1]

missing = [g for g in RSI_GENES if g not in adata.var_names]
present = [g for g in RSI_GENES if g in adata.var_names]
if missing:
    print(f"  WARNING: RSI genes absent from data: {missing}")

log_cpm   = adata.layers["log_cpm"]
var_names = np.array(adata.var_names)

rsi_scores = []
for i in range(adata.n_obs):
    sample_expr = log_cpm[i, :]

    # Global rank (1 = highest expression in that sample)
    global_order    = np.argsort(sample_expr)[::-1]
    global_rank_pos = np.empty(len(sample_expr), dtype=float)
    global_rank_pos[global_order] = np.arange(1, len(sample_expr) + 1)

    # Extract positions for the 10 RSI genes
    gene_positions = {g: global_rank_pos[np.where(var_names == g)[0][0]] for g in present}

    # Re-rank among themselves: sort ascending by global rank position so that
    # the highest-expressed gene (smallest global rank position) gets the highest
    # relative rank (= len(present)), matching the reference np.arange(N, 0, -1).
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
gard  = np.clip(gard, 1, 100)

adata.obs["GARD"]  = gard
adata.obs["alpha"] = alpha
print(f"  GARD range: [{gard.min():.2f}, {gard.max():.2f}]  median: {np.median(gard):.2f}")


# %% 8. Dichotomize GARD (median + maxstat + tertiles)
# ── 8. Dichotomize GARD: median split + maxstat ───────────────────────────────
print("\n=== 8. Dichotomizing GARD ===")

# Derive a balanced cutpoint window: find the tightest [lo, hi] such that any
# split at a value inside the window guarantees >= MIN_GROUP_PCT of patients in
# each group. This is more principled than fixed percentile bounds — it directly
# enforces balance rather than using a proxy. The same window is applied to the
# median, tertile, and maxstat searches so all three are consistent.
# Motivation: splitting near the extremes inflates the log-rank statistic because
# a tiny group separates easily by chance (Lausen & Schumacher 1992).
MIN_GROUP_PCT = 0.25   # each group must contain at least 25% of patients
n_total = len(gard)
min_n   = int(np.ceil(MIN_GROUP_PCT * n_total))

# lo = smallest GARD value such that (gard < lo).sum() >= min_n
# hi = largest  GARD value such that (gard >= hi).sum() >= min_n
gard_sorted = np.sort(gard)
lo_cut = float(gard_sorted[min_n])       # at least min_n patients below this
hi_cut = float(gard_sorted[n_total - min_n - 1])  # at least min_n patients at/above this
gard_trimmed = gard[(gard >= lo_cut) & (gard <= hi_cut)]
pct_lo = (gard < lo_cut).mean() * 100
pct_hi = (gard >= hi_cut).mean() * 100
print(f"  Balanced window (>={int(MIN_GROUP_PCT*100)}% per group): "
      f"[{lo_cut:.2f}, {hi_cut:.2f}]  "
      f"(bottom {pct_lo:.0f}% / top {pct_hi:.0f}% excluded; "
      f"{len(gard_trimmed)}/{n_total} samples in window)")

# Median split — within the balanced window.
gard_median = float(np.median(gard_trimmed))
adata.obs["GARD_cat_median"] = np.where(gard >= gard_median, "High", "Low")
print(f"  Median cutpoint: {gard_median:.2f}  "
      f"groups: {adata.obs['GARD_cat_median'].value_counts().to_dict()}")

# Tertile split — within the balanced window.
t33, t67 = np.percentile(gard_trimmed, [33.3, 66.7])
adata.obs["GARD_tertile"] = pd.cut(
    adata.obs["GARD"], bins=[-np.inf, t33, t67, np.inf],
    labels=["Low", "Intermediate", "High"],
).astype(str)
print(f"  Tertile cuts: T33={t33:.2f}  T67={t67:.2f}")


def maxstat_2g(gard_vals, time_vals, event_vals, lo, hi):
    """
    Scan log-rank chi² over candidate cutpoints within [lo, hi] — the
    pre-computed balanced window where every split has >= MIN_GROUP_PCT patients
    in each group. BH-FDR corrects for the multiple comparisons across cutpoints.
    Returns (best_cut, chi2_array, candidates_array, bh_corrected_p).
    """
    candidates = np.unique(gard_vals[(gard_vals >= lo) & (gard_vals <= hi)])
    chi2_vals, pvals = [], []
    for cut in candidates:
        g_hi = gard_vals >= cut
        g_lo = ~g_hi
        if g_hi.sum() < min_n or g_lo.sum() < min_n:
            chi2_vals.append(0.0); pvals.append(1.0)
            continue
        r = logrank_test(
            time_vals[g_hi], time_vals[g_lo],
            event_observed_A=event_vals[g_hi],
            event_observed_B=event_vals[g_lo],
        )
        chi2_vals.append(r.test_statistic)
        pvals.append(r.p_value)
    chi2_arr = np.array(chi2_vals)
    _, bh_pvals, _, _ = multipletests(pvals, method="fdr_bh")
    best_idx = int(np.argmax(chi2_arr))
    return candidates[best_idx], chi2_arr, candidates, bh_pvals[best_idx]


maxstat_cut = gard_median  # fallback if no survival data
maxstat_bh_p = np.nan

if survival_available:
    obs_s = adata.obs[[TIME_COL, EVENT_COL, "GARD"]].dropna()
    obs_s = obs_s[obs_s[TIME_COL] > 0]
    if len(obs_s) >= 20:
        maxstat_cut, chi2_arr, cut_arr, maxstat_bh_p = maxstat_2g(
            obs_s["GARD"].values, obs_s[TIME_COL].values, obs_s[EVENT_COL].values,
            lo=lo_cut, hi=hi_cut)
        maxstat_pct = float(np.mean(obs_s["GARD"].values < maxstat_cut) * 100)
        print(f"  Maxstat cutpoint: {maxstat_cut:.2f} Gy  "
              f"(sits at ~{maxstat_pct:.0f}th percentile of full GARD distribution; "
              f"search window was [{lo_cut:.2f}, {hi_cut:.2f}])  BH-p: {maxstat_bh_p:.4f}")

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
        print("  Saved: maxstat_scan.png")

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

if not survival_available:
    print("  No survival data -- skipping Cox models")
else:
    obs_cox = adata.obs.copy()
    obs_cox = obs_cox[obs_cox[TIME_COL] > 0].dropna(subset=[TIME_COL, EVENT_COL])

    obs_cox["RSI_z"]              = zscore(obs_cox["RSI"],  nan_policy="omit")
    obs_cox["GARD_z"]             = zscore(obs_cox["GARD"], nan_policy="omit")
    obs_cox["GARD_cat_median_bin"]  = (obs_cox["GARD_cat_median"]  == "High").astype(int)
    obs_cox["GARD_cat_maxstat_bin"] = (obs_cox["GARD_cat_maxstat"] == "High").astype(int)

    # Encode pCR as binary covariate
    obs_cox["pcr_bin"] = (obs_cox["response"].str.strip().str.lower() == "pcr").astype(int)

    # Detect any other clinical covariates available in obs
    candidate_covars = ["age", "sex", "gender", "pcr_bin"]
    available_covars = [c for c in candidate_covars if c in obs_cox.columns]
    print(f"  Clinical covariates: {available_covars}")

    cox_results = []

    def fit_cox(df, cols, label):
        sub = df[cols + [TIME_COL, EVENT_COL]].dropna()
        if len(sub) < 15:
            return None
        sub = pd.get_dummies(sub, drop_first=True)
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(sub, duration_col=TIME_COL, event_col=EVENT_COL, show_progress=False)
        except Exception as e:
            print(f"    WARN {label}: {e}")
            return None
        import contextlib, io as _io
        with contextlib.redirect_stdout(_io.StringIO()):
            ph_warnings = cph.check_assumptions(sub, p_value_threshold=0.05, show_plots=False)
        primary = cols[0]
        coef_row = next(
            (cph.summary.loc[idx] for idx in cph.summary.index if primary in str(idx)),
            cph.summary.iloc[0],
        )
        return {
            "model":    label,
            "n":        len(sub),
            "predictor": primary,
            "HR":       round(float(np.exp(coef_row["coef"])), 3),
            "HR_lower": round(float(np.exp(coef_row["coef lower 95%"])), 3),
            "HR_upper": round(float(np.exp(coef_row["coef upper 95%"])), 3),
            "p":        round(float(coef_row["p"]), 4),
            "c_index":  round(float(cph.concordance_index_), 3),
            "AIC":      round(float(cph.AIC_partial_), 1),
            "ph_warning": bool(ph_warnings),
        }

    predictors = ["RSI_z", "GARD_z", "GARD_cat_median_bin", "GARD_cat_maxstat_bin"]

    for pred in predictors:
        res = fit_cox(obs_cox, [pred], f"Univariate: {pred}")
        if res:
            cox_results.append(res)
            print(f"  {res['model']}: HR={res['HR']} [{res['HR_lower']}-{res['HR_upper']}]  "
                  f"p={res['p']}  C={res['c_index']}")

    for pred in predictors:
        cols = [pred] + available_covars
        res = fit_cox(obs_cox, cols, f"Multivariate: {pred}")
        if res:
            cox_results.append(res)
            print(f"  {res['model']}: HR={res['HR']} [{res['HR_lower']}-{res['HR_upper']}]  "
                  f"p={res['p']}  C={res['c_index']}  PH_warn={res['ph_warning']}")

    cox_df = pd.DataFrame(cox_results)
    cox_df.to_csv(DATA_DIR / "cox_results.csv", index=False)
    print("  Saved: cox_results.csv")

    def forest_plot(df, title, outpath, color):
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.8)))
        y = list(range(len(df)))
        ax.scatter(df["HR"].values, y, color=color, zorder=3, s=60)
        for i, (_, row) in enumerate(df.iterrows()):
            ax.hlines(i, row["HR_lower"], row["HR_upper"], color=color, linewidth=2)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(df["predictor"].tolist())
        ax.set_xlabel("Hazard Ratio (95% CI)")
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(outpath)
        plt.close()

    forest_plot(
        cox_df[cox_df["model"].str.startswith("Univariate")],
        "Univariate Cox -- DFS", FIG_DIR / "forest_univariate.png", "#4878d0",
    )
    forest_plot(
        cox_df[cox_df["model"].str.startswith("Multivariate")],
        "Multivariate Cox -- DFS", FIG_DIR / "forest_multivariate.png", "#ee854a",
    )
    print("  Saved: forest_univariate.png, forest_multivariate.png")


# %% 10. Kaplan-Meier plots
# ── 10. Kaplan-Meier plots ────────────────────────────────────────────────────
print("\n=== 10. Kaplan-Meier plots ===")

GROUP_COLORS_2 = {"High": "#2ca02c", "Low": "#d62728"}
TERTILE_COLORS = {"Low": "#d62728", "Intermediate": "#E6A817", "High": "#2ca02c"}


def plot_km_2group(obs_df, group_col, t_col, e_col, title, outpath, colors=None):
    colors = colors or GROUP_COLORS_2
    df = obs_df[[group_col, t_col, e_col]].copy()
    df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
    df[e_col] = pd.to_numeric(df[e_col], errors="coerce")
    df = df.dropna()
    df = df[df[t_col] > 0]
    if len(df) < 10:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    grps = sorted(df[group_col].dropna().unique(), reverse=True)  # High first
    for grp in grps:
        mask = df[group_col] == grp
        if mask.sum() < 5:
            continue
        kmf.fit(df.loc[mask, t_col], df.loc[mask, e_col], label=grp)
        kmf.plot_survival_function(ax=ax, color=colors.get(grp, "grey"), ci_show=True)

    # Log-rank p between the two extremes
    grp_a = "High" if "High" in grps else grps[0]
    grp_b = "Low" if "Low" in grps else grps[-1]
    if (df[group_col] == grp_a).sum() >= 5 and (df[group_col] == grp_b).sum() >= 5:
        lr = logrank_test(
            df.loc[df[group_col] == grp_a, t_col],
            df.loc[df[group_col] == grp_b, t_col],
            event_observed_A=df.loc[df[group_col] == grp_a, e_col],
            event_observed_B=df.loc[df[group_col] == grp_b, e_col],
        )
        ax.text(0.05, 0.05, f"log-rank p = {lr.p_value:.4f}",
                transform=ax.transAxes, fontsize=10)

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("DFS probability")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close()


def plot_km_3group(obs_df, group_col, t_col, e_col, title, outpath, colors):
    df = obs_df[[group_col, t_col, e_col]].copy()
    df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
    df[e_col] = pd.to_numeric(df[e_col], errors="coerce")
    df = df.dropna()
    df = df[df[t_col] > 0]
    if len(df) < 15:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    for grp in ["High", "Intermediate", "Low"]:
        mask = df[group_col] == grp
        if mask.sum() < 5:
            continue
        kmf.fit(df.loc[mask, t_col], df.loc[mask, e_col], label=grp)
        kmf.plot_survival_function(ax=ax, color=colors.get(grp, "grey"), ci_show=True)
    # Log-rank between High and Low
    hi = df[group_col] == "High"
    lo = df[group_col] == "Low"
    if hi.sum() >= 5 and lo.sum() >= 5:
        lr = logrank_test(
            df.loc[hi, t_col], df.loc[lo, t_col],
            event_observed_A=df.loc[hi, e_col], event_observed_B=df.loc[lo, e_col],
        )
        ax.text(0.05, 0.05, f"High vs Low log-rank p = {lr.p_value:.4f}",
                transform=ax.transAxes, fontsize=10)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("DFS probability")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close()


if not survival_available:
    print("  No survival data -- KM plots skipped")
else:
    plot_km_2group(adata.obs, "GARD_cat_median", TIME_COL, EVENT_COL,
                   f"DFS -- GARD median split (cut={gard_median:.1f})",
                   FIG_DIR / "km_dfs_gard_median.png")
    print("  Saved: km_dfs_gard_median.png")

    plot_km_2group(adata.obs, "GARD_cat_maxstat", TIME_COL, EVENT_COL,
                   f"DFS -- GARD maxstat split (cut={maxstat_cut:.1f})",
                   FIG_DIR / "km_dfs_gard_maxstat.png")
    print("  Saved: km_dfs_gard_maxstat.png")

    plot_km_3group(adata.obs, "GARD_tertile", TIME_COL, EVENT_COL,
                   f"DFS -- GARD tertile split (T33={t33:.1f}, T67={t67:.1f})",
                   FIG_DIR / "km_dfs_gard_tertile.png",
                   colors=TERTILE_COLORS)
    print("  Saved: km_dfs_gard_tertile.png")

    # KM by pCR response
    plot_km_2group(adata.obs, "response", TIME_COL, EVENT_COL,
                   "DFS -- pCR vs non-pCR",
                   FIG_DIR / "km_dfs_response.png",
                   colors={"pCR": "#2ca02c", "Non-pCR": "#d62728"})
    print("  Saved: km_dfs_response.png")


# %% 11. Compute RxRSI
# ── 11. RxRSI ─────────────────────────────────────────────────────────────────
print("\n=== 11. Computing RxRSI ===")

# GARD_T = maxstat cutpoint (or median fallback).
# RxRSI = quadratic inversion: given a target GARD, what total dose does each patient need?
GARD_T = maxstat_cut if not np.isnan(maxstat_cut) else gard_median
print(f"  Target GARD (GARD_T): {GARD_T:.2f}")

n = float(N_FRACTIONS)
disc  = alpha**2 + 4.0 * BETA * GARD_T / n
disc  = np.where(disc < 0, np.nan, disc)
rxrsi = (-alpha + np.sqrt(disc)) / (2.0 * BETA / n)
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
for n_fx, ls in [(20, "--"), (28, "-"), (35, ":")]:
    disc_c = np.clip(alpha_grid**2 + 4.0 * BETA * GARD_T / n_fx, 0, None)
    rx_c   = (-alpha_grid + np.sqrt(disc_c)) / (2.0 * BETA / n_fx)
    ax.plot(rsi_grid, np.clip(rx_c, 0, Y_MAX), color="#1f77b4",
            linestyle=ls, linewidth=2, label=f"RxRSI @ {n_fx} fx", zorder=1)

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
             "GARD_tertile", "RxRSI", "RxRSI_group", "total_dose", "dose_match",
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