"""Run MDS on rosch data with parallel n_init and live progress."""
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

OUT_DIR = "rosch_pipeline_out"
VECTORS_PATH = OUT_DIR + "/rosch_vectors.npy"
RDM_PATH = OUT_DIR + "/rosch_rdm.npy"
LABELS_PATH = OUT_DIR + "/rosch_labels.csv"
OUT_PNG = OUT_DIR + "/rosch_mds.png"
OUT_NPY = OUT_DIR + "/rosch_mds_coords.npy"

N_COMPONENTS = 2
N_INIT = 8
RANDOM_STATE = 42
# Full-N RDM doesn't fit (~106K x 106K). Compute pdist on the subsample only.
USE_PRECOMPUTED_RDM = False
PER_SUBORDINATE = 100   # take this many per 'sub' category (None = all)
COLOR_BY = "basic"      # super | basic | sub


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def load_labels():
    log("loading labels: " + LABELS_PATH)
    df = pd.read_csv(LABELS_PATH)
    log("labels columns: " + str(list(df.columns)))
    log("labels head:\n" + str(df.head()))
    return df


def stratified_indices(labels):
    """Return row indices: first PER_SUBORDINATE rows per subordinate category."""
    if PER_SUBORDINATE is None:
        idx = np.arange(len(labels))
        log("using all " + str(len(idx)) + " rows (no stratification)")
        return idx
    grouped = labels.groupby("sub", sort=False)
    parts = []
    log("stratifying: " + str(PER_SUBORDINATE) + " per sub")
    for name, sub in grouped:
        take = sub.head(PER_SUBORDINATE)
        parts.append(take.index.values)
        log("  " + str(name) + ": " + str(len(sub)) + " available -> took "
            + str(len(take)))
    idx = np.concatenate(parts)
    log("total selected: " + str(len(idx)) + " rows")
    return idx


def load_distance_matrix(idx):
    if USE_PRECOMPUTED_RDM:
        log("loading precomputed RDM: " + RDM_PATH)
        rdm = np.load(RDM_PATH)
        log("RDM full shape: " + str(rdm.shape))
        log("slicing RDM to " + str(len(idx)) + " selected rows/cols")
        rdm = rdm[np.ix_(idx, idx)]
        rdm = (rdm + rdm.T) / 2.0
        np.fill_diagonal(rdm, 0.0)
        return rdm
    log("loading vectors: " + VECTORS_PATH)
    vectors = np.load(VECTORS_PATH)
    log("vectors full shape: " + str(vectors.shape))
    vectors = vectors[idx]
    log("computing pdist on " + str(vectors.shape[0]) + " items")
    t0 = time.time()
    rdm = squareform(pdist(vectors))
    log("pdist done in " + str(round(time.time() - t0, 2)) + "s")
    return rdm


def main():
    t_start = time.time()
    log("start")

    labels_full = load_labels()
    idx = stratified_indices(labels_full)
    labels = labels_full.iloc[idx].reset_index(drop=True)
    rdm = load_distance_matrix(idx)

    log("running MDS: n_components=" + str(N_COMPONENTS)
        + " n_init=" + str(N_INIT)
        + " n_jobs=-1 (all cores)")

    mds = MDS(
        n_components=N_COMPONENTS,
        dissimilarity="precomputed",
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        n_jobs=-1,
        verbose=2,
        normalized_stress="auto",
    )

    t0 = time.time()
    coords = mds.fit_transform(rdm)
    log("MDS done in " + str(round(time.time() - t0, 2)) + "s")
    log("final stress: " + str(round(float(mds.stress_), 4)))
    log("n_iter: " + str(mds.n_iter_))

    np.save(OUT_NPY, coords)
    log("saved coords -> " + OUT_NPY)

    log("plotting - color by '" + COLOR_BY + "'")
    cats = labels[COLOR_BY].astype(str).astype("category")
    codes = cats.cat.codes.values
    n_cats = len(cats.cat.categories)

    log("=== group key (" + str(n_cats) + " groups) ===")
    for i, name in enumerate(cats.cat.categories):
        count = int((codes == i).sum())
        log("  " + str(name) + "  (n=" + str(count) + ")")
    log("=================")

    if n_cats <= 10:
        cmap = plt.get_cmap("tab10")
    elif n_cats <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("nipy_spectral")
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(cats.cat.categories):
        mask = codes == i
        color = cmap(i % cmap.N) if n_cats <= 20 else cmap(i / max(n_cats - 1, 1))
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    s=20, alpha=0.75, color=color, label=str(name))
    plt.legend(title=COLOR_BY, fontsize=8, loc="center left",
               bbox_to_anchor=(1.01, 0.5), markerscale=1.5, framealpha=0.9)

    plt.title("MDS - rosch (stress=" + str(round(float(mds.stress_), 3)) + ")")
    plt.xlabel("MDS 1")
    plt.ylabel("MDS 2")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    log("saved plot -> " + OUT_PNG)
    log("total elapsed: " + str(round(time.time() - t_start, 2)) + "s")


if __name__ == "__main__":
    sys.exit(main())
