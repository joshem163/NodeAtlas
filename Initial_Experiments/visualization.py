import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Files you saved
# -----------------------------
FLA_FILE = "FLA_results.csv"   # e.g., contains rows for cora/citeseer/pubmed
SLA_FILE = "SLA_results.csv"
GCN_FILE = "GCN_results.csv"

datasets = ['Cora', 'CiteSeer', 'texas', 'cornell', 'wisconsin', 'chameleon']

# Column names in your saved files (change if needed)
DATASET_COL = "dataset"
VALUE_COL = "mean_accuracy"

# Map consistent dataset keys (because files may use cora/Cora/CiteSeer/etc.)
def normalize_dataset_name(s: str) -> str:
    s = str(s).strip().lower()
    if s in ["cora"]:
        return "Cora"
    if s in ["citeseer", "cite-seer", "cite_seer"]:
        return "CiteSeer"
    if s in ["pubmed", "pub-med", "pub_med"]:
        return "PubMed"
    return s  # fallback

def read_scores(filepath: str, datasets_order):
    df = pd.read_csv(filepath)
    df[DATASET_COL] = df[DATASET_COL].apply(normalize_dataset_name)

    # If your file contains multiple rows per dataset (e.g., repeated appends),
    # take the last one; or use mean. Here I use last().
    df = df.groupby(DATASET_COL, as_index=False)[VALUE_COL].last()

    scores = []
    for ds in datasets_order:
        row = df[df[DATASET_COL] == ds]
        if len(row) == 0:
            raise ValueError(f"Missing dataset '{ds}' in file: {filepath}")
        scores.append(float(row[VALUE_COL].values[0]))
    return np.array(scores, dtype=float)

# Read arrays from files
FLA = read_scores(FLA_FILE, datasets)
SLA = read_scores(SLA_FILE, datasets)
GCN = read_scores(GCN_FILE, datasets)

print("FLA:", FLA)
print("SLA:", SLA)
print("GCN:", GCN)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 6))

sc = plt.scatter(
    FLA,
    SLA,
    s=100,
    c=GCN,
    cmap='Blues',          # one color: light -> dark
    alpha=0.9,
    edgecolors='black',
    linewidth=1.2
)

for i, name in enumerate(datasets):
    plt.annotate(
        f'{name}',
        (FLA[i], SLA[i]),
        textcoords="offset points",
        xytext=(10, 8),
        fontsize=20
    )

plt.xlabel('FLA', fontsize=20)
plt.ylabel('SLA', fontsize=20)
plt.title('Dataset Positions in (FLA, SLA) Space with GCN Scores', fontsize=17)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

cbar = plt.colorbar(sc)
cbar.set_label('GCN', fontsize=20)
cbar.ax.tick_params(labelsize=20)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()