#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip -q install pandas numpy scikit-learn tqdm drain3 joblib')


# In[1]:


import os, re, math, zipfile, urllib.request, glob, random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize

from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


# In[2]:


DATA_DIR = Path("datasets")
DATA_DIR.mkdir(exist_ok=True)

zip_url = "https://github.com/logpai/loghub/archive/refs/heads/master.zip"
zip_path = DATA_DIR / "loghub_master.zip"

if not zip_path.exists():
    print("Downloading LogHub...")
    urllib.request.urlretrieve(zip_url, zip_path)
    print("Downloaded:", zip_path)

extract_root = DATA_DIR / "loghub-master"
if not extract_root.exists():
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    print("Extracted to:", extract_root)

print("Ready:", extract_root.resolve())


# In[3]:


# Find logs (HDFS first, else any *.log / *.log.gz)
hdfs = glob.glob(str(extract_root / "**" / "*HDFS*.log"), recursive=True) +        glob.glob(str(extract_root / "**" / "*hdfs*.log"), recursive=True) +        glob.glob(str(extract_root / "**" / "*HDFS*.log.gz"), recursive=True) +        glob.glob(str(extract_root / "**" / "*hdfs*.log.gz"), recursive=True)

any_logs = glob.glob(str(extract_root / "**" / "*.log"), recursive=True) +            glob.glob(str(extract_root / "**" / "*.log.gz"), recursive=True)

candidates = hdfs if len(hdfs) else any_logs

print("Found candidates:", len(candidates))
for i, p in enumerate(candidates[:30]):
    print(f"[{i}] {p}")

assert len(candidates) > 0, "No log files found in extracted LogHub. We'll switch source if needed."
path = candidates[0]
print("\nUsing log file:", path)


# In[4]:


def make_template_miner(sim_th=0.4, depth=4):
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = sim_th
    cfg.drain_depth = depth
    cfg.profiling_enabled = False

    # IMPORTANT: pass config as second argument (or named argument)
    try:
        return TemplateMiner(persistence_handler=None, config=cfg)
    except TypeError:
        return TemplateMiner(None, cfg)

tm = make_template_miner(sim_th=0.4, depth=4)
print("✅ Drain3 ready")


# In[5]:


import gzip

BLOCK_RE = re.compile(r"(blk_-?\d+)", re.IGNORECASE)

def stream_lines(p, max_lines=None):
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            yield line

def extract_entity(line: str):
    # HDFS-style entity: BlockId. If not found, we fall back to "GLOBAL"
    m = BLOCK_RE.search(line)
    return m.group(1) if m else "GLOBAL"

def build_sequences(path, max_lines=200000):
    seq_map = defaultdict(list)
    template_counts = Counter()

    total = 0
    for line in tqdm(stream_lines(path, max_lines=max_lines), total=max_lines):
        total += 1
        ent = extract_entity(line)
        r = tm.add_log_message(line.strip())
        tid = r["cluster_id"]
        seq_map[ent].append(tid)
        template_counts[tid] += 1

    stats = {
        "total_lines": total,
        "entities": len(seq_map),
        "num_templates": len(template_counts),
        "template_counts": template_counts
    }
    return seq_map, stats


# In[6]:


seq_map, stats = build_sequences(path, max_lines=200000)

print("✅ Built sequences")
print({k:v for k,v in stats.items() if k!="template_counts"})


# In[7]:


def gini_impurity(counter: Counter):
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return 1.0 - sum((c/n)**2 for c in counter.values())

def darkness_v2(seq, global_template_counts):
    n = len(seq)
    if n == 0:
        return 1.0

    c = Counter(seq)
    redundancy = 1.0 - (len(c) / n)

    inv = np.array([1.0 / math.sqrt(global_template_counts[t]) for t in seq], dtype=float)
    rarity = float(np.clip(inv.mean() / (inv.mean() + 1.0), 0, 1))

    # burstiness = longest run / length
    max_run, cur = 1, 1
    for i in range(1, n):
        if seq[i] == seq[i-1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    burst = float(np.clip(max_run / max(2, n), 0, 1))

    ambiguity = float(np.clip(gini_impurity(c), 0, 1))
    context_poverty = float(np.clip(1.0 / math.sqrt(n), 0, 1))

    score = (0.28*redundancy + 0.22*rarity + 0.18*burst + 0.17*ambiguity + 0.15*context_poverty)
    return float(1 - math.exp(-2.2*score))

darkness = {e: darkness_v2(seq, stats["template_counts"]) for e, seq in seq_map.items()}
pd.Series(darkness).describe()


# In[12]:


from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import IsolationForest

svd = TruncatedSVD(n_components=128, random_state=42)
pipe = make_pipeline(svd, Normalizer(copy=False))
X_dense = pipe.fit_transform(X)  # X can be any sparse type

model = IsolationForest(n_estimators=250, contamination=0.02, random_state=42, n_jobs=-1)
model.fit(X_dense)
anom_scores = -model.decision_function(X_dense)

print("✅ Trained IsolationForest on SVD-dense features:", X_dense.shape)


# In[8]:


def seq_to_ngrams(seq, n=3):
    if len(seq) < n:
        return []
    return ["_".join([f"e{seq[i+j]}" for j in range(n)]) for i in range(len(seq)-n+1)]

def build_text_adaptive(seq, max_tokens=4000):
    L = len(seq)
    if L >= 3:
        tokens = seq_to_ngrams(seq, n=3)
    elif L == 2:
        tokens = seq_to_ngrams(seq, n=2)
    elif L == 1:
        tokens = [f"e{seq[0]}"]
    else:
        tokens = []

    if len(tokens) > max_tokens:
        idx = np.linspace(0, len(tokens)-1, max_tokens).astype(int)
        tokens = [tokens[i] for i in idx]
    return " ".join(tokens)

# Build texts
entities_f, texts_f = [], []
for e, seq in seq_map.items():
    t = build_text_adaptive(seq)
    if t.strip():
        entities_f.append(e)
        texts_f.append(t)

print("Entities kept:", len(entities_f), "| dropped:", len(seq_map)-len(entities_f))

# Sanity: show distribution of sequence lengths
lens = pd.Series([len(seq_map[e]) for e in entities_f])
print(lens.describe())
print("Counts of small lengths:\n", lens.value_counts().head(10))


# In[9]:


from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import IsolationForest
import numpy as np

hv = HashingVectorizer(n_features=2**18, alternate_sign=False, norm=None)
X = hv.transform(texts_f)
X = sparse.csr_matrix(X)

row_nnz = np.array(X.getnnz(axis=1)).ravel()
keep = row_nnz > 0
X = X[keep]
entities_ff = [e for e, k in zip(entities_f, keep) if k]

print("After removing all-zero rows:", X.shape, "| entities:", len(entities_ff))

svd = TruncatedSVD(n_components=min(128, X.shape[1]-1) if X.shape[1] > 2 else 2, random_state=42)
pipe = make_pipeline(svd, Normalizer(copy=False))
X_dense = pipe.fit_transform(X)
X_dense = np.nan_to_num(X_dense, nan=0.0, posinf=0.0, neginf=0.0)

model = IsolationForest(n_estimators=250, contamination=0.02, random_state=42, n_jobs=-1)
model.fit(X_dense)
anom_scores = -model.decision_function(X_dense)

print("✅ Trained. X_dense:", X_dense.shape)


# In[10]:


from pathlib import Path
import pandas as pd

OUT_DIR = Path("darklog_outputs")
OUT_DIR.mkdir(exist_ok=True)

rows = []
for e, a in zip(entities_ff, anom_scores):
    d = float(darkness.get(e, 0.0))
    rows.append((e, float(a), d, 0.65*float(a) + 0.35*d))

report = pd.DataFrame(rows, columns=["entity", "anomaly_score", "darkness_score", "combined"])           .sort_values("combined", ascending=False)

report.to_csv(OUT_DIR / "ranked_report.csv", index=False)
print("✅ Saved:", (OUT_DIR / "ranked_report.csv").resolve())
report.head(10)


# In[11]:


def explain_entity(entity, seq_map, template_counts, top_k=10):
    seq = seq_map[entity]
    c = Counter(seq)

    rare = sorted(c.items(), key=lambda x: template_counts[x[0]])[:top_k]
    common = c.most_common(top_k)

    # longest run
    max_run, cur = 1, 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1

    return {
        "entity": entity,
        "n_events": len(seq),
        "n_unique_templates": len(c),
        "max_run_same_event": max_run,
        "rarest_templates_in_seq": [(tid, int(cnt), int(template_counts[tid])) for tid, cnt in rare],
        "most_common_templates_in_seq": [(tid, int(cnt)) for tid, cnt in common],
    }

top_entity = report.iloc[0]["entity"]
explain = explain_entity(top_entity, seq_map, stats["template_counts"])
explain


# In[12]:


import networkx as nx

G = nx.Graph()

# Add nodes
for entity, seq in seq_map.items():
    G.add_node(entity, node_type="entity")
    for tid in seq:
        tnode = f"T{tid}"
        G.add_node(tnode, node_type="template")
        G.add_edge(entity, tnode, weight=G.get_edge_data(entity, tnode, {}).get("weight", 0) + 1)

print("Graph built:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# In[13]:


graph_features = {}

for e in seq_map.keys():
    if not G.has_node(e):
        continue

    neighbors = list(G.neighbors(e))
    weights = [G[e][n]["weight"] for n in neighbors]

    graph_features[e] = {
        "degree": len(neighbors),
        "weighted_degree": sum(weights),
        "avg_edge_weight": np.mean(weights) if weights else 0,
        "max_edge_weight": max(weights) if weights else 0,
        "template_diversity": len(neighbors)
    }

gf_df = pd.DataFrame.from_dict(graph_features, orient="index")
gf_df.head()


# In[14]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

Xg = gf_df.fillna(0.0).values
Xg = StandardScaler().fit_transform(Xg)

graph_if = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)
graph_if.fit(Xg)

graph_scores = -graph_if.decision_function(Xg)
gf_df["graph_anomaly"] = graph_scores
gf_df.head()


# In[15]:


final = report.set_index("entity").join(gf_df, how="left").fillna(0)

final["final_score"] = (
    0.4 * final["anomaly_score"] +
    0.3 * final["darkness_score"] +
    0.3 * final["graph_anomaly"]
)

final_ranked = final.sort_values("final_score", ascending=False)
final_ranked.head(10)


# In[16]:


from pathlib import Path

OUT_DIR = Path("darklog_outputs")
OUT_DIR.mkdir(exist_ok=True)

report.to_csv(OUT_DIR / "original_sequence_darkness_report.csv", index=False)
gf_df.to_csv(OUT_DIR / "original_graph_features.csv")
final_ranked.to_csv(OUT_DIR / "original_final_ranked_results.csv")

print("Original outputs saved successfully.")


# In[17]:


ablation = final.copy()

ablation["score_anomaly_only"] = ablation["anomaly_score"]
ablation["score_darkness_only"] = ablation["darkness_score"]
ablation["score_graph_only"] = ablation["graph_anomaly"]

ablation["score_anomaly_darkness"] = (
    0.65 * ablation["anomaly_score"] +
    0.35 * ablation["darkness_score"]
)

ablation["score_full_da_egad"] = (
    0.4 * ablation["anomaly_score"] +
    0.3 * ablation["darkness_score"] +
    0.3 * ablation["graph_anomaly"]
)

ablation_result = ablation.sort_values("score_full_da_egad", ascending=False)
ablation_result.to_csv(OUT_DIR / "ablation_study_results.csv")

ablation_result.head(10)


# In[18]:


weight_settings = [
    (0.5, 0.3, 0.2),
    (0.4, 0.3, 0.3),
    (0.3, 0.4, 0.3),
    (0.3, 0.3, 0.4),
    (0.6, 0.2, 0.2)
]

base_top10 = set(final_ranked.head(10).index)
sensitivity_rows = []

for a, b, c in weight_settings:
    temp = final.copy()
    temp["sensitivity_score"] = (
        a * temp["anomaly_score"] +
        b * temp["darkness_score"] +
        c * temp["graph_anomaly"]
    )
    top10 = set(temp.sort_values("sensitivity_score", ascending=False).head(10).index)
    overlap = len(base_top10.intersection(top10))

    sensitivity_rows.append({
        "alpha_anomaly": a,
        "beta_darkness": b,
        "gamma_graph": c,
        "top10_overlap_with_original": overlap
    })

sensitivity_df = pd.DataFrame(sensitivity_rows)
sensitivity_df.to_csv(OUT_DIR / "weight_sensitivity_analysis.csv", index=False)

sensitivity_df


# In[20]:


# Make sure required ablation scores exist inside final itself
final["score_anomaly_darkness"] = (
    0.65 * final["anomaly_score"] +
    0.35 * final["darkness_score"]
)

final["score_full_da_egad"] = (
    0.4 * final["anomaly_score"] +
    0.3 * final["darkness_score"] +
    0.3 * final["graph_anomaly"]
)

top10_anomaly = set(
    final.sort_values("anomaly_score", ascending=False).head(10).index
)

top10_darkness = set(
    final.sort_values("darkness_score", ascending=False).head(10).index
)

top10_graph = set(
    final.sort_values("graph_anomaly", ascending=False).head(10).index
)

top10_ad = set(
    final.sort_values("score_anomaly_darkness", ascending=False).head(10).index
)

top10_full = set(
    final.sort_values("score_full_da_egad", ascending=False).head(10).index
)

ablation_summary = pd.DataFrame([
    {
        "Method": "Anomaly Only",
        "Overlap_with_DA_EGAD_Top10": len(top10_anomaly.intersection(top10_full))
    },
    {
        "Method": "Darkness Only",
        "Overlap_with_DA_EGAD_Top10": len(top10_darkness.intersection(top10_full))
    },
    {
        "Method": "Graph Only",
        "Overlap_with_DA_EGAD_Top10": len(top10_graph.intersection(top10_full))
    },
    {
        "Method": "Anomaly + Darkness",
        "Overlap_with_DA_EGAD_Top10": len(top10_ad.intersection(top10_full))
    },
    {
        "Method": "DA-EGAD Full",
        "Overlap_with_DA_EGAD_Top10": 10
    }
])

ablation_summary.to_csv(OUT_DIR / "ablation_summary_top10_overlap.csv", index=False)
ablation_summary


# In[21]:


# Find BGL log files inside LogHub
bgl_candidates = glob.glob(str(extract_root / "**" / "*BGL*.log"), recursive=True) +                  glob.glob(str(extract_root / "**" / "*bgl*.log"), recursive=True) +                  glob.glob(str(extract_root / "**" / "*BGL*.log.gz"), recursive=True) +                  glob.glob(str(extract_root / "**" / "*bgl*.log.gz"), recursive=True)

print("BGL candidates found:", len(bgl_candidates))
for i, p in enumerate(bgl_candidates[:20]):
    print(f"[{i}] {p}")

assert len(bgl_candidates) > 0, "No BGL file found."
bgl_path = bgl_candidates[0]
print("Using BGL file:", bgl_path)


# In[22]:


BGL_NODE_RE = re.compile(r"(R\d{2}-M\d-N\d-C:[A-Z]\d+-U\d+)", re.IGNORECASE)

def extract_entity_bgl(line: str):
    m = BGL_NODE_RE.search(line)
    if m:
        return m.group(1)
    return "GLOBAL"


# In[23]:


def build_sequences_for_dataset(path, entity_extractor, max_lines=200000):
    local_tm = make_template_miner(sim_th=0.4, depth=4)
    seq_map_local = defaultdict(list)
    template_counts_local = Counter()

    total = 0
    for line in tqdm(stream_lines(path, max_lines=max_lines), total=max_lines):
        total += 1
        ent = entity_extractor(line)
        r = local_tm.add_log_message(line.strip())
        tid = r["cluster_id"]
        seq_map_local[ent].append(tid)
        template_counts_local[tid] += 1

    stats_local = {
        "total_lines": total,
        "entities": len(seq_map_local),
        "num_templates": len(template_counts_local),
        "template_counts": template_counts_local
    }

    return seq_map_local, stats_local


# In[24]:


bgl_seq_map, bgl_stats = build_sequences_for_dataset(
    bgl_path,
    extract_entity_bgl,
    max_lines=200000
)

print("BGL stats:")
print({k:v for k,v in bgl_stats.items() if k != "template_counts"})


# In[25]:


bgl_darkness = {
    e: darkness_v2(seq, bgl_stats["template_counts"])
    for e, seq in bgl_seq_map.items()
}

bgl_darkness_series = pd.Series(bgl_darkness)

print("BGL Darkness Summary:")
print(bgl_darkness_series.describe())


# In[26]:


bgl_entities_f, bgl_texts_f = [], []

for e, seq in bgl_seq_map.items():
    t = build_text_adaptive(seq)
    if t.strip():
        bgl_entities_f.append(e)
        bgl_texts_f.append(t)

print("BGL entities kept:", len(bgl_entities_f), "| dropped:", len(bgl_seq_map)-len(bgl_entities_f))

bgl_lens = pd.Series([len(bgl_seq_map[e]) for e in bgl_entities_f])
print(bgl_lens.describe())
print("Counts of small lengths:\n", bgl_lens.value_counts().head(10))


# In[27]:


from scipy import sparse

bgl_hv = HashingVectorizer(n_features=2**18, alternate_sign=False, norm=None)
bgl_X = bgl_hv.transform(bgl_texts_f)
bgl_X = sparse.csr_matrix(bgl_X)

bgl_row_nnz = np.array(bgl_X.getnnz(axis=1)).ravel()
bgl_keep = bgl_row_nnz > 0

bgl_X = bgl_X[bgl_keep]
bgl_entities_ff = [e for e, k in zip(bgl_entities_f, bgl_keep) if k]

print("BGL after removing all-zero rows:", bgl_X.shape, "| entities:", len(bgl_entities_ff))

bgl_svd = TruncatedSVD(
    n_components=min(128, bgl_X.shape[1]-1) if bgl_X.shape[1] > 2 else 2,
    random_state=42
)

bgl_pipe = make_pipeline(bgl_svd, Normalizer(copy=False))
bgl_X_dense = bgl_pipe.fit_transform(bgl_X)
bgl_X_dense = np.nan_to_num(bgl_X_dense, nan=0.0, posinf=0.0, neginf=0.0)

bgl_model = IsolationForest(
    n_estimators=250,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

bgl_model.fit(bgl_X_dense)
bgl_anom_scores = -bgl_model.decision_function(bgl_X_dense)

print("BGL sequence anomaly trained:", bgl_X_dense.shape)


# In[28]:


bgl_rows = []

for e, a in zip(bgl_entities_ff, bgl_anom_scores):
    d = float(bgl_darkness.get(e, 0.0))
    bgl_rows.append((e, float(a), d, 0.65*float(a) + 0.35*d))

bgl_report = pd.DataFrame(
    bgl_rows,
    columns=["entity", "anomaly_score", "darkness_score", "combined"]
).sort_values("combined", ascending=False)

bgl_report.to_csv(OUT_DIR / "bgl_sequence_darkness_report.csv", index=False)

bgl_report.head(10)


# In[30]:


bgl_G = nx.Graph()

for entity, seq in bgl_seq_map.items():
    bgl_G.add_node(entity, node_type="entity")

    for tid in seq:
        tnode = f"T{tid}"

        bgl_G.add_node(tnode, node_type="template")

        bgl_G.add_edge(
            entity,
            tnode,
            weight=bgl_G.get_edge_data(entity, tnode, {}).get("weight", 0) + 1
        )

print("BGL Graph")
print("Nodes:", bgl_G.number_of_nodes())
print("Edges:", bgl_G.number_of_edges())


# In[31]:


bgl_graph_features = {}

for e in bgl_seq_map.keys():

    if not bgl_G.has_node(e):
        continue

    neighbors = list(bgl_G.neighbors(e))
    weights = [bgl_G[e][n]["weight"] for n in neighbors]

    bgl_graph_features[e] = {
        "degree": len(neighbors),
        "weighted_degree": sum(weights),
        "avg_edge_weight": np.mean(weights) if weights else 0,
        "max_edge_weight": max(weights) if weights else 0,
        "template_diversity": len(neighbors)
    }

bgl_gf_df = pd.DataFrame.from_dict(
    bgl_graph_features,
    orient="index"
)

bgl_gf_df.head()


# In[32]:


Xg_bgl = bgl_gf_df.fillna(0.0).values
Xg_bgl = StandardScaler().fit_transform(Xg_bgl)

bgl_graph_if = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

bgl_graph_if.fit(Xg_bgl)

bgl_graph_scores = -bgl_graph_if.decision_function(Xg_bgl)

bgl_gf_df["graph_anomaly"] = bgl_graph_scores

bgl_gf_df.head()


# In[33]:


bgl_final = (
    bgl_report
    .set_index("entity")
    .join(bgl_gf_df, how="left")
    .fillna(0)
)

bgl_final["final_score"] = (
    0.4 * bgl_final["anomaly_score"] +
    0.3 * bgl_final["darkness_score"] +
    0.3 * bgl_final["graph_anomaly"]
)

bgl_final_ranked = bgl_final.sort_values(
    "final_score",
    ascending=False
)

bgl_final_ranked.head(10)


# In[34]:


dataset_summary = pd.DataFrame([
    {
        "Dataset": "HDFS",
        "Entities": stats["entities"],
        "Templates": stats["num_templates"],
        "Top_Final_Score": float(final_ranked.iloc[0]["final_score"])
    },
    {
        "Dataset": "BGL",
        "Entities": bgl_stats["entities"],
        "Templates": bgl_stats["num_templates"],
        "Top_Final_Score": float(bgl_final_ranked.iloc[0]["final_score"])
    }
])

dataset_summary


# In[35]:


cross_dataset_summary = pd.DataFrame([
    {
        "Dataset": "HDFS",
        "Entities": stats["entities"],
        "Templates": stats["num_templates"],
        "Top_DA_EGAD_Score": round(float(final_ranked.iloc[0]["final_score"]), 6)
    },
    {
        "Dataset": "BGL",
        "Entities": bgl_stats["entities"],
        "Templates": bgl_stats["num_templates"],
        "Top_DA_EGAD_Score": round(float(bgl_final_ranked.iloc[0]["final_score"]), 6)
    }
])

cross_dataset_summary


# In[36]:


cross_dataset_summary.to_csv(
    OUT_DIR / "cross_dataset_summary.csv",
    index=False
)


# In[37]:


thunder_candidates = (
    glob.glob(str(extract_root / "**" / "*Thunderbird*.log"), recursive=True) +
    glob.glob(str(extract_root / "**" / "*thunderbird*.log"), recursive=True) +
    glob.glob(str(extract_root / "**" / "*Thunderbird*.log.gz"), recursive=True) +
    glob.glob(str(extract_root / "**" / "*thunderbird*.log.gz"), recursive=True)
)

print("Thunderbird candidates found:", len(thunder_candidates))

for i, p in enumerate(thunder_candidates[:20]):
    print(f"[{i}] {p}")

assert len(thunder_candidates) > 0, "No Thunderbird dataset found."

thunder_path = thunder_candidates[0]

print("\nUsing Thunderbird file:")
print(thunder_path)


# In[38]:


THUNDER_NODE_RE = re.compile(r"([A-Za-z0-9\-_]+\[\d+\])")

def extract_entity_thunder(line):
    m = THUNDER_NODE_RE.search(line)

    if m:
        return m.group(1)

    return "GLOBAL"


# In[39]:


th_seq_map, th_stats = build_sequences_for_dataset(
    thunder_path,
    extract_entity_thunder,
    max_lines=200000
)

print("Thunderbird stats:")
print({k:v for k,v in th_stats.items() if k != "template_counts"})


# In[40]:


print("Sample Thunderbird entities:")
print(list(th_seq_map.keys())[:20])


# In[41]:


entity_lengths = pd.Series(
    [len(v) for v in th_seq_map.values()]
)

print(entity_lengths.describe())
print("\nTop sequence lengths:")
print(entity_lengths.sort_values(ascending=False).head(20))


# In[42]:


th_darkness = {
    e: darkness_v2(seq, th_stats["template_counts"])
    for e, seq in th_seq_map.items()
}

th_darkness_series = pd.Series(th_darkness)

print(th_darkness_series.describe())


# In[43]:


th_entities_f = []
th_texts_f = []

for e, seq in th_seq_map.items():
    t = build_text_adaptive(seq)

    if t.strip():
        th_entities_f.append(e)
        th_texts_f.append(t)

print(
    "Entities kept:",
    len(th_entities_f),
    "| dropped:",
    len(th_seq_map) - len(th_entities_f)
)


# In[44]:


th_hv = HashingVectorizer(
    n_features=2**18,
    alternate_sign=False,
    norm=None
)

th_X = th_hv.transform(th_texts_f)

from scipy import sparse
th_X = sparse.csr_matrix(th_X)

th_row_nnz = np.array(th_X.getnnz(axis=1)).ravel()
th_keep = th_row_nnz > 0

th_X = th_X[th_keep]

th_entities_ff = [
    e for e, k in zip(th_entities_f, th_keep)
    if k
]

print(
    "After removing all-zero rows:",
    th_X.shape
)

th_svd = TruncatedSVD(
    n_components=min(128, th_X.shape[1]-1)
    if th_X.shape[1] > 2 else 2,
    random_state=42
)

th_pipe = make_pipeline(
    th_svd,
    Normalizer(copy=False)
)

th_X_dense = th_pipe.fit_transform(th_X)

th_X_dense = np.nan_to_num(
    th_X_dense,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

th_model = IsolationForest(
    n_estimators=250,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

th_model.fit(th_X_dense)

th_anom_scores = -th_model.decision_function(
    th_X_dense
)

print("Thunderbird anomaly model trained.")


# In[45]:


th_rows = []

for e, a in zip(th_entities_ff, th_anom_scores):

    d = float(th_darkness.get(e, 0.0))

    th_rows.append(
        (
            e,
            float(a),
            d,
            0.65 * float(a) + 0.35 * d
        )
    )

th_report = pd.DataFrame(
    th_rows,
    columns=[
        "entity",
        "anomaly_score",
        "darkness_score",
        "combined"
    ]
).sort_values(
    "combined",
    ascending=False
)

th_report.head(10)


# In[46]:


three_dataset_summary = pd.DataFrame([
    {
        "Dataset":"HDFS",
        "Entities":stats["entities"],
        "Templates":stats["num_templates"],
        "Top_DA_EGAD_Score":round(float(final_ranked.iloc[0]["final_score"]),6)
    },
    {
        "Dataset":"BGL",
        "Entities":bgl_stats["entities"],
        "Templates":bgl_stats["num_templates"],
        "Top_DA_EGAD_Score":round(float(bgl_final_ranked.iloc[0]["final_score"]),6)
    },
    {
        "Dataset":"Thunderbird",
        "Entities":th_stats["entities"],
        "Templates":th_stats["num_templates"],
        "Top_DA_EGAD_Score":round(float(th_report.iloc[0]["combined"]),6)
    }
])

three_dataset_summary


# In[47]:


from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.02
)

lof_labels = lof.fit_predict(X_dense)

lof_scores = -lof.negative_outlier_factor_

lof_df = pd.DataFrame({
    "entity": entities_ff,
    "lof_score": lof_scores
})

lof_top10 = set(
    lof_df.sort_values(
        "lof_score",
        ascending=False
    )
    .head(10)
    ["entity"]
)

print("LOF Top10 computed.")


# In[48]:


da_top10 = set(final_ranked.head(10).index)

if_top10 = set(
    final.sort_values(
        "anomaly_score",
        ascending=False
    )
    .head(10)
    .index
)

dark_top10 = set(
    final.sort_values(
        "darkness_score",
        ascending=False
    )
    .head(10)
    .index
)

graph_top10 = set(
    final.sort_values(
        "graph_anomaly",
        ascending=False
    )
    .head(10)
    .index
)

baseline_comparison = pd.DataFrame([
    {
        "Method":"Isolation Forest",
        "Top10_Overlap_with_DA_EGAD":
            len(if_top10.intersection(da_top10))
    },
    {
        "Method":"LOF",
        "Top10_Overlap_with_DA_EGAD":
            len(lof_top10.intersection(da_top10))
    },
    {
        "Method":"Darkness Only",
        "Top10_Overlap_with_DA_EGAD":
            len(dark_top10.intersection(da_top10))
    },
    {
        "Method":"Graph Only",
        "Top10_Overlap_with_DA_EGAD":
            len(graph_top10.intersection(da_top10))
    },
    {
        "Method":"DA-EGAD",
        "Top10_Overlap_with_DA_EGAD":
            10
    }
])

baseline_comparison


# In[ ]:




