#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip -q install pandas numpy scikit-learn tqdm drain3 joblib')


# In[2]:


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


# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


seq_map, stats = build_sequences(path, max_lines=200000)

print("✅ Built sequences")
print({k:v for k,v in stats.items() if k!="template_counts"})


# In[8]:


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


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[18]:


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


# In[19]:


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


# In[20]:


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


# In[21]:


final = report.set_index("entity").join(gf_df, how="left").fillna(0)

final["final_score"] = (
    0.4 * final["anomaly_score"] +
    0.3 * final["darkness_score"] +
    0.3 * final["graph_anomaly"]
)

final_ranked = final.sort_values("final_score", ascending=False)
final_ranked.head(10)


# In[ ]:




