# build_npy.py
import json
import numpy as np

BASE_DIR = "../wikidata"

QID_JSONL = f"{BASE_DIR}/wikidata_qids_06b_ali_classed.jsonl"
LABEL_JSONL = f"{BASE_DIR}/wikidata_labels_06b_ali_classed.jsonl"

QID_NPY = f"{BASE_DIR}/wikidata_qids.npy"
LABEL_NPY = f"{BASE_DIR}/wikidata_labels.npy"

# ---- QID ----
qids = []
with open(QID_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        qids.append(obj["id"])

np.save(QID_NPY, np.array(qids, dtype="<U16"))
print("Saved QIDs:", len(qids))

# ---- label ----
labels = {}
with open(LABEL_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        labels.update(json.loads(line))

np.save(LABEL_NPY, labels)
print("Saved labels:", len(labels))
