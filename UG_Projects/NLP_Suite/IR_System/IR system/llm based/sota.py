import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

# Load data
with open("cran_docs.json") as f:
    docs = json.load(f)
with open("cran_queries.json") as f:
    queries = json.load(f)
with open("cran_qrels.json") as f:
    qrels = json.load(f)

# Convert to DataFrames
docs_df = pd.DataFrame(docs)
queries_df = pd.DataFrame(queries)
qrels_df = pd.DataFrame(qrels)

doc_ids = list(map(str, docs_df["id"].tolist()))
query_ids = list(map(str, queries_df["query number"].tolist()))
doc_texts = docs_df["body"].tolist()
query_texts = queries_df["query"].tolist()

# Ground truth mapping
qrels_map = qrels_df.groupby("query_num")["id"].apply(lambda x: set(map(str, x))).to_dict()

# Sentence-BERT embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(doc_texts, convert_to_tensor=True)
query_embeddings = model.encode(query_texts, convert_to_tensor=True)

# Cosine similarity scores
cosine_scores = util.cos_sim(query_embeddings, doc_embeddings).cpu().numpy()

# Evaluation over multiple k values
def evaluate_all_metrics(query_ids, doc_ids, scores, k_values):
    precision_list = []
    recall_list = []
    f1_list = []
    map_list = []
    ndcg_list = []

    for k in k_values:
        precision_k = []
        recall_k = []
        f1_k = []
        map_k = []
        ndcg_k = []

        for i, qid in enumerate(query_ids):
            relevant = qrels_map.get(str(qid), set())
            if not relevant:
                continue

            ranked_indices = np.argsort(scores[i])[::-1][:k]
            top_docs = [doc_ids[j] for j in ranked_indices]

            true_positives = sum(1 for d in top_docs if d in relevant)
            precision = true_positives / k
            recall = true_positives / len(relevant)
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

            # Binary relevance array for MAP and nDCG
            rel_binary = [1 if doc_ids[j] in relevant else 0 for j in np.argsort(scores[i])[::-1][:k]]
            rel_full = [1 if doc_ids[j] in relevant else 0 for j in np.argsort(scores[i])[::-1]]

            # MAP@k (truncated AP)
            if any(rel_binary):
                num_hits = 0
                sum_precisions = 0.0
                for rank, doc_idx in enumerate(ranked_indices):
                    if doc_ids[doc_idx] in relevant:
                        num_hits += 1
                        sum_precisions += num_hits / (rank + 1)
                ap = sum_precisions / len(relevant) if relevant else 0.0
            else:
                ap = 0.0

            # nDCG@k
            rel_binary = [1 if doc_ids[j] in relevant else 0 for j in ranked_indices]
            dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rel_binary))
            ideal_rel = [1] * min(len(relevant), k)
            idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_rel))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            precision_k.append(precision)
            recall_k.append(recall)
            f1_k.append(f1)
            map_k.append(ap)
            ndcg_k.append(ndcg)

        precision_list.append(np.mean(precision_k))
        recall_list.append(np.mean(recall_k))
        f1_list.append(np.mean(f1_k))
        map_list.append(np.mean(map_k))
        ndcg_list.append(np.mean(ndcg_k))

    return precision_list, recall_list, f1_list, map_list, ndcg_list

# Evaluation for k=1 to 10
k_vals = list(range(1, 11))
precision_vals, recall_vals, f1_vals, map_vals, ndcg_vals = evaluate_all_metrics(query_ids, doc_ids, cosine_scores, k_vals)
# Print specific metrics at k=10 (index 9)
f1_avg = np.mean(f1_vals)
# Print final required metrics
print(f"nDCG@10: {ndcg_vals[9]:.4f}")
print(f"MAP@10: {map_vals[9]:.4f}")
print(f"F1 (average over k=1 to 10): {f1_avg:.4f}")


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_vals, precision_vals, marker='o', label="Precision@k")
plt.plot(k_vals, recall_vals, marker='s', label="Recall@k")
plt.plot(k_vals, f1_vals, marker='^', label="F1-score@k")
plt.plot(k_vals, map_vals, marker='d', label="MAP@k")
plt.plot(k_vals, ndcg_vals, marker='x', label="nDCG@k")

plt.xlabel("k")
plt.ylabel("Metric Score")
plt.title("Evaluation Metrics vs k (Sentence-BERT)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(k_vals)
plt.tight_layout()

# Save the plot
plt.savefig("metrics_vs_k.png")
plt.show()
