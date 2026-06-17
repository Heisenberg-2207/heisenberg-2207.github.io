import json
import nltk
import string
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from evaluation import Evaluation
from nltk.tokenize import sent_tokenize, word_tokenize
from autocorrect import Speller
import os
import argparse

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

class CrossEncoderInformationRetriever:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.docs = []
        self.doc_ids = []

    def buildIndex(self, docs, doc_ids):
        self.docs = docs
        self.doc_ids = doc_ids

    def rank(self, queries):
        results = []
        for query in queries:
            pairs = [(query, doc) for doc in self.docs]
            scores = self.model.predict(pairs)
            ranked_indices = np.argsort(-np.array(scores)).tolist()
            ranked_doc_ids = [self.doc_ids[i] for i in ranked_indices]
            results.append(ranked_doc_ids)
        return results


class LLMIRSystem:
    def __init__(self, args):
        self.args = args
        self.evaluator = Evaluation()
        self.informationRetriever = CrossEncoderInformationRetriever()
        self.spell = Speller(lang='en')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def preprocessText(self, texts):
        processed_texts = []
        for text in texts:
            sentences = sent_tokenize(text)
            tokens = []
            for sent in sentences:
                words = word_tokenize(sent)
                words = [w.lower() for w in words if w.isalpha()]
                words = [self.spell(w) for w in words]
                words = [w for w in words if w not in self.stop_words]
                words = [self.lemmatizer.lemmatize(w) for w in words]
                tokens.extend(words)
            processed_texts.append(" ".join(tokens))
        return processed_texts

    def preprocessQueries(self, queries):
        return self.preprocessText(queries)

    def preprocessDocs(self, docs):
        return self.preprocessText(docs)


if __name__ == "__main__":
    # Step 1: Create args
    args = argparse.Namespace(
        dataset="../cranfield/",       # Replace with actual dataset path
        out_folder="../cranfield/"     # Replace with desired output path
    )

    os.makedirs(args.out_folder, exist_ok=True)
    system = LLMIRSystem(args)

    # Step 2: Load and preprocess queries
    with open(args.dataset + "cran_queries.json", "r") as f:
        queries_json = json.load(f)
    query_ids = [item["query number"] for item in queries_json]
    queries = [item["query"] for item in queries_json]
    processed_queries = system.preprocessQueries(queries)
    print("Sample processed query:", processed_queries[0])

    # Step 3: Load and preprocess documents
    with open(args.dataset + "cran_docs.json", "r") as f:
        docs_json = json.load(f)
    doc_ids = [item["id"] for item in docs_json]
    docs = [item["body"] for item in docs_json]
    processed_docs = system.preprocessDocs(docs)
    print("Sample processed doc:", processed_docs[0][:300])

    # Step 4: Build document index
    system.informationRetriever.buildIndex(processed_docs, doc_ids)
    print("Index built on", len(doc_ids), "documents.")

    # Step 5: Rank documents
    doc_IDs_ordered = system.informationRetriever.rank(processed_queries)
    print("Top 5 docs for first query:", doc_IDs_ordered[0][:5])

    # Step 6: Evaluate
    with open(args.dataset + "cran_qrels.json", "r") as f:
        qrels = json.load(f)

    precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
    for k in range(1, 11):
        p = system.evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
        r = system.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
        f = system.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
        m = system.evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
        n = system.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)

        precisions.append(p)
        recalls.append(r)
        fscores.append(f)
        MAPs.append(m)
        nDCGs.append(n)
        print(f"@{k}: Precision={p:.4f}, Recall={r:.4f}, F-score={f:.4f}, MAP={m:.4f}, nDCG={n:.4f}")

    # Step 7: Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), precisions, label="Precision")
    plt.plot(range(1, 11), recalls, label="Recall")
    plt.plot(range(1, 11), fscores, label="F-Score")
    plt.plot(range(1, 11), MAPs, label="MAP")
    plt.plot(range(1, 11), nDCGs, label="nDCG")
    plt.title("Evaluation Metrics - Cranfield Dataset")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.out_folder + "eval_plot.png")
    print("Evaluation plot saved to", args.out_folder + "eval_plot.png")
