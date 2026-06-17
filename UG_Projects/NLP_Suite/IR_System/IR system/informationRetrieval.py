from util import *
from word2vec import Word2VecIndex
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class InformationRetrieval():
    def __init__(self, mode='tfidf', use_wsd=True, wsd_weight=0.6, hidden_dim=400):
        assert mode in ['tfidf', 'lsa', 'autoencoder', 'word2vec'], "Invalid mode."
        self.mode = mode
        self.use_wsd = use_wsd
        self.wsd_weight = wsd_weight
        self.hidden_dim = hidden_dim
        self.index = {}
        self.docIDs = []
        self.docs = []
        self.vectorizer = None
        self.doc_vectors = None
        self.autoencoder = None
        self.svd = None
        self.word_importance = {}
        self.important_terms = set()

    def buildIndex(self, docs, docIDs):
        if self.mode == 'word2vec':
            self.retriever = Word2VecIndex(docs, docIDs, self.hidden_dim)
            self.retriever.buildIndex(docs, docIDs)
            return

        self.docIDs = docIDs
        self.original_docs = docs

        processed_docs = [" ".join([" ".join(sent) for sent in doc]) for doc in docs]
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(processed_docs).toarray()

        # Compute word importance
        tfidf_avg = np.mean(tfidf_matrix, axis=0)
        features = self.vectorizer.get_feature_names_out()
        for i, term in enumerate(features):
            self.word_importance[term] = tfidf_avg[i]
            if tfidf_avg[i] > np.mean(tfidf_avg) + 0.5 * np.std(tfidf_avg):
                self.important_terms.add(term)

        # Apply vectorization
        if self.mode == 'tfidf':
            self.doc_vectors = tfidf_matrix
        elif self.mode == 'lsa':
            self.svd = TruncatedSVD(n_components=self.hidden_dim)
            self.doc_vectors = self.svd.fit_transform(tfidf_matrix)
        elif self.mode == 'autoencoder':
            input_dim = tfidf_matrix.shape[1]
            self.autoencoder = Autoencoder(input_dim, self.hidden_dim)
            optimizer = optim.Adam(self.autoencoder.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            data = torch.tensor(tfidf_matrix, dtype=torch.float32)
            self.autoencoder.train()
            for epoch in range(200):  # You can adjust epochs
                optimizer.zero_grad()
                output, _ = self.autoencoder(data)
                loss = criterion(output, data)
                loss.backward()
                optimizer.step()
            self.autoencoder.eval()
            with torch.no_grad():
                _, embeddings = self.autoencoder(data)
            self.doc_vectors = embeddings.numpy()
        elif self.mode == 'word2vec':
            tokenized_docs = [[word for sent in doc for word in sent] for doc in docs]
            self.word2vec_model = Word2Vec(sentences=tokenized_docs, vector_size=self.hidden_dim,
                                           window=5, min_count=1, workers=4)
            def get_doc_vector(tokens):
                vectors = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
                return np.mean(vectors, axis=0) if vectors else np.zeros(self.hidden_dim)
            self.doc_vectors = np.array([get_doc_vector(tokens) for tokens in tokenized_docs])

        # Build inverted index (based on TF-IDF)
        self.index = {}
        for idx, term in enumerate(features):
            self.index[term] = []
            for doc_idx in range(len(docs)):
                if tfidf_matrix[doc_idx][idx] > 0:
                    self.index[term].append(docIDs[doc_idx])

    def apply_selective_wsd(self, query, query_mode=False):
        # Placeholder function for WSD (implement if necessary)
        return query

    def rank(self, queries):
        if self.mode == 'word2vec':
            return self.retriever.rank(queries)

        doc_IDs_ordered = []
        for query in queries:
            flat_query = " ".join([" ".join(sent) for sent in query])
            query_vec = self.vectorizer.transform([flat_query]).toarray()

            if self.mode == 'autoencoder':
                with torch.no_grad():
                    _, query_vec = self.autoencoder(torch.tensor(query_vec, dtype=torch.float32))
                    query_vec = query_vec.numpy()
            elif self.mode == 'lsa':
                query_vec = self.svd.transform(query_vec)
            elif self.mode == 'word2vec':
                tokens = [word for sent in query for word in sent]
                vectors = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
                if vectors:
                    query_vec = np.mean(vectors, axis=0).reshape(1, -1)
                else:
                    query_vec = np.zeros((1, self.hidden_dim))

            if self.use_wsd and self.mode in ['tfidf', 'lsa', 'autoencoder']:
                wsd_query = self.apply_selective_wsd(query, query_mode=True)
                flat_wsd = " ".join([" ".join(sent) for sent in wsd_query])
                wsd_vec = self.vectorizer.transform([flat_wsd]).toarray()

                if self.mode == 'autoencoder':
                    with torch.no_grad():
                        _, wsd_vec = self.autoencoder(torch.tensor(wsd_vec, dtype=torch.float32))
                        wsd_vec = wsd_vec.numpy()
                elif self.mode == 'lsa':
                    wsd_vec = self.svd.transform(wsd_vec)

                combined = (1 - self.wsd_weight) * query_vec + self.wsd_weight * wsd_vec
            else:
                combined = query_vec

            sims = cosine_similarity(combined, self.doc_vectors)[0]
            ranked_indices = np.argsort(-sims)
            doc_IDs_ordered.append([self.docIDs[i] for i in ranked_indices])

        return doc_IDs_ordered
