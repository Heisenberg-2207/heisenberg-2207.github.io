import itertools
import gensim.downloader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from gensim.models import Word2Vec

class Word2VecIndex:
    def __init__(self, mode="custom"):
        """
        :param mode: "custom" for training Word2Vec on your corpus,
                     "pretrained" for using Google's pretrained model
        """
        self.index = None
        self.docIDs = None
        self.model = None
        self.mode = mode.lower()

    def buildIndex(self, docs, docIDs):
        """
        Build the document embedding index using average word vectors.
        """
        self.docIDs = docIDs
        index = []

        flat_sentences = list(itertools.chain.from_iterable(docs))

        # Load or train Word2Vec
        if self.mode == "custom":
            self.model = Word2Vec(flat_sentences, min_count=1)
            vocab = self.model.wv.index_to_key
        else:
            self.model = gensim.downloader.load('word2vec-google-news-300')
            vocab = self.model.index_to_key

        if (not os.path.exists('word2vec.npy') or self.mode == "custom" or np.load('word2vec.npy').shape[1] != self.model.vector_size):
            for doc in tqdm(docs, desc="Building Word2Vec index"):
                count = 0
                doc_emb = np.zeros(self.model.vector_size)
                for sentence in doc:
                    for word in sentence:
                        if word != '.' and word in vocab:
                            vec = self.model.wv[word] if self.mode == "custom" else self.model[word]
                            doc_emb += vec
                            count += 1
                doc_emb = doc_emb / count if count > 0 else doc_emb
                index.append(doc_emb)
            np.save('word2vec.npy', index)
        else:
            index = np.load('word2vec.npy')

        self.index = np.vstack(index)

    def rank(self, queries):
        """
        Rank documents for each query based on cosine similarity.
        :param queries: list of queries (list of list of tokens)
        """
        doc_IDs_ordered = []
        query_embs = []
        vocab = self.model.wv.index_to_key if self.mode == "custom" else self.model.index_to_key

        for query in tqdm(queries, desc="Ranking queries"):
            count = 0
            q_emb = np.zeros(self.model.vector_size)
            for sentence in query:
                for word in sentence:
                    if word != '.' and word in vocab:
                        vec = self.model.wv[word] if self.mode == "custom" else self.model[word]
                        q_emb += vec
                        count += 1
            q_emb = q_emb / count if count > 0 else q_emb
            query_embs.append(q_emb)

        cos_sim = cosine_similarity(query_embs, self.index)
        for cos_similarity_vector in cos_sim:
            top_doc_indexes = cos_similarity_vector.argsort()[::-1]
            ranked_doc_ids = [self.docIDs[i] for i in top_doc_indexes]
            doc_IDs_ordered.append(ranked_doc_ids)

        return doc_IDs_ordered
