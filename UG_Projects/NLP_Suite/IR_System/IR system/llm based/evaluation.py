from util import *
import math

class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision at k for a single query
        """
        precision = 0
        
        if k <= 0:
            print("gone")
            return precision
            
        # Get top k documents
        top_k_docs = query_doc_IDs_ordered[:k]
        # Count relevant documents in top k
        relevant_count = sum(1 for doc_id in top_k_docs if doc_id in true_doc_IDs)
        

        precision = relevant_count / len(top_k_docs)

        return precision

    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of mean precision at k across all queries
        """
        if len(query_ids) == 0:
            return 0
            
        total_precision = 0
        
        for i, query_id in enumerate(query_ids):
            # Get documents for this query
            query_doc_IDs = doc_IDs_ordered[i]
            
            # Get relevant documents for this query
            true_doc_IDs = [int(rel["id"]) for rel in qrels if int(rel["query_num"]) == int(query_id)]
            
            # Calculate precision for this query
            query_precision = self.queryPrecision(query_doc_IDs, query_id, true_doc_IDs, k)
            total_precision += query_precision
            
        return total_precision / len(query_ids)

    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall at k for a single query
        """
        recall = 0
        
        if not true_doc_IDs:  # No relevant documents
            return recall
            
        # Get top k documents
        top_k_docs = query_doc_IDs_ordered[:k]
        
        # Count relevant documents in top k
        relevant_retrieved = sum(1 for doc_id in top_k_docs if int(doc_id) in true_doc_IDs)
        
        # Calculate recall
        recall = relevant_retrieved / len(true_doc_IDs)
        
        return recall

    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of mean recall at k across all queries
        """
        if len(query_ids) == 0:
            return 0
            
        total_recall = 0
        
        for i, query_id in enumerate(query_ids):
            # Get documents for this query
            query_doc_IDs = doc_IDs_ordered[i]
            
            # Get relevant documents for this query
            true_doc_IDs = [int(rel["id"]) for rel in qrels if int(rel["query_num"]) == int(query_id)]
            
            # Calculate recall for this query
            query_recall = self.queryRecall(query_doc_IDs, query_id, true_doc_IDs, k)
            total_recall += query_recall
            
        return total_recall / len(query_ids)

    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of F-score at k for a single query
        """
        precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        
        if precision + recall == 0:  # Avoid division by zero
            return 0
            
        fscore = 2 * precision * recall / (precision + recall)
        return fscore

    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of mean F-score at k across all queries
        """
        if len(query_ids) == 0:
            return 0
            
        total_fscore = 0
        
        for i, query_id in enumerate(query_ids):
            # Get documents for this query
            query_doc_IDs = doc_IDs_ordered[i]
            
            # Get relevant documents for this query
            true_doc_IDs = [int(rel["id"]) for rel in qrels if int(rel["query_num"]) == int(query_id)]
            
            # Calculate F-score for this query
            query_fscore = self.queryFscore(query_doc_IDs, query_id, true_doc_IDs, k)
            total_fscore += query_fscore
            
        return total_fscore / len(query_ids)

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of nDCG at k for a single query
        """
        if not true_doc_IDs:  # No relevant documents
            return 0
            
        # Get top k documents
        top_k_docs = query_doc_IDs_ordered[:k]
        
        # Calculate DCG
        dcg = 0
        for i, doc_id in enumerate(top_k_docs):
            # Binary relevance: 1 if relevant, 0 otherwise
            rel = 1 if doc_id in true_doc_IDs else 0
            if rel > 0:
                # Position i is 0-based, so add 2 for denominator
                dcg += rel / math.log2(i + 2)
        
        # Calculate ideal DCG
        idcg = 0
        for i in range(min(len(true_doc_IDs), k)):
            idcg += 1 / math.log2(i + 2)
        
        if idcg == 0:  # Avoid division by zero
            return 0
            
        ndcg = dcg / idcg
        return ndcg

    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of mean nDCG at k across all queries
        """
        if len(query_ids) == 0:
            return 0
            
        total_ndcg = 0
        
        for i, query_id in enumerate(query_ids):
            # Get documents for this query
            query_doc_IDs = doc_IDs_ordered[i]
            
            # Get relevant documents for this query
            true_doc_IDs = [int(rel["id"]) for rel in qrels if int(rel["query_num"]) == int(query_id)]
            
            # Calculate nDCG for this query
            query_ndcg = self.queryNDCG(query_doc_IDs, query_id, true_doc_IDs, k)
            total_ndcg += query_ndcg
            
        return total_ndcg / len(query_ids)

    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision at k for a single query
        """
        if not true_doc_IDs:  # No relevant documents
            return 0
            
        # Get top k documents
        top_k_docs = query_doc_IDs_ordered[:k]
        
        relevant_seen = 0
        sum_precision = 0
        
        for i, doc_id in enumerate(top_k_docs):
            if doc_id in true_doc_IDs:
                relevant_seen += 1
                # Calculate precision at position i+1
                precision_at_i = relevant_seen / (i + 1)
                sum_precision += precision_at_i
        
        avg_precision = sum_precision / len(true_doc_IDs)
        return avg_precision

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
        """
        Computation of MAP at k across all queries
        """
        if len(query_ids) == 0:
            return 0
            
        total_avg_precision = 0
        
        for i, query_id in enumerate(query_ids):
            # Get documents for this query
            query_doc_IDs = doc_IDs_ordered[i]
            
            # Get relevant documents for this query
            true_doc_IDs = [int(rel["id"]) for rel in q_rels if int(rel["query_num"]) == int(query_id)]
            
            # Calculate average precision for this query
            query_avg_precision = self.queryAveragePrecision(query_doc_IDs, query_id, true_doc_IDs, k)
            total_avg_precision += query_avg_precision
            
        return total_avg_precision / len(query_ids)
