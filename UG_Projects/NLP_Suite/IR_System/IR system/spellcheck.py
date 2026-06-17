import json
import re
from itertools import product
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
with open('cran_docs.json', 'r') as f:
    docs = json.load(f)
with open('cran_queries.json', 'r') as f:
    queries = json.load(f)

# Extract all text from documents and queries
texts = []
for doc in docs:
    texts.append(doc['title'])
    texts.append(doc['body'])
for query in queries:
    texts.append(query['query'])

# Tokenize and create vocabulary
vocabulary = set()
for text in texts:
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())  # Filter alphanumeric tokens
    vocabulary.update(tokens)

vocabulary = list(vocabulary)  # Convert to list for indexing

# Generate all possible bigrams
bigrams = [''.join(bigram) for bigram in product('abcdefghijklmnopqrstuvwxyz', repeat=2)]

# Create a vector representation for each word in the vocabulary
word_vectors = {}
for word in vocabulary:
    vector = np.zeros(len(bigrams))
    for i in range(len(word) - 1):
        bigram = word[i:i+2]
        if bigram in bigrams:
            vector[bigrams.index(bigram)] += 1
    word_vectors[word] = vector

def find_top_candidates(typo, word_vectors, bigrams, top_k=5):
    # Convert typo to vector
    typo_vector = np.zeros(len(bigrams))
    for i in range(len(typo) - 1):
        bigram = typo[i:i+2]
        if bigram in bigrams:
            typo_vector[bigrams.index(bigram)] += 1

    # Compute cosine similarity between typo and all words
    similarities = {}
    for word, vector in word_vectors.items():
        similarity = cosine_similarity([typo_vector], [vector])[0][0]
        similarities[word] = similarity

    # Sort by similarity and return top candidates
    sorted_candidates = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [candidate[0] for candidate in sorted_candidates]

# Test for typos
typos = ['boundery', 'transiant', 'aerplain']
for typo in typos:
    candidates = find_top_candidates(typo, word_vectors, bigrams)
    print(f"Top 5 candidates for '{typo}': {candidates}")

def edit_distance(s1, s2, ins_cost=1, del_cost=1, sub_cost=1):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j * ins_cost
            elif j == 0:
                dp[i][j] = i * del_cost
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i][j - 1] + ins_cost,  # Insert
                    dp[i - 1][j] + del_cost,  # Delete
                    dp[i - 1][j - 1] + sub_cost  # Substitute
                )
    return dp[m][n]

# Test Edit Distance
s1 = "kitten"
s2 = "sitting"
print(f"Edit Distance between '{s1}' and '{s2}': {edit_distance(s1, s2)}")

def find_closest_candidate(typo, candidates):
    closest_candidate = None
    min_distance = float('inf')
    for candidate in candidates:
        distance = edit_distance(typo, candidate)
        if distance < min_distance:
            min_distance = distance
            closest_candidate = candidate
    return closest_candidate

# Test for typos
for typo in typos:
    candidates = find_top_candidates(typo, word_vectors, bigrams)
    closest = find_closest_candidate(typo, candidates)
    print(f"Closest candidate for '{typo}': {closest}")

# Test with different costs
typo = 'boundery'
candidates = find_top_candidates(typo, word_vectors, bigrams)

# Case 1: Higher substitution cost
print("Case 1: Higher substitution cost")
for candidate in candidates:
    distance = edit_distance(typo, candidate, sub_cost=2)
    print(f"Edit Distance between '{typo}' and '{candidate}': {distance}")

# Case 2: Higher insertion cost
print("\nCase 2: Higher insertion cost")
for candidate in candidates:
    distance = edit_distance(typo, candidate, ins_cost=2)
    print(f"Edit Distance between '{typo}' and '{candidate}': {distance}")

# Case 3: Higher deletion cost
print("\nCase 3: Higher deletion cost")
for candidate in candidates:
    distance = edit_distance(typo, candidate, del_cost=2)
    print(f"Edit Distance between '{typo}' and '{candidate}': {distance}")

# Conditions for Edit Distance to be a valid distance measure:
# 1. Non-negativity: Distance is always >= 0.
# 2. Identity: Distance is 0 if and only if the strings are identical.
# 3. Symmetry: Distance from s1 to s2 is the same as from s2 to s1.
# 4. Triangle Inequality: Distance(s1, s3) <= Distance(s1, s2) + Distance(s2, s3).

# Example to demonstrate symmetry
s1, s2 = "kitten", "sitting"
print(f"Distance(s1, s2): {edit_distance(s1, s2)}")
print(f"Distance(s2, s1): {edit_distance(s2, s1)}")

# Example to demonstrate triangle inequality
s3 = "kitchen"
print(f"Distance(s1, s3): {edit_distance(s1, s3)}")
print(f"Distance(s1, s2) + Distance(s2, s3): {edit_distance(s1, s2) + edit_distance(s2, s3)}")