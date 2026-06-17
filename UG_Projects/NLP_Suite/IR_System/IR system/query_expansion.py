from util import *
import math

def expand_query(query, max_synonyms=2):
    """
    Expands the query by adding synonyms from WordNet, limiting to a specified number.
    """
    expanded_query = set(query.split())  # Start with original query terms
    
    for word in query.split():
        # Get synonyms for each word using WordNet
        synonyms = wordnet.synsets(word)
        
        # Add only the top 'max_synonyms' synonyms to the expanded query
        count = 0
        for synonym in synonyms:
            for lemma in synonym.lemmas():
                expanded_query.add(lemma.name())  # Add synonym to the set
                count += 1
                if count >= max_synonyms:  # Stop after adding 'max_synonyms' synonyms
                    break
            if count >= max_synonyms:
                break
    
    return " ".join(expanded_query)

# Example usage
query = "car repair"
expanded_query = expand_query(query, max_synonyms=2)  # Limit to 3 synonyms
print("Original Query: ", query)
print("Expanded Query: ", expanded_query)
