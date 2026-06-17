from util import *
# nltk.download('stopwords')

class StopwordRemoval:
    def fromList(self, text):
        """
        Remove stopwords from tokenized sentences using a curated stopword list.

        Parameters
        ----------
        text : list
            A list of lists where each sub-list is a sequence of tokens representing a sentence.

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens with stopwords removed.
        """
        
        stop_words = set(stopwords.words('english'))
        # print(stop_words)

        stopwordRemovedText = []
        
        # Iterate over each tokenized sentence.
        for sentence in text:
            
            filtered_sentence = []
            
            # Iterate over each token in the sentence.
            for token in sentence:
                
                # Check if the lowercased token is not a stopword.
                if token.lower() not in stop_words:
                    filtered_sentence.append(token)
                    
            # Append the filtered sentence to the result list.
            stopwordRemovedText.append(filtered_sentence)
        
        return stopwordRemovedText
    
    def TFIDF(self, text):
        """
        Remove tokens with low TF-IDF score from tokenized sentences using a bottom-up approach.

        Parameters
        ----------
        text : list
            A list of lists where each sub-list is a sequence of tokens representing a sentence.

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens with low TF-IDF tokens removed.
        """
        # First, compute document frequency (DF) for each token (case-insensitive).
        df = {}
        D = len(text)
        for sentence in text:
            seen_tokens = set()
            for token in sentence:
                token_lower = str(token).lower()
                if token_lower not in seen_tokens:
                    if token_lower in df:
                        df[token_lower] += 1
                    else:
                        df[token_lower] = 1
                    seen_tokens.add(token_lower)

        # Compute the IDF for each token.
        # We use the logarithm to reduce the impact of tokens that appear in many documents.
        idf = {}
        for token in df:
            idf[token] = math.log(D / df[token])

        # Define a threshold for TF-IDF. Tokens with a score below this will be removed.
        threshold = 0.05

        stopwordRemovedText = []

        # Process each sentence individually.
        for sentence in text:

            # Calculate term frequency (TF) for tokens in this sentence.
            tf = {}
            total_tokens = len(sentence)
            for token in sentence:
                token_lower = token.lower()
                if token_lower in tf:
                    tf[token_lower] += 1
                else:
                    tf[token_lower] = 1
            # Normalize TF by dividing counts by the total number of tokens.
            for token_lower in tf:
                tf[token_lower] = tf[token_lower] / total_tokens

            # Build a filtered sentence based on the TF-IDF score.
            filtered_sentence = []
            for token in sentence:
                token_lower = token.lower()
                token_tf = tf.get(token_lower, 0)
                token_idf = idf.get(token_lower, 0)
                tfidf = token_tf * token_idf
                if tfidf >= threshold:
                    filtered_sentence.append(token)
            stopwordRemovedText.append(filtered_sentence)

        return stopwordRemovedText

# print(StopwordRemoval().fromList([['the', 'quick', 'brown', 'fox'], ['jumped', 'over', 'the', 'lazy', 'dog']]))
# print(StopwordRemoval().fromList([['i', 'love', 'nlp'], ['nlp', 'is', 'fun']]))
# print(StopwordRemoval().TFIDF([['the', 'quick', 'brown', 'fox'], ['jumped', 'over', 'the', 'lazy', 'dog']]))
# print(StopwordRemoval().TFIDF([['i', 'love', 'nlp'], ['nlp', 'is', 'fun']]))

# StopwordRemoval().fromList([['the', 'quick', 'brown', 'fox'], ['jumped', 'over', 'the', 'lazy', 'dog']])



	