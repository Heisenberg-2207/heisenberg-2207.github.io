from util import *

class SentenceSegmentation:
    def naive(self, text):
        """
        Sentence Segmentation using a Naive Approach

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """
        # Define a set of common abbreviations to prevent mis-splitting.
        abbreviations = {"Dr.", "Mr.", "Mrs.", "Ms.", "Jr.", "Sr.", "Inc.", "Ltd.", "a.m.", "p.m."}
        
        # Regex pattern: split the text at punctuation if it is followed by whitespace and an uppercase letter.
        # The capturing group ([.!?]) ensures that the delimiter is retained.
        pattern = r'([.!?])\s*(?=[A-Z])'
        
        # Split the text using the regex pattern.
        parts = re.split(pattern, text)
        
        # Combine the split fragments back into full sentences.
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            # Each sentence is composed of a fragment and its following punctuation.
            sentence = parts[i] + parts[i+1]
            sentences.append(sentence.strip())
        
        # If there is a remaining fragment (without a trailing punctuation), append it as a sentence.
        if len(parts) % 2 == 1:
            sentences.append(parts[-1].strip())
        
        # Merge sentences that were erroneously split because the previous sentence ended with an abbreviation.
        final_sentences = []
        for i in range(len(sentences)):
            if i > 0:
                # Split the previous sentence into words.
                previous_words = sentences[i - 1].split()
                # Check if the last word in the previous sentence is a known abbreviation.
                if previous_words and previous_words[-1] in abbreviations:
                    # Merge the current sentence with the previous one.
                    final_sentences[-1] = final_sentences[-1] + " " + sentences[i]
                else:
                    final_sentences.append(sentences[i])
            else:
                final_sentences.append(sentences[i])
        
        return final_sentences

    def punkt(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters:
            text (str): A string containing a block of sentences.

        Returns:
            list: A list of strings where each string is a single sentence.
        """
        # Create a PunktSentenceTokenizer object.
        tokenizer = PunktSentenceTokenizer()
        
        # Tokenize the text after stripping leading/trailing whitespace.
        segmentedText = tokenizer.tokenize(text.strip())
        return segmentedText


# print(SentenceSegmentation().naive("This is a sentence . This is another sentence. Hi Dr. Aniket. How are you?"))
# print(SentenceSegmentation().punkt("This is a sentence . This is another sentence."))
# print(SentenceSegmentation().punkt("Dr. Smith arrived at 10 a.m. and said, 'Hello!'"))