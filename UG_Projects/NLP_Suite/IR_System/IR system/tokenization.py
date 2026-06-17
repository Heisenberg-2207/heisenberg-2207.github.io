import re
import string
from nltk.tokenize import TreebankWordTokenizer

class Tokenization():

    def clean_tokens(self, tokens):
        """
        Clean tokens by stripping leading/trailing spaces and punctuation.

        Parameters
        ----------
        tokens : list
            A list of tokens.

        Returns
        -------
        list
            A cleaned list of tokens.
        """
        cleaned_tokens = []
        
        for token in tokens:
            # Remove leading/trailing spaces and punctuation.
            cleaned_token = token.strip()  # Remove spaces around the token.
            cleaned_token = cleaned_token.strip(string.punctuation)  # Remove punctuation.
            if cleaned_token:
                cleaned_tokens.append(cleaned_token)
        
        return cleaned_tokens

    def naive(self, text):
        """
        Tokenization using White Space Tokenization with Trailing Punctuation Removal

        Parameters
        ----------
        text : list
            A list of strings where each string is a single sentence.

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens.
        """
        tokenizedText = []
        
        for sentence in text:
            # Strip leading and trailing whitespace.
            sentence = sentence.strip()

            # Tokenize by splitting the sentence on whitespace.
            tokens = sentence.split()

            # Clean each token by removing unwanted punctuation and spaces.
            cleaned_tokens = self.clean_tokens(tokens)

            tokenizedText.append(cleaned_tokens)
        
        return tokenizedText

    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer after cleaning tokens.

        Parameters
        ----------
        text : list
            A list of strings where each string is a single sentence.

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens.
        """
        tokenizedText = []
        tokenizer = TreebankWordTokenizer()

        for sentence in text:
            # Tokenize using the TreebankWordTokenizer.
            tokens = tokenizer.tokenize(sentence)

            # Clean each token by removing unwanted punctuation and spaces.
            cleaned_tokens = self.clean_tokens(tokens)

            tokenizedText.append(cleaned_tokens)

        return tokenizedText


# Testing the tokenization with punctuation removal
# print(Tokenization().naive(["This is a sentence.", "This is Dr. Don't can't ... abc.", "I'm so happy !"])) 
# print(Tokenization().pennTreeBank(["This is a sentence .", "This is Dr. Don't can't ... abc.", "Hello my friend!"]))
