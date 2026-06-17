from util import *

class InflectionReduction:

    def reduce(self, text):
        """
        Stemming/Lemmatization

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of
            stemmed/lemmatized tokens representing a sentence
        """

        reducedText = []

        # nltk lib for stemmer
        porter = PorterStemmer()
        
        for sentence in text:
            reduced_sentence = []
            for word in sentence:
                reduced_word = porter.stem(word)
                reduced_sentence.append(reduced_word)
            reducedText.append(reduced_sentence)
        
        return reducedText

# text = [["running", "jumps", "easily"], ["flies", "better", "studying"]]
# text2 = [['running', 'jumps', 'easily'], ['flies', 'better', 'studying']]
# reducer = InflectionReduction()
# print(reducer.reduce(text2))