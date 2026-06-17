# Add your import statements here
# Text processing
import re
import string
import math
import json
from collections import defaultdict

# NLP libraries
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize, TreebankWordTokenizer, PunktSentenceTokenizer, sent_tokenize
from nltk.stem import PorterStemmer

# Numeric computing
import numpy as np

# Sklearn tools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Gensim for word2vec
from gensim.models import Word2Vec

# PyTorch for autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# Add any utility functions here