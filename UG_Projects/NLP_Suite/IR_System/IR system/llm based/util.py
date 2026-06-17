# Add your import statements here
import re
import string
import nltk
import math
import json
import math
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Add any utility functions here