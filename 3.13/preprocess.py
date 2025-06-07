import re
import unicodedata
from nltk.stem.snowball import SnowballStemmer


stemmer = SnowballStemmer("portuguese")

def preprocess_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return " ".join(stemmer.stem(token) for token in tokens)

