from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import re

'''
Preprocessing pipeline order:

- Descapitalize text
- Tokenize words
- Remove special characters
- Remove stopwords
- Lemmatize text
'''

lemma_model = spacy.load("pt_core_news_sm")

def decapitalize_text(text):
    try:
        if isinstance(text, str):
            return text.lower()
        elif isinstance(text, list):
            return [item.lower() for item in text]
        else:
            raise ValueError("Input must be a string or a list of strings")
    
    except Exception as e:
        raise ValueError(f"Error in decapitalize_text: {e}")

def tokenize_text(text):
    try:
        if isinstance(text, str):
            return word_tokenize(text)
        elif isinstance(text, list):
            return [word_tokenize(item) for item in text]
        else:
            raise ValueError("Input must be a string or a list of strings")
        
    except Exception as e:
        raise ValueError(f"Error in tokenize_text: {e}")

def remove_special_characters(text):
    try:
        if isinstance(text, list) and len(text) > 0:
            if isinstance(text[0], str):
                return [token for token in text if re.match(r'^[a-zA-ZÀ-ÿ]+$', token)]
            elif isinstance(text[0], list):
                return [[token for token in sentence if re.match(r'^[a-zA-ZÀ-ÿ]+$', token)] for sentence in text]
            else:
                raise ValueError("Input must be a list of strings or a list of lists of strings")
        else:
            raise ValueError("Input must be a non-empty list")
            
    except Exception as e:
        raise ValueError(f"Error in remove_special_characters: {e}")
    
def remove_stopwords(text):
    try:
        stopwords_pt = set(stopwords.words('portuguese'))
        # words_to_maintain = {}

        if isinstance(text, list) and len(text) > 0:
            if isinstance(text[0], str):
                return [token for token in text if token not in stopwords_pt]
            elif isinstance(text[0], list):
                return [[token for token in sentence if token not in stopwords_pt] for sentence in text]
            else:
                raise ValueError("Input must be a list of strings or a list of lists of strings")
        else:
            raise ValueError("Input must be a non-empty list")
        
    except Exception as e:
        raise ValueError(f"Error in remove_stopwords: {e}")

def lemmatize_text(text):
    try:
        if isinstance(text, list) and len(text) > 0:
            if isinstance(text[0], str):
                doc = lemma_model(" ".join(text))
                return [token.lemma_ for token in doc]
            elif isinstance(text[0], list):
                result = []
                for sentence in text:
                    doc = lemma_model(" ".join(sentence))
                    result.append([token.lemma_ for token in doc])
                return result
            else:
                raise ValueError("Input must be a list of strings or a list of lists of strings")
        else:
            raise ValueError("Input must be a non-empty list")
            
    except Exception as e:
        raise ValueError(f"Error in lemmatize_text: {e}")

def preprocess_text(text):
    try:
        text = decapitalize_text(text)
        text = tokenize_text(text)
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        text = lemmatize_text(text)
        
        return text
    
    except Exception as e:
        raise ValueError(f"Error in preprocess_text: {e}")

