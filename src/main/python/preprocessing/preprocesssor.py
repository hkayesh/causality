import string
import re
import os
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk.stem import PorterStemmer

from utils.utilities import Utilities


class Preprocessor:
    def __init__(self, params=list()):
        self.remove_urls = True if 'remove_urls' in params else False
        self.remove_mentions = True if 'remove_mentions' in params else False
        self.remove_hashtags = True if 'remove_hashtags' in params else False
        self.normalize = True if 'normalize' in params else False
        self.remove_stopwords = True if 'remove_stopwords' in params else False
        self.remove_punct = True if 'remove_punctuation' in params else False
        self.lower = True if 'lower' in params else False
        self.lemmatize = True if 'lemmatize' in params else False
        self.stemming = True if 'stemming' in params else False
        self.remove_non_letters = True if 'remove_non_letters' in params else False
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.utilities = Utilities()

    def preprocess(self, document):
        """
        Run the preprocessing operations on the input string and returns processed string

        :param document: a string to be preprocessed. 
        :return string: processed string 
        """
        processed_doc = document
        processed_doc = processed_doc.lower() if self.lower else processed_doc
        processed_doc = self.remove_urls_from_text(processed_doc, '') if self.remove_urls else processed_doc
        processed_doc = self.remove_mentions_from_text(processed_doc, '') if self.remove_mentions else processed_doc
        processed_doc = self.remove_hashtags_from_text(processed_doc, '') if self.remove_hashtags else processed_doc
        processed_doc = self.remove_char_repeatation_from_text(processed_doc) if self.normalize else processed_doc
        processed_doc = self.remove_stopwords_from_str(processed_doc) if self.remove_stopwords else processed_doc
        processed_doc = self.remove_punctuation_from_str(processed_doc) if self.remove_punct else processed_doc
        processed_doc = self.remove_non_letter_chars(processed_doc) if self.remove_non_letters else processed_doc
        processed_doc = self.lemmatize_doc(processed_doc) if self.lemmatize else processed_doc
        processed_doc = self.stem_doc(processed_doc) if self.stemming else processed_doc

        return processed_doc.strip()

    def remove_urls_from_text(self, text, replace_with='<url>'):
        clean_text = re.sub(r'(?i)\b((?:https?://[ ]?|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/|www\d{0,3}[.][ ]?[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’]))', replace_with, text.rstrip())

        return clean_text

    def remove_mentions_from_text(self, text, replace_with='<mention>'):

        processed_text = re.sub(r'(?<!\w)(@\w+)\b', replace_with, text.rstrip())

        return processed_text

    def remove_hashtags_from_text(self, text, replace_with='<hashtag>'):
        processed_text = re.sub(r'(?<!\w)(#\w+)\b', replace_with, text.rstrip())

        return processed_text

    def remove_char_repeatation_from_text(self, text):

        processed_text = re.sub(r'(. ?)\1+', r'\1', text)

        return processed_text


    def remove_punctuation_from_str(self, document):
        """
        Remove punctuations from a given string 

        :param document: a string
        :return string: the string without punctuations  
        """
        table = str.maketrans({key: None for key in string.punctuation})
        document = document.translate(table)

        return document

    def remove_stopwords_from_str(self, document):
        """
        Remove stopwords from a given string

        :param document: a string
        :return string: the string without stop words 
        """
        stop_words = set(stopwords.words('english'))
        regex_pat = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, stop_words)), re.IGNORECASE)
        clean_doc = regex_pat.sub('', document)
        clean_doc = re.sub(' +', ' ', clean_doc).strip()  # remove redundant whitespace

        return clean_doc

    def lemmatize_doc(self, document):
        """
        Apply lemmatization on each word of a string

        :param document: a string
        :return string: the string with lemmatized words 
        """

        processed_doc = ""
        for sent in sent_tokenize(document):
            lemmatized_sent = ""
            for token in wordpunct_tokenize(sent):
                lemmatized_token = self.lemmatizer.lemmatize(token)
                lemmatized_sent += lemmatized_token + ' '
            processed_doc += lemmatized_sent.strip()

        return processed_doc

    def stem_doc(self, document):
        """
        Apply stemming on each word of a string

        :param document: a string
        :return string: the string with stemmed words
        """
        processed_doc = ""
        for sent in sent_tokenize(document):
            lemmatized_sent = ""
            for token in wordpunct_tokenize(sent):
                lemmatized_token = self.stemmer.stem(token)
                lemmatized_sent += lemmatized_token + ' '
            processed_doc += lemmatized_sent.strip()

        return processed_doc

    def remove_non_letter_chars(self, document):
        """
        Remove anything but letters and spaces

        :param document: a string
        :return string: a string that contains only letters and spaces
        """
        regex = re.compile('[^a-zA-Z ]')
        processed_doc = regex.sub('', document)
        processed_doc = re.sub(' +', ' ', processed_doc)  # remove multiple spaces

        return processed_doc
