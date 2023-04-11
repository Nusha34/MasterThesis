from bs4 import BeautifulSoup
import spacy
import os
import glob
import logging
import nltk
from nltk.stem import PorterStemmer

class PubMedSentences: 
    def __init__(self, path_to_xmls):
        logging.info('Parsing pubmed corpus')
        self.path_to_xmls=path_to_xmls
        self.nlp = spacy.load("en_core_web_sm")
        self.stemmer = PorterStemmer()

    def preprocess_sentences(self, xml_file):
        sentences_abstract = []
        soup = BeautifulSoup(open(xml_file), "lxml-xml")
        text = soup.find_all("AbstractText")
        for el in text:
            text_new = el.get_text()
            tok_sentences = self.nlp(text_new)
            for sent in tok_sentences.sents:
                 sentences_abstract.append(
                    [self.stemmer.stem(token.text.lower()) for token in sent if not token.is_stop and not token.is_punct])
        return sentences_abstract


    def __iter__(self):
        for file_name in glob.glob(os.path.join(self.path_to_xmls, "*.xml")):
            for sentence in self.preprocess_sentences(file_name):
                yield sentence
