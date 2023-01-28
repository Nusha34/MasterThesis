from bs4 import BeautifulSoup
import spacy
import os
import glob
import pickle

nlp = spacy.load("en_core_web_sm")


def preprocess_sentences(xml_file):
    sentences_abstract = []
    soup = BeautifulSoup(open(xml_file), "lxml-xml")
    text = soup.find_all("AbstractText")
    for el in text:
        text_new = el.get_text()
        tok_sentences = nlp(text_new)
        for sent in tok_sentences.sents:
            sentences_abstract.append(
                [token.text.lower() for token in sent if not token.is_stop and not token.is_punct])
    return sentences_abstract


if __name__ == '__main__':
    path = '/workspaces/master_thesis/data/pubmed_corpus/'
    sentences = []
    for file_name in glob.glob(os.path.join(path, "*.xml")):
        sentences += preprocess_sentences(file_name)
    with open("pubmed_corpus", "wb") as fp:
        pickle.dump(sentences, fp)
