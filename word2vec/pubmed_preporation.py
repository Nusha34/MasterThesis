import pubmed_parsing_full_nost as p
import pickle
import logging
from gensim.models.word2vec import Word2Vec


if __name__ == "__main__":
    logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
    logging.info("Pubmed parsing")
    pubmed_sentences=p.PubMedSentences('/workspaces/master_thesis/data/pubmed/pubmedxml/')
    logging.info("Save pubmed corpus")
    with open("0206_pubmed_corpus_full_300_no_stem", "wb") as fp: 
        pickle.dump(pubmed_sentences, fp)