import pubmed_parsing_full as p
import pickle
import logging
from gensim.models.word2vec import Word2Vec


if __name__ == "__main__":
    logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
    logging.info("Pubmed parsing")
    pubmed_sentences=p.PubMedSentences('/workspaces/master_thesis/data/pubmed_corpus/')
    logging.info("Save pubmed corpus")
    with open("pubmed_corpus_full", "wb") as fp: 
        pickle.dump(pubmed_sentences, fp)
    logging.info("Training word2vec model..")
    model = Word2Vec(
        pubmed_sentences, vector_size=300, min_count=10, sg=0, workers=12
    )
    logging.info("Training done.")
    logging.info("Save model")
    model.save("word2vec_pubmed.model")
    logging.info("Done")