import wiki as w
import pickle
import logging
from gensim.models.word2vec import Word2Vec


WIKIXML = "/workspaces/master_thesis/{lang}wiki-latest-pages-articles-multistream.xml.bz2"

if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
    logging.info("Wiki parsing")
    wiki_sentences = w.WikiSentences(WIKIXML.format(lang="en"), "en")
    logging.info("Save wiki corpus")
    with open("wiki_corpus_full", "wb") as fp: 
        pickle.dump(wiki_sentences, fp)
    #logging.info("Wiki loading")
    #with open(
    #    "/workspaces/master_thesis/wiki_corpus_full", "rb"
    #) as fp:  # Unpickling
    #    wiki_sentences = pickle.load(fp)
    logging.info("Training word2vec model..")
    model = Word2Vec(
        wiki_sentences, vector_size=300, min_count=10, sg=0, workers=12
    )
    logging.info("Training done.")
    logging.info("Save model")
    model.save("word2vec_wiki.model")
    logging.info("Done")
