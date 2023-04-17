import wiki as w
import pickle
import logging
import pubmed_parsing_full_nost as p
from gensim.models.word2vec import Word2Vec
from combined_sentences import CombinedSentences


WIKIXML = "/workspaces/master_thesis/{lang}wiki-latest-pages-articles-multistream.xml.bz2"

if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
    logging.info("Wiki parsing")
    wiki_sentences = w.WikiSentences(WIKIXML.format(lang="en"), "en")
    logging.info("Pubmed parsing")
    pubmed_sentences=p.PubMedSentences('/workspaces/master_thesis/data/pubmed/pumbed_xml_300_1000/')
    combined_sentences = CombinedSentences(pubmed_sentences, wiki_sentences)
    model = Word2Vec(
        combined_sentences, vector_size=300, min_count=10, sg=0, workers=12
    )
    logging.info("Training done.")
    logging.info("Save model")
    model.save("word2vec_pubmed_nostemming_together.model")
    logging.info("Done")
