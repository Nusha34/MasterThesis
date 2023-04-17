class CombinedSentences:
    def __init__(self, pubmed_sentences, wiki_sentences):
        self.pubmed_sentences = pubmed_sentences
        self.wiki_sentences = wiki_sentences

    def __iter__(self):
        for sentence in self.pubmed_sentences:
            yield sentence
        for sentence in self.wiki_sentences:
            yield sentence