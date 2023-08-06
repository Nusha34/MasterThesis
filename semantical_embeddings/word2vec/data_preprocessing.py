import spacy  

    
class Preprocessor:   
    """
    Class to prepare a corpus
    """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp.Defaults.stop_words.remove("no")
        self.nlp.Defaults.stop_words.remove("not")
        self.nlp.Defaults.stop_words.remove("none")
        self.nlp.Defaults.stop_words.remove("noone")
        self.nlp.Defaults.stop_words.remove("back")
        self.nlp.Defaults.stop_words.add("doctor")

    def __call__(self, data: Iterable[str]) -> Iterable[str]:
        for sample in data:
            sample = sample.lower()
            token_list = []
            doc = self.nlp(sample)
            token_list = [
                token.lemma_
                for token in doc
                if not token.is_stop and not token.is_punct
            ]
            text = " ".join(token_list)
            yield text