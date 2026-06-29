import pymorphy3
import re


class Lemmatizer:
    PREPOSITIONS = {
        'в', 'на', 'с', 'из', 'за', 'по', 'о', 'к', 'от', 'до', 'для', 'без',
        'при', 'через', 'под', 'над', 'перед', 'между', 'после', 'около',
        'вокруг', 'вследствие', 'насчёт', 'ради', 'вроде', 'наподобие',
        'вследствие', 'насчёт', 'ради', 'несмотря', 'включая', 'исключая'
    }

    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    def lemmatize(self, text: str) -> list[str]:
        words = re.findall(r'\b[а-яёА-ЯЁ]+\b', text.lower())
        lemmas = []
        for word in words:
            p = self.morph.parse(word)[0]
            lemmas.append(p.normal_form)
        return lemmas

    def lemmatize_filtered(self, text: str) -> set[str]:
        lemmas = self.lemmatize(text)
        return {lemma for lemma in lemmas if lemma not in self.PREPOSITIONS}

    def get_used_words(self, original: str, paraphrase: str) -> list[str]:
        original_lemmas = self.lemmatize_filtered(original)
        paraphrase_lemmas = set(self.lemmatize(paraphrase))
        used = original_lemmas & paraphrase_lemmas
        return list(used)

    def check_no_words_used(self, original: str, paraphrase: str) -> bool:
        used_words = self.get_used_words(original, paraphrase)
        return len(used_words) == 0

    def get_lemmas_info(self, original: str, paraphrase: str) -> dict:
        original_lemmas = self.lemmatize_filtered(original)
        paraphrase_lemmas = self.lemmatize_filtered(paraphrase)
        used_lemmas = original_lemmas & paraphrase_lemmas
        unique_original = original_lemmas - used_lemmas
        unique_paraphrase = paraphrase_lemmas - used_lemmas

        return {
            'original': sorted(list(original_lemmas)),
            'paraphrase': sorted(list(paraphrase_lemmas)),
            'matched': sorted(list(used_lemmas)),
            'unique_original': sorted(list(unique_original)),
            'unique_paraphrase': sorted(list(unique_paraphrase))
        }