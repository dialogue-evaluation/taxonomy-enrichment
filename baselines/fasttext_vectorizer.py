from collections import defaultdict

import numpy as np
from gensim.models.fasttext import load_facebook_model
from string import punctuation
from ruwordnet.ruwordnet_reader import RuWordnet


class FasttextVectorizer:
    def __init__(self, model_path):
        self.model = load_facebook_model(model_path)
        print('Model loaded')

    # -------------------------------------------------------------
    # vectorize ruwordnet
    # -------------------------------------------------------------

    def vectorize_ruwordnet(self, synsets, output_path):
        ids, vectors = self.__get_ruwordnet_vectors(synsets)
        self.save_as_w2v(ids, vectors, output_path)

    def __get_ruwordnet_vectors(self, synsets):
        ids = []
        vectors = np.zeros((len(synsets), self.model.vector_size))
        for i, (_id, texts) in enumerate(synsets.items()):
            ids.append(_id)
            vectors[i, :] = self.__get_avg_vector(texts)
        return ids, vectors

    def __get_avg_vector(self, texts):
        sum_vector = np.zeros(self.model.vector_size)
        for text in texts:
            words = [i.strip(punctuation) for i in text.split()]
            sum_vector += np.sum(self.__get_data_vectors(words), axis=0)/len(words)
        return sum_vector/len(texts)

    # -------------------------------------------------------------
    # vectorize data
    # -------------------------------------------------------------

    def vectorize_data(self, data, output_path):
        data_vectors = self.__get_data_vectors(data)
        self.save_as_w2v(data, data_vectors, output_path)

    def __get_data_vectors(self, data):
        vectors = np.zeros((len(data), self.model.vector_size))
        for i, word in enumerate(data):  # TODO: how to do it more effective or one-line
            vectors[i, :] = self.model[word]
        return vectors

    # -------------------------------------------------------------
    # save
    # -------------------------------------------------------------

    @staticmethod
    def save_as_w2v(words: list, vectors: np.array, output_path: str):
        assert len(words) == len(vectors)
        with open(output_path, 'w', encoding='utf-8') as w:
            w.write(f"{vectors.shape[0]} {vectors.shape[1]}\n")
            for word, vector in zip(words, vectors):
                vector_line = " ".join(map(str, vector))
                w.write(f"{word.upper()} {vector_line}\n")


def process_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = f.read().lower().split("\n")[:-1]
    ft_vec.vectorize_data(dataset, output_file)


if __name__ == '__main__':
    ft_vec = FasttextVectorizer("models/cc.ru.300.bin")
    ruwordnet = RuWordnet(db_path="../data/ruwordnet.db", ruwordnet_path=None)
    noun_synsets = defaultdict(list)
    verb_synsets = defaultdict(list)
    for sense_id, synset_id, text in ruwordnet.get_all_senses():
        if synset_id.endswith("N"):
            noun_synsets[synset_id].append(text.lower())
        elif synset_id.endswith("V"):
            verb_synsets[synset_id].append(text.lower())

    ft_vec.vectorize_ruwordnet(noun_synsets, "models/vectors/ruwordnet_nouns.txt")
    ft_vec.vectorize_ruwordnet(verb_synsets, "models/vectors/ruwordnet_verbs.txt")

    process_data("../data/public_test/verbs_public.tsv", "models/vectors/verbs_public.txt")
    process_data("../data/public_test/nouns_public.tsv", "models/vectors/nouns_public.txt")
    process_data("../data/private_test/verbs_private.tsv", "models/vectors/verbs_private.txt")
    process_data("../data/private_test/nouns_private.tsv", "models/vectors/nouns_private.txt")
