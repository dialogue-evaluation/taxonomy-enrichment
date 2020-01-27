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
                w.write(f"{word} {vector_line}\n")


if __name__ == '__main__':
    ft_vec = FasttextVectorizer("models/cc.ru.300.bin")
    ruwordnet = RuWordnet(db_path="../dataset/ruwordnet.db", ruwordnet_path=None)
    noun_synsets = defaultdict(list)
    verb_synsets = defaultdict(list)
    for sense_id, synset_id, text in ruwordnet.get_all_senses():
        if synset_id.endswith("N"):
            noun_synsets[synset_id].append(text)
        elif synset_id.endswith("V"):
            verb_synsets[synset_id].append(text)

    ft_vec.vectorize_ruwordnet(noun_synsets, "models/vectors/nouns_ruwordnet_fasttext.txt")
    ft_vec.vectorize_ruwordnet(verb_synsets, "models/vectors/verbs_ruwordnet_fasttext.txt")

    with open("../dataset/public/verbs_public_no_labels.tsv", 'r', encoding='utf-8') as f:
        dataset = f.read().split("\n")[:-1]
    ft_vec.vectorize_data(dataset, "models/vectors/verbs_public_fasttext.txt")

    with open("../dataset/public/nouns_public_no_labels.tsv", 'r', encoding='utf-8') as f:
        dataset = f.read().split("\n")[:-1]
    ft_vec.vectorize_data(dataset, "models/vectors/nouns_public_fasttext.txt")

    with open("../dataset/private/verbs_private_no_labels.tsv", 'r', encoding='utf-8') as f:
        dataset = f.read().split("\n")[:-1]
    ft_vec.vectorize_data(dataset, "models/vectors/verbs_private_fasttext.txt")

    with open("../dataset/private/nouns_private_no_labels.tsv", 'r', encoding='utf-8') as f:
        dataset = f.read().split("\n")[:-1]
    ft_vec.vectorize_data(dataset, "models/vectors/nouns_private_fasttext.txt")
