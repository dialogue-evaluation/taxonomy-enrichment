from abc import abstractmethod

from ruwordnet.ruwordnet_reader import RuWordnet
from gensim.models import KeyedVectors


class Model:
    def __init__(self, params):
        self.ruwordnet = RuWordnet(db_path=params["db_path"], ruwordnet_path=params["ruwordnet_path"])
        self.w2v_ruwordnet = KeyedVectors.load_word2vec_format(params['ruwordnet_vectors_path'], binary=False)
        self.w2v_data = KeyedVectors.load_word2vec_format(params['data_vectors_path'], binary=False)

    @abstractmethod
    def predict_hypernyms(self, neologisms, topn=10):
        pass

    @abstractmethod
    def __compute_hypernyms(self, neologisms, topn=10):
        pass


class BaselineModel(Model):
    def __init__(self, params):
        super().__init__(params)

    def predict_hypernyms(self, neologisms, topn=10) -> dict:
        return {neologism: self.__compute_hypernyms(neologism, topn) for neologism in neologisms}

    def __compute_hypernyms(self, neologism, topn=10) -> list:
        return [i[0] for i in self.w2v_ruwordnet.similar_by_vector(self.w2v_data[neologism], topn)]


class SecondOrderModel(Model):
    def __init__(self, params):
        super().__init__(params)

    def predict_hypernyms(self, neologisms, topn=10) -> dict:
        return {neologism: self.__compute_hypernyms(neologism, topn) for neologism in neologisms}

    def __compute_hypernyms(self, neologism, topn=10) -> list:
        hypernyms = []
        associates = [i[0] for i in self.w2v_ruwordnet.similar_by_vector(self.w2v_data[neologism], topn)]
        for associate in associates:
            hypernyms.extend(self.ruwordnet.get_hypernyms_by_id(associate))
        return hypernyms[:10]
