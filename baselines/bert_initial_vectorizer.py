import argparse
from collections import defaultdict
from string import punctuation
from tqdm import tqdm
import numpy as np

from ruwordnet.ruwordnet_reader import RuWordnet
from bert_model import BertPretrained


class BertVectorizer:
    def __init__(self, model_path):
        self.bert = BertPretrained(model_path)

    # -------------------------------------------------------------
    # get ruwordnet
    # -------------------------------------------------------------

    def vectorize_ruwordnet(self, synsets, output_path):
        vectors = {synset: np.mean([np.mean(sentence_vectors, 0)
                                    for sentence_vectors in self.bert.vectorize_sentences(texts)], 0)
                   for synset, texts in tqdm(synsets.items())}
        self.save_as_w2v(vectors, output_path)

    # -------------------------------------------------------------
    # get dataset
    # -------------------------------------------------------------

    def vectorize_data(self, data, output_path):
        batch = self.bert.vectorize_sentences([[i] for i in data])
        vectors = {word: np.mean(sentence_vectors, 0) for sentence_vectors, word in zip(batch, data)}
        self.save_as_w2v(vectors, output_path)


    @staticmethod
    def save_as_w2v(dictionary, output_path):
        with open(output_path, 'w', encoding='utf-8') as w:
            w.write(f"{len(dictionary)} {list(dictionary.values())[0].shape[-1]}\n")
            for word, vector in dictionary.items():
                vector_line = " ".join(map(str, vector))
                w.write(f"{word} {vector_line}\n")


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().split("\n")[:-1]


def parse_args():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--bert_path', type=str, dest="bert_path", help='bert model dir')
    parser.add_argument('--output_path', type=str, dest="output_path", help='output_path')
    subparsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "ruwordnet" command
    ruwordnet_parser = subparsers.add_parser('ruwordnet', help='ruwordnet help')
    ruwordnet_parser.add_argument('--ruwordnet_path', type=str, help='ruwordnet database path')
    ruwordnet_parser.add_argument('--pos', choices='NV', help="choose pos-tag to subset ruwordnet")

    # create the parser for the "data" command
    parser_b = subparsers.add_parser('data', help='data help')
    parser_b.add_argument('--data_path', type=str, dest="data_path", help='path to test data')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    bert_vectorizer = BertVectorizer(args.bert_path)

    if 'ruwordnet_path' in args:
        # --bert_path "models/rubert_cased_torch" --output_path "models/vectors/ruwordnet_nouns_single_bert3.txt"
        # ruwordnet --pos N --ruwordnet_path "../dataset/ruwordnet.db"

        senses = RuWordnet(args.ruwordnet_path, None).get_all_senses()
        synsets = defaultdict(list)

        for sense_id, synset_id, text in senses:
            sentence = [i.strip(punctuation) for i in text.lower().split()]
            if synset_id.endswith(args.pos):
                synsets[synset_id].append(sentence)

        bert_vectorizer.vectorize_ruwordnet(synsets, args.output_path)

    if "data_path" in args:
        # --bert_path "models/rubert_cased_torch" --output_path "models/vectors/nouns_public_single_bert3.txt"
        # data --data_path "../dataset/public/nouns_public_no_labels.tsv"

        data = read_file(args.data_path)
        bert_vectorizer.vectorize_data(data, args.output_path)
