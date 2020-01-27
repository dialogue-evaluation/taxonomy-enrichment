import os
import argparse
import json
import time
from collections import defaultdict

from ruwordnet.ruwordnet_reader import RuWordnet


# -------------------------------------------------------------
# get ruwordnet
# -------------------------------------------------------------

def retrieve_ruwordnet_positions(input_filename: str, output_path: str, synset_senses: dict, sense2synset: dict):
    with open(input_filename, 'rt', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as w:
        texts = f.read().split("SpacesAfter=\\r\\n")
        context = [[line.split("\t")[:3] for line in text.split("\n") if line and not line.startswith("#")]
                   for text in texts]
        for sentence in context:
            tokens = [word[1] for word in sentence]
            lemmas = []
            for index, _, lemma in sentence:
                synsets, end = get_end(sentence, lemma, int(index), synset_senses, sense2synset)
                if synsets:
                    for synset in synsets:
                        lemmas.append((synset, (int(index)-1, end)))
            if lemmas:
                w.write(json.dumps([tokens, lemmas]) + "\n")


def get_end(sentence, first_lemma, index, senses_chain, sense2synset):
    last_index = index
    if first_lemma in senses_chain:
        sense_phrase = [first_lemma]
        for cur_index, token, lemma in sentence[index:]:
            if lemma in senses_chain[sense_phrase[-1]]:
                sense_phrase.append(lemma)
                last_index = int(cur_index)
            else:
                break
        if len(sense2synset[" ".join(sense_phrase).upper()]) > 0:
            return sense2synset[" ".join(sense_phrase).upper()], last_index
    return False, last_index

# -------------------------------------------------------------
# get test data
# -------------------------------------------------------------


def retrieve_word_positions(input_filename, output_path, testset) -> None:
    with open(input_filename, 'rt', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as w:
        texts = f.read().split("SpacesAfter=\\r\\n")
        context = [[line.split("\t")[:3] for line in text.split("\n") if line and not line.startswith("#")]
                   for text in texts]
        for sentence in context:
            tokens = [word[1] for word in sentence]
            lemmas = [(lemma, (int(index)-1, int(index))) for index, token, lemma in sentence if lemma in testset]
            if lemmas:
                w.write(json.dumps([tokens, lemmas]) + "\n")


# -------------------------------------------------------------
#  ruwordnet transformations
# -------------------------------------------------------------

def create_sense2synset(senses, pos):
    sense2synset = defaultdict(list)
    for id, synset, text in senses:
        if synset.endswith(pos):
            sense2synset[text].append(synset)
    return sense2synset


def create_senses_chain(ruwordnet, pos):
    synset_senses = defaultdict(set)
    for _, synset, text in ruwordnet.get_all_senses():
        if synset.endswith(pos):
            for token, next_token in create_synset_senses(text.lower()):
                synset_senses[token].add(next_token)
    return synset_senses


def create_synset_senses(text):
    tokens = text.split()
    return [(token, next_token) for token, next_token in zip(tokens, tokens[1:] + [True])]


def read_test_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return set(f.read().lower().split("\n")[:-1])


def parse_args():
    # create the top-level parser
    parser = argparse.ArgumentParser(description="get context with positions")
    parser.add_argument('--corpus_path', type=str, dest="corpus_path", help="lemmatized ud news corpus path")
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


def main():
    args = parse_args()
    file_paths = [os.path.join(x, i) for x, _, z in os.walk(args.corpus_path) for i in z]

    if "ruwordnet_path" in args:
        ruwordnet = RuWordnet(db_path=args.ruwordnet_path, ruwordnet_path="")
        sense2synset = create_sense2synset(ruwordnet.get_all_senses(), args.pos)
        synset_senses = create_senses_chain(ruwordnet, args.pos)
        for filename in file_paths:
            start_time = time.time()
            retrieve_ruwordnet_positions(filename, args.output_path, synset_senses, sense2synset)
            print(f"---- File {filename} took {(time.time() - start_time)} seconds ----")

    elif "data_path" in args:
        data = read_test_data(args.data_path)
        for filename in file_paths:
            start_time = time.time()
            retrieve_word_positions(filename, args.output_path, data)
            print(f"---- File {filename} took {(time.time() - start_time)} seconds ----")


if __name__ == '__main__':
    main()
