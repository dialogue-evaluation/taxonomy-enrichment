import argparse
import codecs
from collections import defaultdict


# Consistent with Python 2


def read_dataset(data_path, sep='\t'):
    vocab = defaultdict(list)
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.replace("\n", '').split(sep)
            word = line_split[0]
            hypernyms = line_split[1]
            vocab[word].append(hypernyms)
    return vocab


def get_score(full, direct, predicted, k=10):
    ap_sum = 0
    rr_sum = 0

    for neologism in full:
        full_hypernyms = set(full.get(neologism, []))
        direct_hypernyms = set(direct.get(neologism, []))
        predicted_hypernyms = predicted.get(neologism, [])

        ap_sum += max(compute_ap(full_hypernyms, predicted_hypernyms, k),
                      compute_ap(direct_hypernyms, predicted_hypernyms, k))
        rr_sum += compute_rr(full_hypernyms, predicted_hypernyms, k)

    return ap_sum / len(full), rr_sum / len(full)


def compute_ap(actual, predicted, k=10):
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def compute_rr(true, predicted, k=10):
    for i, synset in enumerate(predicted[:k]):
        if synset in true:
            return 1.0 / (i + 1.0)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('full')
    parser.add_argument('direct')
    parser.add_argument('predicted')
    args = parser.parse_args()

    full = read_dataset(args.full)
    direct = read_dataset(args.direct)
    submitted = read_dataset(args.predicted)
    if set(full) != set(submitted):
        print("Not all words are presented in your file")
    mean_ap, mean_rr = get_score(full, direct, submitted)
    print("map: {0}\nmrr: {1}\n".format(mean_ap, mean_rr))


if __name__ == '__main__':
    main()
