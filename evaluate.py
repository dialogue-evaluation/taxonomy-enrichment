import argparse
import codecs
import json
from collections import defaultdict


# Consistent with Python 2


def read_dataset(data_path, read_fn=lambda x: x, sep='\t'):
    vocab = defaultdict(list)
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.replace("\n", '').split(sep)
            word = line_split[0]
            hypernyms = read_fn(line_split[1])
            vocab[word].append(hypernyms)
    return vocab


def get_score(reference, predicted, k=10):
    ap_sum = 0
    rr_sum = 0

    for neologism in reference:
        reference_hypernyms = reference.get(neologism, [])
        predicted_hypernyms = predicted.get(neologism, [])

        ap_sum += compute_ap(reference_hypernyms, predicted_hypernyms, k)
        rr_sum += compute_rr([j for i in reference_hypernyms for j in i], predicted_hypernyms, k)
    return ap_sum / len(reference), rr_sum / len(reference)


def compute_ap(actual, predicted, k=10):
    if not actual:
        return 0.0

    predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    already_predicted = set()
    skipped = 0
    for i, p in enumerate(predicted):
        if p in already_predicted:
            skipped += 1
            continue
        for parents in actual:
            if p in parents:
                num_hits += 1.0
                score += num_hits / (i + 1.0 - skipped)
                already_predicted.update(parents)
                break

    return score / min(len(actual), k)


def compute_rr(true, predicted, k=10):
    for i, synset in enumerate(predicted[:k]):
        if synset in true:
            return 1.0 / (i + 1.0)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('predicted')
    args = parser.parse_args()

    reference = read_dataset(args.reference, lambda x: json.loads(x))
    submitted = read_dataset(args.predicted)
    if len(set(reference).intersection(set(submitted))) == 0:
        raise Exception("Reference and Submitted files have no samples in common")
    elif set(reference) != set(submitted):
        print("Not all words are presented in your file")
    mean_ap, mean_rr = get_score(reference, submitted, k=10)
    print("map: {0}\nmrr: {1}\n".format(mean_ap, mean_rr))


if __name__ == '__main__':
    main()
