import sys
import json
import codecs

from predict_models import BaselineModel, SecondOrderModel


def save_to_file(words_with_hypernyms, output_path, ruwordnet):
    with codecs.open(output_path, 'w', encoding='utf-8') as f:
        for word, hypernyms in words_with_hypernyms.items():
            for hypernym in hypernyms:
                f.write(f"{word}\t{hypernym}\t{ruwordnet.get_name_by_id(hypernym)}\n")


def load_config():
    if len(sys.argv) < 2:
        raise Exception("Please specify path to config file")
    with open(sys.argv[1], 'r', encoding='utf-8')as j:
        params = json.load(j)
    return params


def main():
    models = {"baseline": BaselineModel, "second_order": SecondOrderModel}
    params = load_config()
    with open(params['test_path'], 'r', encoding='utf-8') as f:
        test_data = f.read().split("\n")[:-1]
    baseline = models[params["model"]](params)
    print("Model loaded")
    results = baseline.predict_hypernyms(list(test_data))
    save_to_file(results, params['output_path'], baseline.ruwordnet)


if __name__ == '__main__':
    main()
