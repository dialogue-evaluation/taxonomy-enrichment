import os
import gzip
import json


def main():
    file_paths = [os.path.join(x, i) for x, _, z in os.walk("../../dataset/news_dataset") for i in z]
    for i, filename in enumerate(file_paths):
        output_path = os.path.join("../../dataset/news_texts",
                                   os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0]+".txt")
        with gzip.open(filename, 'rt', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as w:
            _ = next(f)
            for line in f:
                name, sentences_line = line[:-1].split("\t", 1)
                for sentence in json.loads(sentences_line):
                    w.write(sentence+"\n")
        if i % 10 == 0:
            print(f"{i} texts done")


if __name__ == '__main__':
    main()
