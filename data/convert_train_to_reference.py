import sys


def main():
    if len(sys.argv) < 3:
        raise Exception("Please specify path to train file and output_file")
    with open(sys.argv[1], 'r', encoding='utf-8')as f, open(sys.argv[2], 'w', encoding='utf-8') as w:
        _ = next(f)
        for line in f:
            synset, _, parents, _ = line.split('\t')
            parents = parents.replace("'", '"')
            w.write(f"{synset}\t{parents}\n")


if __name__ == '__main__':
    main()
