# taxonomy-enrichment baselines

## Baseline using fastText embedidngs:

#### Pipeline steps

1. Compute fastText embeddings of the ruWordNet synsets averaging word embeddings of each sense in the synset.
2. Compute fastText embeddings of the new (not yet included) words.
3. Select *k=10* hypernyms of all top closest synsets as possible hypernym candidates.

#### Pre-trained models

fastText: [link](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz)

#### RUN

1. Compute fastText embeddings: _**baselines/fasttext_vectorizer.py**_
2. Select *k=10* hypernyms of all top closest synsets as possible hypernym candidates: _**baselines/main.py**_

File  _**baselines/main.py**_ requires config path in *json* format.

```
python3 main.py config.json
```

Example with the required _json_ pairs in _config.json_ file:

```json
{
  "ruwordnet_vectors_path": "../baselines/models/vectors/ruwordnet_nouns_fasttext.txt",
  "data_vectors_path": "../baselines/models/vectors/nouns_public_fasttext.txt",
  "test_path": "../dataset/public/nouns_public_no_labels.tsv",
  "output_path": "predictions/predicted_public_nouns_fasttext.tsv",
  "db_path": "../dataset/ruwordnet.db",
  "ruwordnet_path": null,
  "model": "baseline"
}
```

---------------------------------------

## Baseline using BERT embedidngs:

#### Pipeline steps

1. Compute BERT embeddings of the ruWordNet synsets by averaging BERT embeddings of each sense in the synset.
2. Lemmatize news documents to be able to find not only exact word matches, but also its grammatical forms.
3. Extract sentences with the positions of the new words in those sentences.
4. Compute BERT embeddings for those words, averaging vectors for multiple sentences occurrences.
5. Select *k=10* hypernyms of the top closest synsets for each new word using cosine similarity measure.

#### Pre-trained models

* ruBERT: [deeppavlov link](http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v2.tar.gz)
* UDPipe: [link](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2998/russian-syntagrus-ud-2.4-190531.udpipe)

#### RUN

1. Read news dataset: _**baselines/helpers/texts_extractor.py**_
2. Lemmatize news documents: _**baselines/helpers/lemmatize_ud.bash**_
3. Extract sentences with the positions of the new words in those sentence: _**baselines/helpers/news_corpus_reader.py**_

```
usage: news_corpus_reader.py [-h] [--corpus_path CORPUS_PATH]
                             [--output_path OUTPUT_PATH]
                             {ruwordnet,data} ...

get context sentences with synsets positions

positional arguments:
  {ruwordnet,data}      sub-command help
    ruwordnet           ruwordnet help
    data                data help


optional arguments:
  -h, --help            show this help message and exit
  --corpus_path CORPUS_PATH
                        lemmatized ud news corpus path
  --output_path OUTPUT_PATH
                        output_path
  ```

Example: 
```sh
python3 news_corpus_reader.py --corpus_path ../models/news_lemmatized --output_path ../models/parsed_news_for_bert/ruwordnet_verbs.json ruwordnet --ruwordnet_path ../../data/ruwordnet.db --pos V
```

3. Compute BERT embeddings (without context): _**baselines/bert_initial_vectorizer.py**_

```

usage: PROG [-h] [--bert_path BERT_PATH] [--output_path OUTPUT_PATH]
            {ruwordnet,data} ...

positional arguments:
  {ruwordnet,data}      sub-command help
    ruwordnet           ruwordnet help
    data                data help

optional arguments:
  -h, --help            show this help message and exit
  --bert_path BERT_PATH
                        bert model dir
  --output_path OUTPUT_PATH
                        output_path
```

Example
```sh
python3 bert_initial_vectorizer.py --bert_path "models/rubert_cased_torch" --output_path "models/vectors/ruwordnet_nouns_bert.txt ruwordnet --pos N --ruwordnet_path "../dataset/ruwordnet.db"
```

4. Compute BERT embeddings (with context): _**baselines/bert_context_vectorizer.py**_

```
usage: BERT context vectorizer [-h] [--bert_path BERT_PATH]
                               [--vectors_path VECTORS_PATH]
                               [--output_path OUTPUT_PATH]
                               [--texts_dir TEXTS_DIR]
                               [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --bert_path BERT_PATH
                        bert model dir
  --vectors_path VECTORS_PATH
                        vectors_path
  --output_path OUTPUT_PATH
                        output_path
  --texts_dir TEXTS_DIR
                        texts_dir
  --batch_size BATCH_SIZE
                        batch size
```


Example:
```
python3 bert_context_vectorizer.py --bert_path "models/rubert_cased_torch" --texts_dir "models/parsed_news_for_bert" --vectors_path "models/vectors/nouns_public_bert.txt" --output_path "models/vectors/nouns_public_context_bert.txt"
```

5. Select *k=10* hypernyms of the top closest synsets for each new word using cosine similarity measure: _**baselines/main.py**_

File  _**baselines/main.py**_ requires config path in *json* format.

```
python3 main.py config.json
```

Example with the required _json_ pairs in _config.json_ file:

```json
{
  "ruwordnet_vectors_path": "../baselines/models/vectors/ruwordnet_nouns_bert.txt",
  "data_vectors_path": "../baselines/models/vectors/nouns_public_bert.txt",
  "test_path": "../dataset/public/nouns_public_no_labels.tsv",
  "output_path": "predictions/predicted_public_nouns_bert.tsv",
  "db_path": "../dataset/ruwordnet.db",
  "ruwordnet_path": null,
  "model": "second_order"
}
```
