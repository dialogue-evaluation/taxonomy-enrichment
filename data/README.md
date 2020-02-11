# Datasets

We provide the following data:

- full RuWordNet
- public testset (new words with no ground truth synsets, available for scoring at Codalab)
- private testset (new words with no ground truth synsets, available for submission at Codalab, scores will be available after the end of evaluation period)
- sample answers (example of the submission format)
- training data
- get_reference_format.py -- script for converting synsets_\[nouns|verbs\].tsv files to format of the reference and for making partition into train/dev/test (allows arbitrary number of subsets and their sizes)

**Training data** includes:
- synsets_\[nouns|verbs\].tsv -- data extracted from RuWordNet, in the format "SYNSET\<TAB\>SENSES\<TAB\>PARENT SYNSETS\<TAB\>DEFINITION". The files contain synsets from RuWordNet which are:
    - leaves
    - at least 5 hops from the nearest root
- all_data_\[nouns|verbs\].tsv -- synsets_\[nouns|verbs\].tsv converted to the format "SENSE\<TAB\>PARENT SYNSETS" (format of the reference)
- training_\[nouns|verbs\].tsv, dev_\[nouns|verbs\].tsv, test_\[nouns|verbs\].tsv -- partition of all_data_\[nouns|verbs\].tsv in proportion 0.8/0.1/0.1