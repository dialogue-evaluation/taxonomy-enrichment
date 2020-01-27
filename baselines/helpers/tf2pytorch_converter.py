import torch
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

BERT_MODEL_PATH = '../models/rubert_cased_deeppavlov/'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=False)

convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
    BERT_MODEL_PATH + 'bert_model.ckpt',
    BERT_MODEL_PATH + 'bert_config.json',
    '../models/rubert_cased_torch/pytorch_model.bin')
