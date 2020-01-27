import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch

device = torch.device('cuda')
n_gpu = torch.cuda.device_count()

for i in range(n_gpu):
    print(torch.cuda.get_device_name(i))

torch.manual_seed(117)

from pytorch_pretrained_bert import BertTokenizer, BertModel


class BertPretrained:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.model.cuda('cuda')
        self.model.eval()

    def vectorize_sentences(self, sentences):
        result_batch = []
        indexed_batch = []
        segment_batch = []
        mapping = []
        for sentence in sentences:
            tokens, ids, map_to_tok = self.tokenize(sentence)
            indexed_batch.append(tokens)
            segment_batch.append(ids)
            mapping.append(map_to_tok)

        tokens_tensor = torch.LongTensor(pad_sequences(indexed_batch, dtype='long', padding='post'))
        segments_tensors = torch.LongTensor(pad_sequences(segment_batch, dtype='long', padding='post'))
        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = segments_tensors.to('cuda')

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)

        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = token_embeddings[-4:, :, :, :].permute(1, 2, 0, 3)
        token_embeddings = torch.flatten(token_embeddings, start_dim=2)

        for index, tensor in zip(mapping, token_embeddings):
            result_batch.append(self.unmap_to_tokens(index, tensor))
        return result_batch

    def tokenize(self, orig_tokens):
        tokenized_text = ["[CLS]"]
        orig_to_tok_map = []
        for orig_token in orig_tokens:
            orig_to_tok_map.append(len(tokenized_text))
            tokenized_text.extend(self.tokenizer.tokenize(orig_token))
        orig_to_tok_map.append(len(tokenized_text))
        tokenized_text.append("[SEP]")
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        return indexed_tokens, segments_ids, orig_to_tok_map

    def unmap_to_tokens(self, mapping, tensor):
        result = np.zeros((len(mapping) - 1, 3072))
        for j, index in enumerate(mapping):
            if j != len(mapping) - 1:
                result[j] = torch.mean(tensor.cpu()[index:mapping[j + 1]], 0)
        return result
