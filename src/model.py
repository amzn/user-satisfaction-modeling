# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import AdamW, BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from tqdm import tqdm
import copy
from transformer import TransformerEncoder



def init_params(model):
    for name, param in model.named_parameters():
        if param.data.dim() > 1:
            xavier_uniform_(param.data)
        else:
            pass


def universal_sentence_embedding(sentences, mask, sqrt=True):
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = (mask.sum(dim=1).view(-1, 1).float())
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums



class BERTBackbone(nn.Module):
    def __init__(self, **config):
        super().__init__()
        bert_name = config.get('bert_name', 'bert-base-uncased')
        cache_dir = config.get('cache_dir')
        self.bert = BertModel.from_pretrained(bert_name, cache_dir=cache_dir)
        self.d_model = 768*2

    def forward(self, input_ids, **kwargs):
        attention_mask = input_ids.ne(0).detach()
        outputs = self.bert(input_ids, attention_mask)
        h = universal_sentence_embedding(outputs[0], attention_mask)
        cls = outputs[1]

        out = torch.cat([cls, h], dim=-1)
        return out


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, din):
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        return dout


class USDA(nn.Module):
    def __init__(self, args, vocab_size, pretrained_private=None, pretrained_encoder=None):
        super().__init__()
        self.drop_out = nn.Dropout(args.dropout)
        if pretrained_private is not None:
            self.private = pretrained_private
        else:
            self.private = BERTBackbone(bert_name=args.bert_name, cache_dir=args.cache_dir)
        d_model = self.private.d_model

        self.content_gru = nn.GRU(d_model, d_model, num_layers=1, bidirectional=False, batch_first=True)

        self.sat_classifier = nn.Linear(d_model, 3)

        self.w_a = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=1)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        init_params(self.sat_classifier)
        init_params(self.w_a)

    def forward(self, input_ids, schema_ids=None, sat=None, frequency=False, score=None, **kwargs):
        self.content_gru.flatten_parameters()

        batch_size, dialog_len, utt_len = input_ids.size()
        batch_size, attribute_len, des_len = schema_ids.size()

        input_ids = input_ids.view(-1, utt_len)
        schema_ids = schema_ids.view(-1, des_len)

        dialogue = self.private(input_ids=input_ids, **kwargs)
        dialogue = dialogue.view(batch_size, dialog_len, -1)
        attribute = self.private(input_ids=schema_ids, **kwargs)
        attribute = attribute.view(batch_size, attribute_len, -1)

        attention = self.softmax(torch.exp(self.w_a(dialogue).mm(attribute.transpose(1, 2)))) # (batch, dialogue_len, attribute_len)
        attention_attribute = attention.transpose(1, 2).mm(dialogue) # (batch, attribute_len, -1)

        if frequency:
            dialogue_tile = torch.tile(dialogue, (1, 1, attribute_len)).view(batch_size, dialog_len*attribute_len, -1)
            attribute_tile = torch.tile(attribute, (1, 1, dialog_len)).view(batch_size, dialog_len * attribute_len, -1)

            relevance = self.cos(dialogue_tile, attribute_tile).view(batch_size, dialog_len, attribute_len)
            selected_attribute = torch.argmax(relevance, dim=2) # (batch, dialogue_len)
            selected_attribute_onehot = torch.nn.functional.one_hot(selected_attribute, num_classes=attribute_len) # (batch, dialogue_len, attribute_len)
            position = torch.tile(torch.range(1, dialog_len).view(1, dialog_len), (1, batch_size)).view(batch_size, dialog_len)
            f = torch.sum(torch.div(selected_attribute_onehot, position), (1, 0)).view(1, attribute_len)
            return f

        # score(1, attribute_len)
        score = torch.tile(score, (1, batch_size)).view(batch_size, 1, attribute_len)
        hidden = torch.squeeze(score.mm(attention_attribute)) # (batch, dim)

        sat_res = self.sat_classifier(hidden)

        if self.training:
            sat_loss = F.cross_entropy(sat_res, sat)
            return sat_res, sat_loss

        return sat_res

