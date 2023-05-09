# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import random
import torch
import numpy as np
import os
import pickle


def set_cuda(args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    devices_id = [int(device_id) for device_id in args.gpu.split()]
    device = (
        torch.device("cuda:{}".format(str(devices_id[0])))
        if use_cuda
        else torch.device("cpu")
    )
    return device, devices_id

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_data(args, tokenizer):
    dirname = f'../dataset/{args.data}'
    print(dirname)

    if not os.path.exists(f'{dirname}/tokenized'):
        os.makedirs(f'{dirname}/tokenized')

    if os.path.exists(f'{dirname}/tokenized/{args.data}_{args.max_seq_len}.pkl') and not args.rewrite_data:
        return read_pkl(f'{dirname}/tokenized/{args.data}_{args.max_seq_len}.pkl')

    print('tokenzie data')

    act_list = {}
    with open(os.path.join(dirname, f'act_{args.data}.txt'), 'r', encoding='utf-8') as infile:
        for line in infile:
            items = line.strip('\n').split('\t')
            act_list[items[0]] = items[1]

    data = {'act_list': act_list}
    for set_name in ['train', 'valid', 'test']:
        max_utt_len = 0
        max_dia_len = 0
        avg_utt_len = []
        avg_dia_len = []
        data[set_name] = {'input_ids': [], 'input_text': [], 'act_seq': [], 'sat': [], 'schema_ids': [], 'schema_text': []}
        with open(os.path.join(dirname, f'{args.data}_schema.json'), 'r', encoding='utf-8') as sfile:
            schema = json.load(sfile)
            schema_text = []
            schema_ids = []
            for service in schema:
                for attribute in service["slots"]:
                    schema_text.append(attribute["name"] + ". " + attribute["description"])
                    s_ids = tokenizer.encode(schema_text)[1:]
                    if len(s_ids) >= args.max_seq_len - 1:
                        s_ids = s_ids[:args.max_seq_len - 2] + [102]
                        break
                    schema_ids.append([101] + s_ids)  # [CLS] + (max_len-1) tokens

        with open(os.path.join(dirname, f'{set_name}_{args.data}.txt'), 'r', encoding='utf-8') as infile:
            for line in infile:
                items = line.strip('\n').split('\t')
                input_text = eval(items[0])
                act_seq = eval(items[1])
                sat = int(items[2])
                input_ids = []
                for text in input_text:
                    ids = []
                    for sent in text.split('|||'):
                        ids += tokenizer.encode(sent)[1:]
                        if len(ids) >= args.max_seq_len - 1:
                            ids = ids[:args.max_seq_len - 2] + [102]
                            break

                    avg_utt_len.append(len(ids))
                    max_utt_len = max(max_utt_len, len(ids))
                    input_ids.append([101] + ids)  # [CLS] + (max_len-1) tokens

                avg_dia_len.append(len(input_ids))
                max_dia_len = max(max_dia_len, len(input_ids))
                data[set_name]['input_ids'].append(input_ids)
                data[set_name]['input_text'].append(input_text)
                data[set_name]['act_seq'].append(act_seq)
                data[set_name]['sat'].append(sat)
                data[set_name]['schema_ids'].append(schema_ids)
                data[set_name]['schema_text'].append(schema_text)
        print('{} set, max_utt_len: {}, max_dia_len: {}, avg_utt_len: {}, avg_dia_len: {}'.format(set_name, max_utt_len,
                                                                                                  max_dia_len, float(
                sum(avg_utt_len)) / len(avg_utt_len), float(sum(avg_dia_len)) / len(avg_dia_len)))

    write_pkl(data, f'{dirname}/tokenized/{args.data}_{args.max_seq_len}.pkl')

    return data
