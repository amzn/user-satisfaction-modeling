# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import AdamW, BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import random
import numpy as np
import os
import pickle
import utils
import torch
from model import USDA
from sklearn.metrics import f1_score, precision_score, recall_score
import logging



class DataFrame(Dataset):
    def __init__(self, data, args):
        self.input_ids = data['input_ids']
        self.schema_ids = data['schema_ids']
        self.act_seq = data['act_seq']
        self.sat = data['sat']
        self.max_len = args.max_dia_len

    def __getitem__(self, index):
        return self.input_ids[index][-self.max_len:], self.act_seq[index][-self.max_len:], self.sat[index], self.schema_ids[index]

    def __len__(self):
        return len(self.input_ids)


def collate_fn(data):
    input_ids, act_seq, sat, schema_ids = zip(*data)
    batch_size = len(input_ids)
    act_seq = [torch.tensor(item).long() for item in act_seq]
    act_seq = pad_sequence(act_seq, batch_first=True, padding_value=-1)
    dialog_len = len(act_seq[0])
    schema_len = len(schema_ids[0])
    schema_ids = [torch.tensor(item).long() for attributes in schema_ids for item in attributes]
    schema_ids = pad_sequence(schema_ids, batch_first=True, padding_value=0)
    schema_ids = schema_ids.view(batch_size, schema_len, -1)

    pad_input_ids = []
    for dialog in input_ids:
        x = dialog + [[101, 102]] * (dialog_len - len(dialog))
        pad_input_ids.append(x)
    input_ids = [torch.tensor(item).long() for dialog in pad_input_ids for item in dialog]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_ids = input_ids.view(batch_size, dialog_len, -1)

    return {'input_ids': input_ids,
            'sat': torch.tensor(sat).long(),
            'schema_ids': schema_ids
            }


def train(args):
    print('[TRAIN]')

    torch.cuda.empty_cache()

    data_name = args.data.replace('\r', '')
    model_name = args.model.replace('\r', '')

    name = f'{data_name}_{model_name}_{args.max_dia_len}'
    if args.pretrain_model is not None:
        name += '_pretrain'
    print('TRAIN ::', name)

    save_path = f'outputs/{name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logging.basicConfig(level=logging.DEBUG, filename=save_path + '/log.txt', filemode='a')

    tokenizer = BertTokenizer.from_pretrained(args.bert_name, cache_dir=args.cache_dir)

    data = utils.load_data(args, tokenizer)

    
    if args.pretrain_model is not None:
        pretrain_model = Pretrained_HiTrans(args=args, vocab_size=tokenizer.vocab_size)
        checkpoint = torch.load(args.pretrain_model)
        pretrain_model.load_state_dict(checkpoint)
        pretrained_private = pretrain_model.private
        pretrained_encoder = pretrain_model.encoder
    else:
        pretrained_private = None
        pretrained_encoder = None
    if model_name == 'USDA':
        model = USDA(args=args, vocab_size=tokenizer.vocab_size, pretrained_private=pretrained_private, pretrained_encoder=pretrained_encoder)
    optimizer = AdamW(model.parameters(), 2e-5)


    sat_best_result = [0. for _ in range(4)]

    model.to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if len(args.device_id) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_id)
    
    utils.set_seed(args.seed)
    batch_size = args.batch_size * max(1, len(args.device_id))

    for i in range(args.epoch_num):
        logging.info('train epoch, {}, {}'.format(i, name))
        train_loader = DataLoader(DataFrame(data['train'], args), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        tk0 = tqdm(train_loader)
        model.train()
        epoch_loss = [0. for _ in range(2)]

        score = []

        for j, batch in enumerate(tk0):
            input_ids = batch['input_ids'].to(args.device)
            schema_ids = batch['schema_ids'].to(args.device)
            sat = batch['sat'].to(args.device)

            f = model(input_ids=input_ids, schema_ids=schema_ids, sat=sat, frequency=True)
            score.append(f)
        score = np.array(score)
        score = np.sum(score, axis=0)

        for j, batch in enumerate(tk0):
            input_ids = batch['input_ids'].to(args.device)
            schema_ids = batch['schema_ids'].to(args.device)
            sat = batch['sat'].to(args.device)

            sat_pred, sat_loss = model(input_ids=input_ids, schema_ids=schema_ids, sat=sat, score=score)

            if len(args.device_id) > 1:
                sat_loss = sat_loss.mean()

            epoch_loss[0] += sat_loss
            loss = sat_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss = [x/len(tk0) for x in epoch_loss]
        logging.info(f'loss: {epoch_loss}')

        sat_test_result = test(model, DataFrame(data['valid'], args), args)
        if sat_test_result[-1] > sat_best_result[-1]:
            sat_best_result = sat_test_result
            model_to_save = model.module if hasattr(model,
                        'module') else model  # Take care of distributed/parallel training
            torch.save(model_to_save.state_dict(), save_path+'/best_pretrain.pt')
        logging.info(f'satisfaction: valid_result={sat_test_result}')
        logging.info(f'satisfaction: best_valid_result={sat_best_result}')
        sat_test_result = test(model, DataFrame(data['test'], args), args)
        logging.info(f'satisfaction: test_result={sat_test_result}')
    
    # evaluation
    logging.info(f'evaluation ...')
    checkpoint = torch.load(save_path+'/best_pretrain.pt')
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    sat_test_result = test(model, DataFrame(data['test'], args), args)
    logging.info(f'satisfaction: test_result={sat_test_result}')


def test(model, test_data, args):
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    tk0 = tqdm(test_loader)

    sat_prediction = []
    sat_label = []

    model.eval()

    score = []
    for j, batch in enumerate(tk0):
        input_ids = batch['input_ids'].to(args.device)
        schema_ids = batch['schema_ids'].to(args.device)
        sat = batch['sat'].to(args.device)

        f = model(input_ids=input_ids, schema_ids=schema_ids, frequency=True)
        score.append(f)
    score = np.array(score)
    score = np.sum(score, axis=0)

    for j, batch in enumerate(tk0):
        input_ids = batch['input_ids'].to(args.device)
        schema_ids = batch['schema_ids'].to(args.device)
        sat = batch['sat'].to(args.device)
        with torch.no_grad():
            sat_pred, _, _, _ = model(input_ids=input_ids, schema_ids=schema_ids, score=score)
        sat_prediction.extend(sat_pred.argmax(dim=-1).cpu().tolist())
        sat_label.extend(sat.cpu().tolist())

    # satisfaction evaluation
    acc = sum([int(p == l) for p, l in zip(sat_prediction, sat_label)]) / len(sat_label)
    precision = precision_score(sat_label, sat_prediction, average='macro', zero_division=0)
    recall = recall_score(sat_label, sat_prediction, average='macro', zero_division=0)
    f1 = f1_score(sat_label, sat_prediction, average='macro', zero_division=0)
    sat_result = (acc, precision, recall, f1)

    return sat_result


def evaluate(args):
    print('[EVALUATE]')

    data_name = args.data.replace('\r', '')
    model_name = args.model.replace('\r', '')

    name = f'{data_name}_{model_name}_{args.max_dia_len}'
    if args.pretrain_model is not None:
        name += '_pretrain'
    print('EVALUATE ::', name)

    save_path = f'outputs/{name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logging.basicConfig(level=logging.DEBUG, filename=save_path + '/log_eval.txt', filemode='a')

    tokenizer = BertTokenizer.from_pretrained(args.bert_name, cache_dir=args.cache_dir)

    data = utils.load_data(args, tokenizer)

    if model_name == 'USDA':
        model = USDA(args=args, vocab_size=tokenizer.vocab_size)

    model.to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if len(args.device_id) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_id)

    logging.info(f'evaluation ...')
    checkpoint = torch.load(save_path+'/best_pretrain.pt')
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    print(sum(p.numel() for p in model.parameters()))
    test_loader = DataLoader(DataFrame(data[f'{args.eval_set}'], args), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    tk0 = tqdm(test_loader)

    sat_prediction = []
    sat_label = []
    
    pointer_list = []
    att_a_list = []
    att_c_list = []

    model.eval()

    score = []
    for j, batch in enumerate(tk0):
        input_ids = batch['input_ids'].to(args.device)
        schema_ids = batch['schema_ids'].to(args.device)
        sat = batch['sat'].to(args.device)

        f = model(input_ids=input_ids, schema_ids=schema_ids, frequency=True)
        score.append(f)
    score = np.array(score)
    score = np.sum(score, axis=0)

    for j, batch in enumerate(tk0):
        input_ids = batch['input_ids'].to(args.device)
        schema_ids = batch['schema_ids'].to(args.device)
        sat = batch['sat'].to(args.device)
        with torch.no_grad():
            sat_pred, pointer, att_a, att_c = model(input_ids=input_ids, schema_ids=schema_ids, score=score)
        sat_prediction.extend(sat_pred.argmax(dim=-1).cpu().tolist())
        sat_label.extend(sat.cpu().tolist())
        pointer_list.extend(pointer.squeeze(-1).cpu().tolist())
        att_a_list.extend(att_a.cpu().tolist())
        att_c_list.extend(att_c.cpu().tolist())

    # satisfaction evaluation
    acc = sum([int(p == l) for p, l in zip(sat_prediction, sat_label)]) / len(sat_label)
    precision = precision_score(sat_label, sat_prediction, average='macro', zero_division=0)
    recall = recall_score(sat_label, sat_prediction, average='macro', zero_division=0)
    f1 = f1_score(sat_label, sat_prediction, average='macro', zero_division=0)
    sat_result = (acc, precision, recall, f1)

    precision = precision_score(sat_label, sat_prediction, average=None, zero_division=0)
    recall = recall_score(sat_label, sat_prediction, average=None, zero_division=0)
    f1 = f1_score(sat_label, sat_prediction, average=None, zero_division=0)
    sat_result_details = (precision, recall, f1)

    logging.info(f'satisfaction details: {sat_result_details}')


    # detailed analyses
    avg_p = sum(pointer_list)/len(pointer_list)
    logging.info(f'average point value: {avg_p}')

    with open(save_path + '/results.txt', 'w') as outfile:
        for i in range(len(pointer_list)):
            mask = len(data[f"{args.eval_set}"]["input_text"][i])
            outfile.write(f'{data[f"{args.eval_set}"]["input_text"][i]}\t{sat_label[i]}\t{sat_prediction[i]}\t{att_a_list[i][:mask]}\t{att_c_list[i][:mask]}\t{pointer_list[i]}\n')
