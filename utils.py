import os
import json
import gzip
from copy import deepcopy, copy
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler

from transformers.tokenization_utils import trim_batch

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smooth, tgt_vocab_size, ignore_index=-100):
        assert 0. < label_smooth <= 1.
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smooth / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0).unsqueeze(0))

        self.confidence = 1.0 - label_smooth
        self.lossfct = torch.nn.KLDivLoss(reduction='none')

    def forward(self, pred, target):
        """
        Args:
            pred: [bsz, seq_len, vocab_size]
            target: [bsz, seq_len]

        Returns:
        """
        model_prob = self.one_hot.repeat(target.size(0), target.size(1), 1)  # [bsz, seq_len, vocab_size]
        model_prob.scatter_(2, target.unsqueeze(2), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(2), 0)
        pred_prob = F.log_softmax(pred, dim=2)

        #return F.kl_div(pred_prob, model_prob, reduction='mean')
        loss = self.lossfct(pred_prob, model_prob)
        loss = torch.sum(loss, dim=2).masked_fill_((target == self.ignore_index), 0)
        avg_loss = torch.sum(loss) / torch.sum((target != self.ignore_index).to(torch.float))
        return avg_loss

# Special symbols
SOS_token = "<SOS>" # start of sentence
EOS_token = "<EOS>" # end of sentence
PAD_token = SOS_token # padding symbol
INPUT_TOKENS_SCAN = ['jump', 'opposite', 'right', 'twice', 'and', 'turn', 'thrice', 'run', 'after', 'around', 'left', 'walk', 'look']
OUTPUT_TOKENS_SCAN = ['I_TURN_RIGHT', 'I_JUMP', 'I_TURN_LEFT', 'I_RUN', 'I_WALK', 'I_LOOK']

# ACTION_TO_TEXT = {'I_TURN_RIGHT': 'right', 'I_JUMP': 'jump', 'I_TURN_LEFT': 'left', 'I_RUN': 'run', 'I_WALK': 'walk', 'I_LOOK': 'look'}

class Lang:
    # Class for converting strings/words to numerical indices, and vice versa.
    #  Should use separate class for input language (English) and output language (actions)
    #
    def __init__(self, symbols, io_type):
        # symbols : list of all possible symbols
        n = len(symbols)
        self.symbols = [_s.strip('\n') for _s in symbols]
        self.io_type = io_type

        if SOS_token not in self.symbols:
            assert EOS_token not in self.symbols
            self.index2symbol = {n: SOS_token, n+1: EOS_token}
            self.symbol2index = {SOS_token: n, EOS_token: n + 1}
            self.sos_id, self.eos_id = n, n + 1

        else:
            self.index2symbol = {}
            self.symbol2index = {}
            self.sos_id, self.eos_id = 0, 1

        self.pad_token_id = self.sos_id

        for idx,s in enumerate(self.symbols):
            self.index2symbol[idx] = s
            self.symbol2index[s] = idx

        self.n_symbols = len(self.index2symbol)

    def variableFromSymbols(self, mylist, add_eos=True):
        # Convert a list of symbols to a tensor of indices (adding a EOS token at end)
        #
        # Input
        #  mylist : list of m symbols
        #  add_eos : true/false, if true add the EOS symbol at end
        #
        # Output
        #  output : [m or m+1 LongTensor] indices of each symbol (plus EOS if appropriate)
        mylist = copy(mylist)
        if add_eos:
            mylist.append(EOS_token)
        indices = [self.symbol2index[s] for s in mylist]
        output = torch.LongTensor(indices)
        #if USE_CUDA:
        output = output.cuda()
        return output

    def symbolsFromVector(self, v):
        # Convert indices to symbols, breaking where we get a EOS token
        #
        # Input
        #  v : list of m indices
        #
        # Output
        #  mylist : list of m or m-1 symbols (excluding EOS)
        mylist = []
        for x in v:
            s = self.index2symbol[x]
            if s == EOS_token:
                break
            mylist.append(s)
        return mylist

    def encode_scan_file(self, data, max_length):
        encoded_data = []
        for dp in data:
            input, output = dp[0], dp[1]
            if self.io_type == 'input':
                raw = input
            else:
                assert self.io_type == 'output'
                raw = output
            encoded = self.variableFromSymbols(raw.split(' '))
            encoded_data.append(encoded)
        return encoded_data

    def encode_scan_file_2_seg(self, data, max_length, cutoffs):
        encoded_data_1, encoded_data_2 = [], []
        for _id, dp in enumerate(data):
            input, output, cutoff = dp[0], dp[1], cutoffs[_id]
            assert self.io_type == 'output'
            raw = output
            encoded_1 = self.variableFromSymbols(raw.split(' ')[:cutoff])
            encoded_2 = self.variableFromSymbols(raw.split(' ')[cutoff:])
            encoded_data_1.append(encoded_1)
            encoded_data_2.append(encoded_2)
        return encoded_data_1, encoded_data_2

    def encode_cfq_file(self, data, max_length):
        encoded_data = []
        for dp in data:
            input, output = dp['query_ids'], dp['sparql_ids']
            if self.io_type == 'input':
                raw = input
            else:
                assert self.io_type == 'output'
                raw = output + [self.eos_id]
            encoded = torch.LongTensor(raw).cuda()
            encoded_data.append(encoded)
        return encoded_data

    def encode_cogs_file(self, data, max_length):
        encoded_data = []
        for dp in data:
            input, output = dp['src'], dp['trg']
            if self.io_type == 'input':
                raw = input
            else:
                assert self.io_type == 'output'
                raw = output
            encoded = self.variableFromSymbols(raw.split(' '))
            encoded_data.append(encoded)
        return encoded_data

    def decode(self, ids):
        out = self.symbolsFromVector(ids.cpu().numpy())
        if out == []:
            return out
        if out[0] in ['<SOS>', '<SOS_2>']:
            out = out[1:]
        return out


def calculate_accuracy(preds, gts):
    assert len(preds) == len(gts)
    match = 0
    for pred, gt in zip(preds, gts):
        if pred == gt:
            match += 1
    return match / len(preds)

def encode_file(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt", max_examples=None):
    examples = []
    if data_path[-3:] == '.gz':
        print('Data file is gzipped')
        f = gzip.open(data_path, "rt")
    else:
        print('Data file is plain text')
        print(data_path)
        f = open(data_path, "r", encoding='utf-8')

    for i, text in enumerate(f.readlines()):
        tokenized = tokenizer.batch_encode_plus( [text + ' </s>'], max_length=max_length, 
            pad_to_max_length=pad_to_max_length, return_tensors=return_tensors )

        if max_examples and i >= max_examples:
            break
        examples.append(tokenized)

    f.close()
    return examples

# def encode_file_iterator(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt", max_examples=None):
#     '''
#     This provides a low-memory usage way of iterating thru all of the source/target lines for processing by JIT loader.
#     '''
#     if data_path[-3:] == '.gz':
#         print('Data file is gzipped')
#         f = gzip.open(data_path, "rt")
#     else:
#         print('Data file is plain text')
#         f = open(data_path, "r", encoding='utf-8')
#
#     for i, text in enumerate(f):
#
#         tokenized = tokenizer.batch_encode_plus( [text + ' </s>'], max_length=max_length,
#             pad_to_max_length=pad_to_max_length, return_tensors=return_tensors )
#
#         yield tokenized
#
#         if max_examples and i >= max_examples:
#             break
#
#     f.close()

# def convert_scan_actions_to_text(actions):
#     return ' '.join([ACTION_TO_TEXT[_action] for _action in actions.split(' ')])

# def encode_scan_file(tokenizer, data, io_type, max_length, pad_to_max_length=True, return_tensors="pt", max_examples=None):
#     examples = []
#     # a = tokenizer.batch_encode_plus( ['right jump left run walk look' + ' <s> </s>'], max_length=max_length,
#     #         pad_to_max_length=pad_to_max_length, return_tensors=return_tensors )
#     # print(a)
#     # exit()
#     for dp in data:
#         input, output = dp[0], dp[1]
#         if io_type == 'input':
#             raw = input
#         else:
#             assert io_type == 'output'
#             raw = convert_scan_actions_to_text(output)
#
#         tokenized = tokenizer.batch_encode_plus( [raw + ' </s>'], max_length=max_length,
#             pad_to_max_length=pad_to_max_length, return_tensors=return_tensors )
#
#         if max_examples and i >= max_examples:
#             break
#         examples.append(tokenized)
#
#     return examples

def load_scan_file(mytype, split):
    # Load SCAN dataset from file
    #
    # Input
    #  mytype : type of SCAN experiment
    #  split : 'train' or 'test'
    #
    # Output
    #  commands : list of input/output strings (as tuples)
    assert mytype in ['simple', 'addprim_jump', 'length', 'addprim_turn_left', 'all', 'template_around_right', 'viz',
                      'examine', 'template_jump_around_right', 'template_right', 'template_around_right',
                      'mcd1', 'mcd2', 'mcd3', 'mcd1.1', 'mcd1.2', 'debug', 'attn_vis']
    assert split in ['train', 'test', 'val']
    if split == 'val' and mytype not in ['mcd1', 'mcd2', 'mcd3', 'mcd1.1', 'mcd1.2']:
        split = 'test'
    fn = 'data/scan/tasks_' + split + '_' + mytype + '.txt'
    fid = open(fn, 'r')
    lines = fid.readlines()
    fid.close()
    lines = [l.strip() for l in lines]
    lines = [l.lstrip('IN: ') for l in lines]
    commands = [l.split(' OUT: ') for l in lines]
    return commands


class CompositionDataset(Dataset):
    def __init__(
        self,
        src_lang,
        trg_lang,
        data_dir,
        type_path,
        sub_task,
        max_source_length=20,
        max_target_length=20,
        tokenized=False,
    ):
        super().__init__()
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenized = tokenized
        self.src_lang = src_lang
        self.trg_lang = trg_lang

    def __len__(self):
        if self.tokenized:
            return len(self.dataset)
        else:
            return len(self.source)

    def __getitem__(self, index):
        if self.tokenized:
            dp = self.dataset[index]
            source_ids, src_mask, target_ids = dp[0], dp[1], dp[2]
            source_ids = source_ids[:self.max_source_length]
            #src_mask = src_mask[:self.max_source_length]
            target_ids = target_ids[:self.max_target_length]
        else:
            source_ids = self.source[index]
            target_ids = self.target[index]

        return {"source_ids": source_ids, "target_ids": target_ids}

    @staticmethod
    def trim_seq2seq_batch(batch, src_pad_token_id, trg_pad_token_id, trim_y=True):
        if trim_y:
            y = trim_batch(batch["target_ids"], trg_pad_token_id)
        else:
            y = batch["target_ids"]
        source_ids, source_mask = trim_batch(batch["source_ids"], src_pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def pad_to_max_len(self, ids, max_len, pad_token_id):
        ids_length = ids.size(0)
        if ids_length == max_len:
            return ids
        pad_tokens = torch.tensor([pad_token_id] * (max_len - ids_length))
        # if ids.type() == 'torch.cuda.FloatTensor':
        #     print(ids)
        #     exit()
        padded_ids = torch.cat([ids, pad_tokens.cuda()])
        return padded_ids

    def create_mask(self, ids, max_len):
        ids_length = ids.size(0)
        mask = torch.tensor([1] * ids_length + [0] * (max_len - ids_length)).cuda()
        return mask

    def collate_fn(self, batch):
        max_src_len = max(map(len, [x["source_ids"] for x in batch]))
        max_trg_len = max(map(len, [x["target_ids"] for x in batch]))

        src_mask = torch.stack([self.create_mask(x["source_ids"], max_src_len) for x in batch])
        src_ids = torch.stack([self.pad_to_max_len(x["source_ids"], max_src_len, self.src_lang.pad_token_id) for x in batch])
        #masks = torch.stack([x["source_mask"] for x in batch])
        trg_ids = torch.stack([self.pad_to_max_len(x["target_ids"], max_trg_len, self.trg_lang.pad_token_id) for x in batch])

        y = trim_batch(trg_ids, self.trg_lang.pad_token_id)
        src_ids, src_mask = trim_batch(src_ids, self.src_lang.pad_token_id, attention_mask=src_mask)

        return {"source_ids": src_ids, "source_mask": src_mask, "target_ids": y}


class ScanDataset(CompositionDataset):
    def __init__(
        self,
        src_lang,
        trg_lang,
        data_dir="./data/scan/",
        type_path="train",
        sub_task="addprim_jump",
        max_source_length=20,
        max_target_length=20,
        tokenized=False,
    ):
        super().__init__(src_lang, trg_lang, data_dir, type_path, sub_task, max_source_length,
                         max_target_length, tokenized)
        scan_data = load_scan_file(sub_task, type_path)
        print(len(scan_data))
        all_scan_dict = self.convert_to_dict(load_scan_file('all', 'train'))
        self.action_count_labels, self.action_group_labels, self.action_type_labels = self.construct_count_label(scan_data, all_scan_dict)

        if not tokenized:
            self.source = self.src_lang.encode_scan_file(scan_data, max_source_length)
            self.target = self.trg_lang.encode_scan_file(scan_data, max_target_length)
        else:
            self.dataset = torch.load(os.path.join(data_dir, type_path))

    def construct_count_label(self, raw_data, all_data_dict):
        all_count_labels = []
        count_label_scheme = "v1"
        group_label_scheme = "v2"
        type_label_scheme = "v2"
        all_action_group_labels, all_action_type_labels = [], []
        # Group 1: single prim (jump), Group 2: prim + direction (jump left), Group 3: prim opposite, Group 4: prim around

        #no_skip_id = np.random.randint(0, len(raw_data), int(len(raw_data)*0.05))
        #no_skip_id = np.random.choice(range(len(raw_data)), int(len(raw_data)*0.07), replace=False)

        # no_skip_id = np.random.choice(range(len(raw_data)), 10, replace=False)
        skip_cnt, sup_cnt = 0, 0

        for _id, dp in enumerate(raw_data):
            input_text, output_text = dp[0], dp[1]
            input_tok, output_tok = input_text.split(' '), output_text.split(' ')
            count_labels, group_labels, type_labels = [], [], []
            first_part_output_text, second_part_output_text = '', ''
            if 'and' in input_tok:
                first_part_input_tok = input_tok[:input_tok.index('and')]
                second_part_input_tok = input_tok[input_tok.index('and')+1:]
                first_part_output_text = all_data_dict[' '.join(first_part_input_tok)]
                second_part_output_text = all_data_dict[' '.join(second_part_input_tok)]
            elif 'after' in input_tok:
                second_part_input_tok = input_tok[:input_tok.index('after')]
                first_part_input_tok = input_tok[input_tok.index('after') + 1:]
                first_part_output_text = all_data_dict[' '.join(first_part_input_tok)]
                second_part_output_text = all_data_dict[' '.join(second_part_input_tok)]
            else:
                first_part_input_tok, second_part_input_tok = input_tok, []
                first_part_output_text = output_text

            first_part_output_tok, second_part_output_tok = first_part_output_text.split(' '), second_part_output_text.split(' ')
            if second_part_output_text == '':
                second_part_output_tok = []
            assert len(first_part_output_tok) + len(second_part_output_tok) == len(output_tok), \
                (len(first_part_output_tok), len(second_part_output_tok), len(output_tok), first_part_output_text, second_part_output_text, output_text)

            ### 1. Build the action count labels ###
            if count_label_scheme == 'v1':
                ### For the first part output
                if 'twice' in first_part_input_tok:
                    if 'after' in input_tok:
                        count_labels += ([4] * int(len(first_part_output_tok) / 2) + [3] * int(len(first_part_output_tok) / 2))
                    else:
                        count_labels += ([1] * int(len(first_part_output_tok) / 2) + [0] * int(len(first_part_output_tok) / 2))
                    # count_labels += ([1] + [0] * (int(len(first_part_output_tok) / 2) - 1)) * 2
                elif 'thrice' in first_part_input_tok:
                    if 'after' in input_tok:
                        count_labels += ([5] * int(len(first_part_output_tok) / 3) + [4] * int(len(first_part_output_tok) / 3) + \
                                        [3] * int(len(first_part_output_tok) / 3))
                    else:
                        count_labels += ([2] * int(len(first_part_output_tok) / 3) + [1] * int(len(first_part_output_tok) / 3) + \
                                        [0] * int(len(first_part_output_tok) / 3))
                    # count_labels += ([1] + [0] * (int(len(first_part_output_tok) / 3) - 1)) * 3
                else:
                    if 'after' in input_tok:
                        count_labels += ([3] * len(first_part_output_tok))
                    else:
                        count_labels += ([0] * len(first_part_output_tok))
                    # count_labels += ([1] + [0] * (int(len(first_part_output_tok)) - 1))

                ### For the second part output
                if len(second_part_output_tok) > 0:
                    if 'twice' in second_part_input_tok:
                        if 'after' in input_tok:
                            count_labels += ([1] * int(len(second_part_output_tok) / 2) + [0] * int(len(second_part_output_tok) / 2))
                        else:
                            count_labels += ([4] * int(len(second_part_output_tok) / 2) + [3] * int(len(second_part_output_tok) / 2))
                        # count_labels += ([1] + [0] * (int(len(second_part_output_tok) / 2) - 1)) * 2
                    elif 'thrice' in second_part_input_tok:
                        if 'after' in input_tok:
                            count_labels += ([2] * int(len(second_part_output_tok) / 3) + [1] * int(len(second_part_output_tok) / 3) + \
                                             [0] * int(len(second_part_output_tok) / 3))
                        else:
                            count_labels += ([5] * int(len(second_part_output_tok) / 3) + [4] * int(len(second_part_output_tok) / 3) + \
                                             [3] * int(len(second_part_output_tok) / 3))
                        # count_labels += ([1] + [0] * (int(len(second_part_output_tok) / 3) - 1)) * 3
                    else:
                        if 'after' in input_tok:
                            count_labels += ([0] * len(second_part_output_tok))
                        else:
                            count_labels += ([3] * len(second_part_output_tok))
                        # count_labels += ([1] + [0] * (int(len(second_part_output_tok)) - 1))
            elif count_label_scheme == 'v2':
                ### For the first part output
                if 'twice' in first_part_input_tok:
                    count_labels += ([1] * int(len(first_part_output_tok) / 2) + [0] * int(
                            len(first_part_output_tok) / 2))
                elif 'thrice' in first_part_input_tok:
                    count_labels += ([2] * int(len(first_part_output_tok) / 3) + [1] * int(
                            len(first_part_output_tok) / 3) + \
                                         [0] * int(len(first_part_output_tok) / 3))
                else:
                    count_labels += ([0] * len(first_part_output_tok))

                ### For the second part output
                if len(second_part_output_tok) > 0:
                    if 'twice' in second_part_input_tok:
                        count_labels += ([1] * int(len(second_part_output_tok) / 2) + [0] * int(
                                len(second_part_output_tok) / 2))
                    elif 'thrice' in second_part_input_tok:
                        count_labels += ([2] * int(len(second_part_output_tok) / 3) + [1] * int(
                                len(second_part_output_tok) / 3) + [0] * int(len(second_part_output_tok) / 3))
                    else:
                        count_labels += ([0] * len(second_part_output_tok))
            elif count_label_scheme == 'v3':
                ### For the first part output
                if 'thrice' in first_part_input_tok and 'thrice' in second_part_input_tok:
                    start_count = 5
                elif ('thrice' in first_part_input_tok and 'twice' in second_part_input_tok) or \
                    ('twice' in first_part_input_tok and 'thrice' in second_part_input_tok):
                    start_count = 4
                elif ('twice' in first_part_input_tok and 'twice' in second_part_input_tok) or \
                    ('thrice' in first_part_input_tok) or ('thrice' in second_part_input_tok):
                    start_count = 3
                elif 'twice' in first_part_input_tok or 'twice' in second_part_input_tok:
                    start_count = 2
                else:
                    start_count = 1

                if 'twice' in first_part_input_tok:
                    if 'after' in input_tok:
                        count_labels += ([start_count] * int(len(first_part_output_tok) / 2) + [start_count-1] * int(len(first_part_output_tok) / 2))
                    else:
                        count_labels += ([1] * int(len(first_part_output_tok) / 2) + [0] * int(len(first_part_output_tok) / 2))
                    # count_labels += ([1] + [0] * (int(len(first_part_output_tok) / 2) - 1)) * 2
                elif 'thrice' in first_part_input_tok:
                    if 'after' in input_tok:
                        count_labels += ([start_count] * int(len(first_part_output_tok) / 3) + [start_count-1] * int(len(first_part_output_tok) / 3) + \
                                        [start_count-2] * int(len(first_part_output_tok) / 3))
                    else:
                        count_labels += ([2] * int(len(first_part_output_tok) / 3) + [1] * int(len(first_part_output_tok) / 3) + \
                                        [0] * int(len(first_part_output_tok) / 3))
                    # count_labels += ([1] + [0] * (int(len(first_part_output_tok) / 3) - 1)) * 3
                else:
                    if 'after' in input_tok:
                        count_labels += ([start_count] * len(first_part_output_tok))
                    else:
                        count_labels += ([0] * len(first_part_output_tok))
                    # count_labels += ([1] + [0] * (int(len(first_part_output_tok)) - 1))

                ### For the second part output
                if len(second_part_output_tok) > 0:
                    if 'twice' in second_part_input_tok:
                        if 'after' in input_tok:
                            count_labels += ([1] * int(len(second_part_output_tok) / 2) + [0] * int(len(second_part_output_tok) / 2))
                        else:
                            count_labels += ([start_count] * int(len(second_part_output_tok) / 2) + [start_count-1] * int(len(second_part_output_tok) / 2))
                        # count_labels += ([1] + [0] * (int(len(second_part_output_tok) / 2) - 1)) * 2
                    elif 'thrice' in second_part_input_tok:
                        if 'after' in input_tok:
                            count_labels += ([2] * int(len(second_part_output_tok) / 3) + [1] * int(len(second_part_output_tok) / 3) + \
                                             [0] * int(len(second_part_output_tok) / 3))
                        else:
                            count_labels += ([start_count] * int(len(second_part_output_tok) / 3) + [start_count-1] * int(len(second_part_output_tok) / 3) + \
                                             [start_count-2] * int(len(second_part_output_tok) / 3))
                        # count_labels += ([1] + [0] * (int(len(second_part_output_tok) / 3) - 1)) * 3
                    else:
                        if 'after' in input_tok:
                            count_labels += ([0] * len(second_part_output_tok))
                        else:
                            count_labels += ([start_count] * len(second_part_output_tok))
                        # count_labels += ([1] + [0] * (int(len(second_part_output_tok)) - 1))
            elif count_label_scheme == 'v3.1':
                ### For the first part output
                if 'thrice' in first_part_input_tok and 'thrice' in second_part_input_tok:
                    start_count = 5
                elif ('thrice' in first_part_input_tok and 'twice' in second_part_input_tok) or \
                        ('twice' in first_part_input_tok and 'thrice' in second_part_input_tok):
                    start_count = 4
                elif ('twice' in first_part_input_tok and 'twice' in second_part_input_tok) or \
                        ('thrice' in first_part_input_tok) or ('thrice' in second_part_input_tok):
                    start_count = 3
                elif 'twice' in first_part_input_tok or 'twice' in second_part_input_tok:
                    start_count = 2
                else:
                    start_count = 1

                if 'twice' in first_part_input_tok:
                    count_labels += ([start_count] * int(len(first_part_output_tok) / 2) + [start_count - 1] * int(
                            len(first_part_output_tok) / 2))

                    # count_labels += ([1] + [0] * (int(len(first_part_output_tok) / 2) - 1)) * 2
                elif 'thrice' in first_part_input_tok:
                    count_labels += ([start_count] * int(len(first_part_output_tok) / 3) + [start_count - 1] * int(
                        len(first_part_output_tok) / 3) + \
                                     [start_count - 2] * int(len(first_part_output_tok) / 3))
                else:
                    count_labels += ([start_count] * len(first_part_output_tok))

                ### For the second part output
                if len(second_part_output_tok) > 0:
                    if 'twice' in second_part_input_tok:
                            count_labels += ([1] * int(len(second_part_output_tok) / 2) + [0] * int(
                                len(second_part_output_tok) / 2))
                        # count_labels += ([1] + [0] * (int(len(second_part_output_tok) / 2) - 1)) * 2
                    elif 'thrice' in second_part_input_tok:
                        count_labels += ([2] * int(len(second_part_output_tok) / 3) + [1] * int(
                            len(second_part_output_tok) / 3) + \
                                         [0] * int(len(second_part_output_tok) / 3))
                    else:
                        count_labels += ([0] * len(second_part_output_tok))

            else:
                ### For the first part output
                if 'twice' in first_part_input_tok:
                    if 'after' in input_tok:
                        new_count_labels = list(range(int(len(first_part_output_tok) / 2)))[::-1] * 2
                    else:
                        new_count_labels = list(range(int(len(first_part_output_tok) / 2)))[::-1] * 2
                elif 'thrice' in first_part_input_tok:
                    if 'after' in input_tok:
                        new_count_labels = list(range(int(len(first_part_output_tok) / 3)))[::-1] * 3
                    else:
                        new_count_labels = list(range(int(len(first_part_output_tok) / 3)))[::-1] * 3
                else:
                    if 'after' in input_tok:
                        new_count_labels = list(range(len(first_part_output_tok)))[::-1]
                    else:
                        new_count_labels = list(range(len(first_part_output_tok)))[::-1]

                count_labels += new_count_labels

                ### For the second part output
                if len(second_part_output_tok) > 0:
                    if 'twice' in second_part_input_tok:
                        if 'after' in input_tok:
                            new_count_labels = list(range(int(len(second_part_output_tok) / 2)))[::-1] * 2
                            new_count_labels = [_c + 8 for _c in new_count_labels]
                        else:
                            new_count_labels = list(range(int(len(second_part_output_tok) / 2)))[::-1] * 2
                            new_count_labels = [_c + 8 for _c in new_count_labels]
                    elif 'thrice' in second_part_input_tok:
                        if 'after' in input_tok:
                            new_count_labels = list(range(int(len(second_part_output_tok) / 3)))[::-1] * 3
                            new_count_labels = [_c + 8 for _c in new_count_labels]
                        else:
                            new_count_labels = list(range(int(len(second_part_output_tok) / 3)))[::-1] * 3
                            new_count_labels = [_c + 8 for _c in new_count_labels]
                    else:
                        if 'after' in input_tok:
                            new_count_labels = list(range(len(second_part_output_tok)))[::-1]
                            new_count_labels = [_c + 8 for _c in new_count_labels]
                        else:
                            new_count_labels = list(range(len(second_part_output_tok)))[::-1]
                            new_count_labels = [_c + 8 for _c in new_count_labels]

                    count_labels += new_count_labels

            # count_labels = []
            # count_labels += list(range(len(first_part_output_tok)))[::-1]
            # count_labels += list(range(len(second_part_output_tok)))[::-1]
            assert len(count_labels) == len(output_tok), (len(count_labels), len(output_tok), input_text, first_part_input_tok, count_labels, output_tok,
                                                          first_part_output_text, first_part_output_tok, second_part_output_text, second_part_output_tok)
            count_labels.append(-1) # For the EOS token
            # count_labels.append(7)  # For the EOS token

            ### 2. Build the action group labels ###
            if group_label_scheme == 'v1': ## As used in exp 9.0-9.4
                if 'around' in first_part_input_tok:
                    if 'after' in input_tok:
                        group_labels += ([4] * len(first_part_output_tok))
                    else:
                        group_labels += ([0] * len(first_part_output_tok))
                elif 'opposite' in first_part_input_tok:
                    if 'after' in input_tok:
                        group_labels += ([5] * len(first_part_output_tok))
                    else:
                        group_labels += ([1] * len(first_part_output_tok))
                elif 'left' in first_part_input_tok or 'right' in first_part_input_tok:
                    if 'after' in input_tok:
                        group_labels += ([6] * len(first_part_output_tok))
                    else:
                        group_labels += ([2] * len(first_part_output_tok))
                else:
                    if 'after' in input_tok:
                        group_labels += ([7] * len(first_part_output_tok))
                    else:
                        group_labels += ([3] * len(first_part_output_tok))

                if 'around' in second_part_input_tok:
                    if 'after' in input_tok:
                        group_labels += ([0] * len(second_part_output_tok))
                    else:
                        group_labels += ([4] * len(second_part_output_tok))
                elif 'opposite' in second_part_input_tok:
                    if 'after' in input_tok:
                        group_labels += ([1] * len(second_part_output_tok))
                    else:
                        group_labels += ([5] * len(second_part_output_tok))
                elif 'left' in second_part_input_tok or 'right' in second_part_input_tok:
                    if 'after' in input_tok:
                        group_labels += ([2] * len(second_part_output_tok))
                    else:
                        group_labels += ([6] * len(second_part_output_tok))
                else:
                    if 'after' in input_tok:
                        group_labels += ([3] * len(second_part_output_tok))
                    else:
                        group_labels += ([7] * len(second_part_output_tok))
            else:
                ### For the first part output
                if 'twice' in first_part_input_tok:
                    if 'after' in input_tok:
                        new_group_labels = list(range(int(len(first_part_output_tok) / 2)))[::-1] * 2
                        new_group_labels = [_c + 8 for _c in new_group_labels]
                    else:
                        new_group_labels = list(range(int(len(first_part_output_tok) / 2)))[::-1] * 2
                elif 'thrice' in first_part_input_tok:
                    if 'after' in input_tok:
                        new_group_labels = list(range(int(len(first_part_output_tok) / 3)))[::-1] * 3
                        new_group_labels = [_c + 8 for _c in new_group_labels]
                    else:
                        new_group_labels = list(range(int(len(first_part_output_tok) / 3)))[::-1] * 3
                else:
                    if 'after' in input_tok:
                        new_group_labels = list(range(len(first_part_output_tok)))[::-1]
                        new_group_labels = [_c + 8 for _c in new_group_labels]
                    else:
                        new_group_labels = list(range(len(first_part_output_tok)))[::-1]

                group_labels += new_group_labels

                ### For the second part output
                if len(second_part_output_tok) > 0:
                    if 'twice' in second_part_input_tok:
                        if 'after' in input_tok:
                            new_group_labels = list(range(int(len(second_part_output_tok) / 2)))[::-1] * 2
                        else:
                            new_group_labels = list(range(int(len(second_part_output_tok) / 2)))[::-1] * 2
                            new_group_labels = [_c + 8 for _c in new_group_labels]
                    elif 'thrice' in second_part_input_tok:
                        if 'after' in input_tok:
                            new_group_labels = list(range(int(len(second_part_output_tok) / 3)))[::-1] * 3
                        else:
                            new_group_labels = list(range(int(len(second_part_output_tok) / 3)))[::-1] * 3
                            new_group_labels = [_c + 8 for _c in new_group_labels]
                    else:
                        if 'after' in input_tok:
                            new_group_labels = list(range(len(second_part_output_tok)))[::-1]
                        else:
                            new_group_labels = list(range(len(second_part_output_tok)))[::-1]
                            new_group_labels = [_c + 8 for _c in new_group_labels]

                    group_labels += new_group_labels

            assert len(group_labels) == len(output_tok)
            group_labels.append(-1)  # For the EOS token
            # group_labels.append(17)  # For the EOS token

            ### 3. Build the action type labels ###
            ### For the first part output
            if type_label_scheme == 'v1':
                if 'around' in first_part_input_tok:
                    new_type_labels = [3] * len(first_part_output_tok)
                elif 'opposite' in first_part_input_tok:
                    new_type_labels = [2] * len(first_part_output_tok)
                elif 'left' in first_part_input_tok or 'right' in first_part_input_tok:
                    new_type_labels = [1] * len(first_part_output_tok)
                else:
                    new_type_labels = [0] * len(first_part_output_tok)

                # if 'after' in input_tok:
                #     new_type_labels = [_c + 4 for _c in new_type_labels]
                type_labels += new_type_labels

                ### For the second part output
                if len(second_part_output_tok) > 0:
                    if 'around' in second_part_input_tok:
                        new_type_labels = [3] * len(second_part_output_tok)
                    elif 'opposite' in second_part_input_tok:
                        new_type_labels = [2] * len(second_part_output_tok)
                    elif 'left' in second_part_input_tok or 'right' in second_part_input_tok:
                        new_type_labels = [1] * len(second_part_output_tok)
                    else:
                        new_type_labels = [0] * len(second_part_output_tok)

                    # if 'after' not in input_tok:
                    #     new_type_labels = [_c + 4 for _c in new_type_labels]
                    type_labels += new_type_labels
            elif type_label_scheme == 'v2':
                if 'twice' in first_part_input_tok:
                    type_labels += ([1] * int(len(first_part_output_tok) / 2) + [0] * int(
                            len(first_part_output_tok) / 2))
                elif 'thrice' in first_part_input_tok:
                    type_labels += ([2] * int(len(first_part_output_tok) / 3) + [1] * int(
                            len(first_part_output_tok) / 3) + \
                                         [0] * int(len(first_part_output_tok) / 3))
                else:
                    type_labels += ([0] * len(first_part_output_tok))

                ### For the second part output
                if len(second_part_output_tok) > 0:
                    if 'twice' in second_part_input_tok:
                        type_labels += ([1] * int(len(second_part_output_tok) / 2) + [0] * int(
                                len(second_part_output_tok) / 2))
                    elif 'thrice' in second_part_input_tok:
                        type_labels += ([2] * int(len(second_part_output_tok) / 3) + [1] * int(
                                len(second_part_output_tok) / 3) + [0] * int(len(second_part_output_tok) / 3))
                    else:
                        type_labels += ([0] * len(second_part_output_tok))

            assert len(type_labels) == len(output_tok)
            type_labels.append(-1)  # For the EOS token
            # group_labels.append(17)  # For the EOS token

            # if _id not in no_skip_id:
            #     count_labels = [-1] * len(count_labels)
            #     group_labels = [-1] * len(group_labels)
            #     skip_cnt += 1
            # else:
            #     sup_cnt += 1

            all_action_type_labels.append(torch.tensor(type_labels).cuda())
            all_count_labels.append(torch.tensor(count_labels).cuda())
            all_action_group_labels.append(torch.tensor(group_labels).cuda())
        print(skip_cnt, sup_cnt)

        return all_count_labels, all_action_group_labels, all_action_type_labels

    def convert_to_dict(self, raw_data):
        dict_data = {}
        for dp in raw_data:
            input, output = dp[0], dp[1]
            assert input not in dict_data
            dict_data[input] = output
        return dict_data

    def __getitem__(self, index):
        if self.tokenized:
            dp = self.dataset[index]
            source_ids, src_mask, target_ids = dp[0], dp[1], dp[2]
            source_ids = source_ids[:self.max_source_length]
            #src_mask = src_mask[:self.max_source_length]
            target_ids = target_ids[:self.max_target_length]
        else:
            source_ids = self.source[index]
            target_ids = self.target[index]
            count_labels = self.action_count_labels[index]
            group_labels = self.action_group_labels[index]
            type_labels = self.action_type_labels[index]

        return {"source_ids": source_ids, "target_ids": target_ids, "action_count_labels": count_labels,
                "action_group_labels": group_labels, "action_type_labels": type_labels}

    @staticmethod
    def trim_seq2seq_batch(batch, src_pad_token_id, trg_pad_token_id, trim_y=True):
        if trim_y:
            y = trim_batch(batch["target_ids"], trg_pad_token_id)
        else:
            y = batch["target_ids"]
        source_ids, source_mask = trim_batch(batch["source_ids"], src_pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        max_src_len = max(map(len, [x["source_ids"] for x in batch]))
        max_trg_len = max(map(len, [x["target_ids"] for x in batch]))

        src_mask = torch.stack([self.create_mask(x["source_ids"], max_src_len) for x in batch])
        trg_mask = torch.stack([self.create_mask(x["target_ids"], max_trg_len) for x in batch])
        src_ids = torch.stack([self.pad_to_max_len(x["source_ids"], max_src_len, self.src_lang.pad_token_id) for x in batch])
        #masks = torch.stack([x["source_mask"] for x in batch])
        trg_ids = torch.stack([self.pad_to_max_len(x["target_ids"], max_trg_len, self.trg_lang.pad_token_id) for x in batch])
        action_count_labels = torch.stack([self.pad_to_max_len(x["action_count_labels"], max_trg_len, -1) for x in batch])
        action_group_labels = torch.stack([self.pad_to_max_len(x["action_group_labels"], max_trg_len, -1) for x in batch])
        action_type_labels = torch.stack(
            [self.pad_to_max_len(x["action_type_labels"], max_trg_len, -1) for x in batch])

        y = trim_batch(trg_ids, self.trg_lang.pad_token_id)
        #action_count_labels = trim_batch(action_count_labels, -1)

        # _src_ids, src_mask = trim_batch(src_ids, self.src_lang.pad_token_id, attention_mask=src_mask)
        # print(_src_ids.size(), src_ids.size())
        return {"source_ids": src_ids, "source_mask": src_mask, "target_ids": y, "target_mask": trg_mask,
                "action_count_labels": action_count_labels, "action_group_labels": action_group_labels,
                "action_type_labels": action_type_labels}
