# coding:utf-8
import math
import json
from tqdm import tqdm
import pickle
import torch
from torch.utils import data
import numpy as np
from dataset_utils import tokenize
from transformers import BertTokenizer
# print('Loading bert_tokenizer ...')
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_tokenizer.add_special_tokens({"additional_special_tokens": ["[unused0]", "[unused10]", "[unused1]", "[unused2]", "[unused3]", "[unused4]", "[unused5]", "[unused6]", "[unused7]", "[unused8]", "[unused9]"] })
#import nltk
#from nltk.stem import WordNetLemmatizer
#wlem = WordNetLemmatizer()


def get_entity_mask(src_ids, entity_ids):
    mask = []
    idx_src_s, idx_src_e = 0, 0
    while idx_src_s < len(src_ids):
        idx_ent = 0
        if src_ids[idx_src_s] == entity_ids[idx_ent]:
            idx_src_e = idx_src_s
            idx_ent = 0
            while (
                idx_src_e < len(src_ids)
                and idx_ent < len(entity_ids)
                and src_ids[idx_src_e] == entity_ids[idx_ent]
            ):
                idx_src_e += 1
                idx_ent += 1
            if idx_ent == len(entity_ids):  # matched
                mask += [1] * len(entity_ids)
                idx_src_s = idx_src_e
            else:
                mask.append(0)
                idx_src_s += 1
        else:
            mask.append(0)
            idx_src_s += 1
    assert len(mask) == len(src_ids)
    return mask


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def replace_speaker(s, e1="", e2=""):
    res = []
    for itm in s.split(): 
        if itm.startswith("speaker") and itm[7:].isdigit():
            if itm == e1:
                new_itm = "[unused0]"
            elif itm == e2:
                new_itm = "[unused10]"
            else:
                new_itm = "[unused{}]".format(itm[7:])
            res.append(new_itm)
        else:
            res.append(itm)
    return " ".join(res)


def convert_input_con(word_lst, con_id, con_path_id, e1, e2, vocab, max_length=512, word2node=None):
    assert len(con_path_id) == len(con_id) and len(con_path_id[0]) == len(con_id), 'Inconsistent lengths, {} vs {}x{}'.format(len(con_id), len(con_path_id), len(con_path_id[0]))
    input_len = min(len(con_id), max_length)
    con_id = con_id[:input_len]
    word_lst = word_lst.replace(" ##", '')
    word_id = tokenize(vocab, word_lst, 1, 1, dtype="concept")[:-1]
    e1_id = tokenize(vocab, e1.lower(), 1, 1, dtype="concept")[:-1]
    e2_id = tokenize(vocab, e2.lower(), 1, 1, dtype="concept")[:-1]
    mask_e1_word = get_entity_mask(word_id, e1_id)
    mask_e2_word = get_entity_mask(word_id, e2_id)
    mask_e1_con = [0 for _ in range(input_len)]
    mask_e2_con = [0 for _ in range(input_len)]
    for idx, itm in enumerate(mask_e1_word):
        if itm == 1:
            if str(idx) in word2node:
                con_idx = word2node[str(idx)]
                assert con_idx < input_len
                mask_e1_con[con_idx] = 1
                print('e1: {} in concepts'.format(word_lst.split()[idx]))
            else:
                # print('{} not in concepts'.format(word_lst.split()[idx]))
                pass
    
    for idx, itm in enumerate(mask_e2_word):
        if itm == 1:
            if str(idx) in word2node:
                con_idx = word2node[str(idx)]
                assert con_idx < input_len
                mask_e2_con[con_idx] = 1
                print('e2: {} in concepts'.format(word_lst.split()[idx]))
            else:
                # print('{} not in concepts'.format(word_lst.split()[idx]))
                pass

    while len(con_id) < max_length:
        con_id.append(0)
        mask_e1_con.append(0)
        mask_e2_con.append(0)
    pad_path = [
        [0 for i in range(max_length)] for j in range(max_length)
    ]
    for j in range(input_len):
        pad_path[j][:input_len] = map(int, con_path_id[j])
    return con_id, input_len, pad_path, mask_e1_con, mask_e2_con


def convert_input_con2(con_lst, con_path_id, e1, e2, con_vocab, max_length=512, use_lemma=False):
    con_id = tokenize(con_vocab, con_lst, 1, 1, dtype="concept")
    assert len(con_path_id) == len(con_id) and len(con_path_id[0]) == len(con_id), 'Inconsistent lengths, {} vs {}x{}'.format(len(con_id), len(con_path_id), len(con_path_id[0]))
    input_len = min(len(con_id), max_length)
    con_id = con_id[:input_len]
    if use_lemma:
        e1_lemma = ' '.join([wlem.lemmatize(itm, 'n') for itm in e1.lower().split()])
        e2_lemma = ' '.join([wlem.lemmatize(itm, 'n') for itm in e2.lower().split()])
    else:
        e1_lemma = e1
        e2_lemma = e2
    e1_id = tokenize(con_vocab, e1_lemma.lower(), 1, 1, dtype="concept")[:-1]
    e2_id = tokenize(con_vocab, e2_lemma.lower(), 1, 1, dtype="concept")[:-1]
    
    mask_e1_con = get_entity_mask(con_id, e1_id)
    mask_e2_con = get_entity_mask(con_id, e2_id)
    # if sum(mask_e1_con) == 0:
    #     print('e1: ori:{}, lemma:{} not in concept'.format(e1,e1_lemma))
    # if sum(mask_e2_con) == 0:
    #     print('e2: ori:{}, lemma:{} not in concept'.format(e2, e2_lemma))

    while len(con_id) < max_length:
        con_id.append(0)
        mask_e1_con.append(0)
        mask_e2_con.append(0)
    
    pad_path = [
        [0 for i in range(max_length)] for j in range(max_length)
    ]
    for j in range(input_len):
        pad_path[j][:input_len] = map(int, con_path_id[j])
    # print('con_id', con_id, len(con_id))
    # print('inp_len', input_len)
    # print('pad_path', len(pad_path), len(pad_path[0]))
    return con_id, input_len, pad_path, mask_e1_con, mask_e2_con


def convert_input(src, e1, e2, word_mask_id=None, word_rel_id=None, max_seq_length=512, max_d_len=491):
    e1_new = replace_speaker(e1, e1, e2)
    e2_new = replace_speaker(e2, e1, e2)
    src = replace_speaker(src, e1, e2)
    tokens_a = src.split()
    _tokens_b = bert_tokenizer.tokenize(e1_new)
    _tokens_c = bert_tokenizer.tokenize(e2_new)

    # _truncate_seq_tuple(tokens_a, _tokens_b, _tokens_c, max_seq_length - 4)
    tokens_b = _tokens_b + ["[SEP]"] + _tokens_c

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    input_len = len(tokens)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    assert len(input_ids) <= max_seq_length, "Invalid instance {}, length: {}".format(' '.join(tokens), len(input_ids))
    b_ids = bert_tokenizer.convert_tokens_to_ids(_tokens_b)
    c_ids = bert_tokenizer.convert_tokens_to_ids(_tokens_c)

    mask_b = get_entity_mask(input_ids[: len(tokens_a) + 1], b_ids)
    mask_c = get_entity_mask(input_ids[: len(tokens_a) + 1], c_ids)

    while len(mask_b) < len(input_ids):
        mask_b.append(0)
        mask_c.append(0)
    
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        mask_b.append(0)
        mask_c.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(mask_b) == max_seq_length
    assert len(mask_c) == max_seq_length
    
    # if sum(mask_b) == 0:
    #     print('instance: ', ' '.join(tokens))
    #     print('all inp ids:', ' '.join([str(itm) for itm in input_ids]))
    #     print('a ids:', ' '.join([str(itm) for itm in b_ids]))
    #     print('b ids:', ' '.join([str(itm) for itm in c_ids]))
    #     print("Lengths:,", len(tokens_a) + len(_tokens_b) + len(_tokens_c))
    # if sum(mask_b) == 0:
    #     print("Invalid instance: {}".format(' '.join(tokens)))
    #     print("Lengths:,", len(tokens_a) + len(_tokens_b) + len(_tokens_c))
    # if sum(mask_c) == 0:
    #     print("Invalid instance: {}".format(' '.join(tokens)))
    #     print("Lengths:,", len(tokens_a) + len(_tokens_b) + len(_tokens_c))

    if word_mask_id is not None and word_rel_id is not None:
        pad_rel_mask = [
            [0 for i in range(max_d_len)] for j in range(max_d_len)
        ]
        pad_wr = [
            [0 for i in range(max_d_len)] for j in range(max_d_len)
        ]

        ith_mask_len = len(word_mask_id)
        for j in range(ith_mask_len):
            mask_len_j = len(word_mask_id[j])
            pad_rel_mask[j][:mask_len_j] = map(int, word_mask_id[j])

        ith_wr_len = len(word_rel_id)
        for j in range(ith_wr_len):
            wr_len_j = len(word_rel_id[j])
            pad_wr[j][:wr_len_j] = map(int, word_rel_id[j])
    else:
        pad_rel_mask, pad_wr = None, None

    return input_ids, input_len, input_mask, segment_ids, mask_b, mask_c, pad_rel_mask, pad_wr


class AMRDialogDataSet():
    def __init__(self, data_path, tokenize_fn, word_rel_vocab, word_vocab=None, concept_vocab=None, path_vocab=None, save_path='', data_type='std'):
        self.word_inp_lst, self.context_len_lst, self.inp_mask_lst, self.seg_mask_lst, self.e1_mask_lst, self.e2_mask_lst, self.rel_mask_lst, self.wr_lst, self.rid_lst = [], [], [], [], [], [], [], [], []
        self.concept_lst, self.concept_len_lst, self.path_lst, self.e1_mask_con_lst, self.e2_mask_con_lst = [], [], [], [], []
        self.word_rel_vocab = word_rel_vocab
        self.word_vocab = word_vocab
        self.concept_vocab = concept_vocab
        self.path_vocab = path_vocab
        self.tokenize_fn = tokenize_fn
        
        with open(data_path + '.src.spm', 'r', encoding='utf-8') as srcf:
            src_lines = srcf.readlines()
        with open(data_path + '.concept', 'r', encoding='utf-8') as conf:
            concept_lines = conf.readlines()
        with open(data_path + '.path', 'r', encoding='utf-8') as pathf:
            path_lines = pathf.readlines()
        with open(data_path + '.mask', 'r', encoding='utf-8') as maskf:
            mask_lines = maskf.readlines()
        with open(data_path + '.rel', 'r', encoding='utf-8') as word_relf:
            word_rel_lines = word_relf.readlines()
        RE_data = json.load(open(data_path + '.re.json', 'r', encoding='utf-8'))
        self.build_data(src_lines, concept_lines, path_lines, mask_lines, word_rel_lines, RE_data, 0, save_path, data_type)

    def __len__(self):
        return len(self.rid_lst)

    def cal_max_len(self, ids, curdepth, maxdepth):
        """calculate max sequence length"""
        assert curdepth <= maxdepth
        if isinstance(ids[0], list):
            res = max([self.cal_max_len(k, curdepth + 1, maxdepth) for k in ids])
        else:
            res = len(ids)
        return res
    
    def build_data(self, srcf, conf, pathf, maskf, word_relf, RE_data, id, save_path, dtype='std'):
        word_inp_lst, context_len_lst, inp_mask_lst, seg_mask_lst, e1_mask_lst, e2_mask_lst, rel_mask_lst, wr_lst, rid_lst = [], [], [], [], [], [], [], [], []
        con_inp_lst, concept_len_lst, con_path_lst, e1_mask_con_lst, e2_mask_con_lst = [], [], [], [], []
        if not id:
            pbar = tqdm(total=len(RE_data))
        all_res, idx, total, split_idx = 0, 0, len(srcf), 0                               # ith dialog history
        for re_data in RE_data:
            dialog_itm, re_item =  re_data[0], re_data[1]
            dialog_len = len(re_data[0]) if dtype != 'std' else 1
            assert idx + dialog_len <= total, 'Error index {} out of range {}'.format(idx + dialog_len, total)
            for j in range(len(re_item)):                       # 遍历 relation
                rid = []
                for k in range(36):
                    if k + 1 in re_item[j]["rid"]:
                        rid += [1]
                    else:
                        rid += [0]
                e1, e2 = re_item[j]["x"].lower().replace("speaker ", "speaker"), re_item[j]["y"].lower().replace("speaker ", "speaker")
                for src_tok, con_tok, con_path, word_rel_mask, word_rel in zip(srcf[idx: idx + dialog_len], conf[idx: idx + dialog_len], pathf[idx: idx + dialog_len], maskf[idx: idx + dialog_len], word_relf[idx: idx + dialog_len]):     # 遍历 utterace 
                    src_tok = src_tok.strip().replace(' <sep>', '')
                    word_mask_lst = [itm.split(' ') for itm in word_rel_mask.strip().split('\t')]
                    word_rel_lst = [itm for itm in word_rel.strip().split('\t')]
                    word_mask_id = [[int(iitm) for iitm in itm] for itm in word_mask_lst]
                    word_rel_id = self.tokenize_fn(self.word_rel_vocab, word_rel_lst, 1, 2, dtype="relation")
                    assert len(src_tok.split()) + 1 == len(word_mask_id[0]) and len(src_tok.split()) + 1 == len(word_rel_id[0]), "word_len: {}, mask_size:{} x {}, rel_size: {} x {}".format(len(src_tok.split()+1), len(word_mask_id), len(word_mask_id[0]), len(word_rel_id), len(word_rel_id[0]))
                    con_lst = con_tok.strip().lower()
                    con_path_lst_raw = con_path.strip().split()
                    seg_len = int(math.sqrt(len(con_path_lst_raw)))
                    assert seg_len * seg_len == len(con_path_lst_raw)
                    con_path_lst_s = [
                        ' '.join(con_path_lst_raw[i: i + seg_len])
                        for i in range(0, len(con_path_lst_raw), seg_len)
                    ]
                    con_path_id = self.tokenize_fn(self.path_vocab, con_path_lst_s, 1, 2, dtype="relation")
                    
                    new_inp, context_len, input_mask, seg_mask, e1_mask, e2_mask, rel_mask, wr = convert_input(src_tok, e1, e2, word_mask_id, word_rel_id, max_seq_length=512, max_d_len=491)
                    new_con, new_con_len, new_path_id, e1_mask_con, e2_mask_con = convert_input_con2(con_lst, con_path_id, e1, e2, self.concept_vocab, max_length=512)                
                    word_inp_lst.append(new_inp)
                    con_inp_lst.append(new_con)
                    context_len_lst.append(context_len)
                    concept_len_lst.append(new_con_len)
                    inp_mask_lst.append(input_mask)
                    seg_mask_lst.append(seg_mask)
                    e1_mask_lst.append(e1_mask)
                    e2_mask_lst.append(e2_mask)
                    rel_mask_lst.append(rel_mask)
                    wr_lst.append(wr)
                    con_path_lst.append(new_path_id)
                    e1_mask_con_lst.append(e1_mask_con)
                    e2_mask_con_lst.append(e2_mask_con)
                    rid_lst.append(rid)
                    all_res += 1
                    if len(rid_lst) % 5000 == 0:
                        # tmp_res = [word_inp_lst[len(rid_lst)-5000: len(rid_lst) - 1], context_len_lst[len(rid_lst)-5000: len(rid_lst) - 1], inp_mask_lst[len(rid_lst)-5000: len(rid_lst) - 1], seg_mask_lst[len(rid_lst)-5000: len(rid_lst) - 1], e1_mask_lst[len(rid_lst)-5000: len(rid_lst) - 1], e2_mask_lst[len(rid_lst)-5000: len(rid_lst) - 1], rel_mask_lst[len(rid_lst)-5000: len(rid_lst) - 1], wr_lst[len(rid_lst)-5000: len(rid_lst) - 1], rid_lst[len(rid_lst)-5000: len(rid_lst) - 1], con_inp_lst[len(rid_lst)-5000: len(rid_lst) - 1], concept_len_lst[len(rid_lst)-5000: len(rid_lst) - 1], con_path_lst[len(rid_lst)-5000: len(rid_lst) - 1], e1_mask_con_lst[len(rid_lst)-5000: len(rid_lst) - 1], e2_mask_con_lst[len(rid_lst)-5000: len(rid_lst) - 1]]
                        tmp_res = [word_inp_lst, context_len_lst, inp_mask_lst, seg_mask_lst, e1_mask_lst, e2_mask_lst, rel_mask_lst, wr_lst, rid_lst, con_inp_lst, concept_len_lst, con_path_lst, e1_mask_con_lst, e2_mask_con_lst]
                        pkl_file = "{:s}-{}.pkl".format(save_path, split_idx)
                        pickle.dump(tmp_res, open(pkl_file, 'wb'))
                        word_inp_lst, context_len_lst, inp_mask_lst, seg_mask_lst, e1_mask_lst, e2_mask_lst, rel_mask_lst, wr_lst, rid_lst = [], [], [], [], [], [], [], [], []
                        con_inp_lst, concept_len_lst, con_path_lst, e1_mask_con_lst, e2_mask_con_lst = [], [], [], [], []
                        split_idx += 1
            idx += dialog_len
            if not id:
                pbar.update(1)
        tmp_res = [word_inp_lst, context_len_lst, inp_mask_lst, seg_mask_lst, e1_mask_lst, e2_mask_lst, rel_mask_lst, wr_lst, rid_lst, con_inp_lst, concept_len_lst, con_path_lst, e1_mask_con_lst, e2_mask_con_lst]
        pkl_file = "{:s}-{}.pkl".format(save_path, split_idx)
        pickle.dump(tmp_res, open(pkl_file, 'wb'))
        if not id:
            pbar.close()
        print('All {} instances'.format(all_res))
        return
