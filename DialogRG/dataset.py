# coding:utf-8
import math
from torch.utils import data
from dataset_utils import load_file, load_json_file
import numpy as np
from transformers import BertTokenizer
# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class DialogSet(data.Dataset):
    def __init__(self, data_path, tokenize_fn, vocab):
        data = load_file(data_path)
        self.instance = []
        for index in range(len(data)):
            src, tgt, filter_knowledge = data[index]
            src_id = tokenize_fn(vocab, src, 1, 1)
            tgt_id = tokenize_fn(vocab, tgt, 1, 1)
            filter_knowledge_id = tokenize_fn(vocab, filter_knowledge, 1, 2)
            self.instance.append((src_id, tgt_id, filter_knowledge_id, tgt))

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, index):
        src_id, tgt_id, filter_knowledge_id, tgt = self.instance[index]
        return src_id, tgt_id, filter_knowledge_id, tgt

    def cal_max_len(self, ids, curdepth, maxdepth):
        """calculate max sequence length"""
        assert curdepth <= maxdepth
        if isinstance(ids[0], list):
            res = max([self.cal_max_len(k, curdepth + 1, maxdepth) for k in ids])
        else:
            res = len(ids)
        return res
    
    def collate_fn(self, batch):
        sv, tv, kv, tstr = zip(*batch)  # sv and tv: [batch, seq]; kv: [batch, know_num, know_seq]
        sv = [s[1:-1] for s in sv]
        kv = [[kk[1:-1] for kk in k] for k in kv]
        pad_sv = []
        pad_tv = []
        pad_kv = []

        sv_max_len = max([self.cal_max_len(s, 1, 1) for s in sv])
        tv_max_len = max([self.cal_max_len(t, 1, 1) for t in tv])
        kn_max_len = max([self.cal_max_len(k, 1, 2) for k in kv])
        kn_max_num = max([len(k) for k in kv])

        for i in range(len(sv)):
            tmp_sv = [0] * sv_max_len
            tmp_tv = [0] * tv_max_len
            tmp_kv = [[0 for i in range(kn_max_len)] for j in range(kn_max_num)]  # [know_num, know_seq]

            sv_len = len(sv[i])
            tmp_sv[:sv_len] = map(int, sv[i])

            tv_len = len(tv[i])
            tmp_tv[:tv_len] = map(int, tv[i])

            kv_num = len(kv[i])
            for j in range(kv_num):
                kn_len = len(kv[i][j])
                tmp_kv[j][:kn_len] = map(int, kv[i][j])

            pad_sv.append(tmp_sv)
            pad_tv.append(tmp_tv)
            pad_kv.append(tmp_kv)  # [batch, know_num, know_seq]

        sv_len = [len(s) for s in sv]
        tv_len = [len(t) for t in tv]
        kv_len = [
            [len(kt) for kt in k] + [0 for _ in range(kn_max_num - len(k))] for k in kv
        ]  # [batch, know_num]

        return (
            np.asarray(pad_sv).reshape((-1, sv_max_len)),
            np.asarray(sv_len),
            np.asarray(pad_tv).reshape((-1, tv_max_len)),
            np.asarray(tv_len),
            np.asarray(pad_kv).reshape((-1, kn_max_num, kn_max_len)),
            np.asarray(kv_len).reshape((-1, kn_max_num)),
            tstr,
        )

    def GetDataloader(self, batch_size, shuffle, num_workers):
        data_loader = data.DataLoader(
            self.instance,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )
        return data_loader


class AMRDialogSetNew(data.Dataset):
    def __init__(self, data_path, tokenize_fn, word_vocab, concept_vocab, relation_vocab, word_rel_vocab, lower=True, use_bert=False):
        self.instance = []
        self.max_tok_len = 512 if use_bert else 1000
        with open(data_path + '.src', 'r', encoding='utf-8') as srcf:
            with open(data_path + '.concept', 'r', encoding='utf-8') as conf:
                with open(data_path + '.path', 'r', encoding='utf-8') as con_relf:
                    with open(data_path + '.tgt', 'r', encoding='utf-8') as tgtf:
                        with open(data_path + '.mask', 'r', encoding='utf-8') as maskf:
                            with open(data_path + '.rel', 'r', encoding='utf-8') as word_relf:
                                for src_tok, src_concept, con_relation_raw, tgt_tok, word_rel_mask, word_rel in zip(srcf, conf, con_relf, tgtf, maskf, word_relf):
                                    src_tok = src_tok.replace(' <sep>', '')
                                    con_relation_lst_raw = con_relation_raw.strip().split(" ")
                                    seg_len = int(math.sqrt(len(con_relation_lst_raw)))
                                    assert seg_len * seg_len == len(con_relation_lst_raw)
                                    if lower:
                                        con_relation_lst = [
                                            ' '.join(con_relation_lst_raw[i: i + seg_len]).lower()
                                            for i in range(0, len(con_relation_lst_raw), seg_len)
                                        ]
                                    else:
                                        con_relation_lst = [
                                            ' '.join(con_relation_lst_raw[i: i + seg_len])
                                            for i in range(0, len(con_relation_lst_raw), seg_len)
                                        ]
                                    word_mask_lst = [itm.split(' ') for itm in word_rel_mask.split('\t')]
                                    word_rel_lst = [itm for itm in word_rel.split('\t')]
                                    # print(src_relation_lst)
                                    if lower:
                                        if use_bert:
                                            # print('using bert ...')
                                            src_id = bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + src_tok.strip().lower().split())
                                        else:
                                            src_tok = src_tok.lower().strip()
                                            src_id = tokenize_fn(word_vocab, src_tok, 1, 1, dtype="word")

                                        concept_id = tokenize_fn(concept_vocab, src_concept.strip().lower(), 1, 1, dtype="concept")
                                        tgt_id = tokenize_fn(word_vocab, tgt_tok.strip().lower(), 1, 1, dtype="word")
                                    else:
                                        if use_bert:
                                            src_id = bert_tokenizer.convert_tokens_to_ids(src_tok.strip().lower().split())
                                        else:
                                            src_id = tokenize_fn(word_vocab, src_tok.strip(), 1, 1, dtype="word")
                                        concept_id = tokenize_fn(concept_vocab, src_concept.strip(), 1, 1, dtype="concept")
                                        tgt_id = tokenize_fn(word_vocab, tgt_tok.strip(), 1, 1, dtype="word")

                                    con_rel_id = tokenize_fn(relation_vocab, con_relation_lst, 1, 2, dtype="relation")
                                    word_mask_id = [[int(iitm) for iitm in itm] for itm in word_mask_lst]
                                    word_rel_id = tokenize_fn(word_rel_vocab, word_rel_lst, 1, 2, dtype="relation")
                                    assert len(con_rel_id) == len(concept_id) and len(con_rel_id[0]) == len(concept_id), "concept_len: {}, relation_size:{} x {}".format(len(concept_id), len(con_rel_id), len(con_rel_id[0]))
                                    assert len(src_id) - 1 == len(word_mask_id[0]) and len(src_id) - 1 == len(word_rel_id[0]), "word_len: {}, mask_size:{} x {}, rel_size: {} x {}\nword_tok:{}\nword_ids:{}".format(len(src_id), len(word_mask_id), len(word_mask_id[0]), len(word_rel_id), len(word_rel_id[0]), src_tok, ' '.join([str(itm) for itm in src_id]))
                                    # filter_knowledge_id = tokenize_fn(vocab, filter_knowledge, 1, 2)
                                    self.instance.append((src_id, concept_id, con_rel_id, tgt_id, tgt_tok, word_mask_id, word_rel_id))

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, index):
        src_id, concept_id, con_rel_id, tgt_id, tgt_tok, word_mask, word_rel_id = self.instance[index]
        return src_id, concept_id, con_rel_id, tgt_id, tgt_tok, word_mask, word_rel_id

    def cal_max_len(self, ids, curdepth, maxdepth):
        """calculate max sequence length"""
        assert curdepth <= maxdepth
        if isinstance(ids[0], list):
            res = max([self.cal_max_len(k, curdepth + 1, maxdepth) for k in ids])
        else:
            res = len(ids)
        return res

    def collate_fn(self, batch, ):
        sv, cv, rv, tv, tstr, mask, wr = zip(*batch)    # sv, cv, rv and tv: [batch, seq];
        sv = [s[:-1] for s in sv]                       # remove eos 

        pad_sv, pad_cv, pad_rv, pad_tv, pad_mask, pad_wr = [], [], [], [], [], []

        sv_max_len = min(max([self.cal_max_len(s, 1, 1) for s in sv]), self.max_tok_len)
        cv_max_len = max([self.cal_max_len(c, 1, 1) for c in cv])
        rv_max_len = max([self.cal_max_len(r, 1, 2) for r in rv])
        tv_max_len = max([self.cal_max_len(t, 1, 1) for t in tv])
        mask_max_len = min(max([self.cal_max_len(m, 1, 2) for m in mask]), self.max_tok_len)
        wr_max_len = min(max([self.cal_max_len(w, 1, 2) for w in wr]), self.max_tok_len)
        # kn_max_len = max([self.cal_max_len(k, 1, 2) for k in kv])
        # kn_max_num = max([len(k) for k in kv])

        assert rv_max_len == cv_max_len, "Error, rv_max_len should be equal with cv_max_len!!"
        assert sv_max_len == mask_max_len and sv_max_len == wr_max_len, "Error, sv_max_len should be equal with mask_max_len and wr_max_len!!"
        # print('sv_max_len', sv_max_len)
        for i in range(len(sv)):
            tmp_sv = [0] * sv_max_len
            tmp_cv = [0] * cv_max_len
            tmp_tv = [0] * tv_max_len
            tmp_rv = [
                [0 for i in range(rv_max_len)] for j in range(rv_max_len)
            ]  # [rv_max_len, rv_max_len]
            tmp_mask = [
                [0 for i in range(mask_max_len)] for j in range(mask_max_len)
            ]
            tmp_wr = [
                [0 for i in range(wr_max_len)] for j in range(wr_max_len)
            ]

            ith_sv_len = min(len(sv[i]), self.max_tok_len)
            # print('sv: ori_len: {}, sv_len: {}'.format(len(sv[i]), ith_sv_len))
            # tmp_sv[:ith_sv_len] = map(int, sv[i][:self.max_tok_len])    #
            tmp_sv[:ith_sv_len] = map(int, sv[i][-self.max_tok_len:])    #

            ith_cv_len = len(cv[i])
            tmp_cv[:ith_cv_len] = map(int, cv[i])

            ith_tv_len = len(tv[i])
            tmp_tv[:ith_tv_len] = map(int, tv[i])

            ith_rv_len = len(rv[i])     # rv_len
            for j in range(ith_rv_len):
                rv_len_j = len(rv[i][j])
                tmp_rv[j][:rv_len_j] = map(int, rv[i][j])

            ith_mask_len = min(len(mask[i]), self.max_tok_len)  # rv_len
            for j in range(ith_mask_len):
                mask_len_j = min(len(mask[i][j]), self.max_tok_len)
                # tmp_mask[j][:mask_len_j] = map(int, mask[i][j][:self.max_tok_len])
                tmp_mask[j][:mask_len_j] = map(int, mask[i][j][-self.max_tok_len:])

            ith_wr_len = min(len(wr[i]), self.max_tok_len)  # rv_len
            for j in range(ith_wr_len):
                wr_len_j = min(len(wr[i][j]), self.max_tok_len)
                # tmp_wr[j][:wr_len_j] = map(int, wr[i][j][:self.max_tok_len])
                tmp_wr[j][:wr_len_j] = map(int, wr[i][j][-self.max_tok_len:])

            # kv_num = len(kv[i])
            # for j in range(kv_num):
            #     kn_len = len(kv[i][j])
            #     tmp_kv[j][:kn_len] = map(int, kv[i][j])

            pad_sv.append(tmp_sv)   # [batch, sv_max_len]
            pad_cv.append(tmp_cv)
            pad_rv.append(tmp_rv)   # [batch, len_c, len_c]
            pad_tv.append(tmp_tv)
            pad_mask.append(tmp_mask)
            pad_wr.append(tmp_wr)
            # pad_kv.append(tmp_kv)  # [batch, know_num, know_seq]

        sv_len = [min(len(s), self.max_tok_len) for s in sv]
        cv_len = [len(c) for c in cv]
        tv_len = [len(t) for t in tv]
        # kv_len = [
        #     [len(kt) for kt in k] + [0 for _ in range(kn_max_num - len(k))] for k in kv
        # ]  # [batch, know_num]
        # print(pad_sv)
        # tmp_sv_ = np.asarray(pad_sv)
        # print('tmp_sv', tmp_sv_.shape)
        return (
            np.asarray(pad_sv).reshape((-1, sv_max_len)),
            np.asarray(sv_len),
            np.asarray(pad_cv).reshape((-1, cv_max_len)),
            np.asarray(cv_len),
            np.asarray(pad_rv).reshape((-1, rv_max_len, rv_max_len)),
            np.asarray(pad_tv).reshape((-1, tv_max_len)),
            np.asarray(tv_len),
            np.asarray(pad_mask).reshape((-1, mask_max_len, mask_max_len)),
            np.asarray(pad_wr).reshape((-1, wr_max_len, wr_max_len)),
            tstr,
        )

    def GetDataloader(self, batch_size, shuffle, num_workers):
        data_loader = data.DataLoader(
            self.instance,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )
        return data_loader