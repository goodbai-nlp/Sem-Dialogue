# coding:utf-8
import codecs
import re
from collections import OrderedDict, Counter, defaultdict
import json
import numpy as np


def tokenize(vocab_dict, src, curdepth, maxdepth, dtype='word', lower=True):
    """map src to ids"""
    assert curdepth <= maxdepth, 'cur:{}, max:{}'.format(curdepth, maxdepth)
    if isinstance(src, str):
        if dtype != 'relation':
            if lower:
                src = re.sub("\d+", "<num>", src).lower()
            else:
                src = re.sub("\d+", "<num>", src)
        else:
            if lower:
                src = src.lower()
        tokens = src.split()
        if dtype == 'word':         # add bos, eos
            tokens_ids = (
                [vocab_dict.get("<bos>")]
                + [vocab_dict.get(tok, vocab_dict.get("<unk>")) for tok in tokens]
                + [vocab_dict.get("<eos>")]
            )
        elif dtype == 'relation':   # add None
            # print('tokens', tokens)
            tokens_ids = ([vocab_dict.get(tok, vocab_dict.get("<unk>")) for tok in tokens])
            # print('token_ids', tokens_ids)
        elif dtype == 'concept':    # add eos
            tokens_ids = ([vocab_dict.get(tok, vocab_dict.get("<unk>")) for tok in tokens] + [vocab_dict.get("<eos>")])
        else:
            print("Invalid dtype", dtype)
        return tokens_ids
    elif isinstance(src, list):
        tokens_list = [tokenize(vocab_dict, t, curdepth + 1, maxdepth, dtype=dtype) for t in src]
        return tokens_list


def bpe_tokenize(bpe_tokenizer, src, curdepth, maxdepth):
    if isinstance(src, str):
        words = src.split(" ")
        ids = [
            bpe_tokenizer.BOS,
        ]
        for i in range(len(words)):
            ids += bpe_tokenizer.encode_ids(words[i])
        ids += [
            bpe_tokenizer.EOS,
        ]
        return ids
    elif isinstance(src, list):
        ids_list = []
        for s in src:
            ids = bpe_tokenize(bpe_tokenizer, s, curdepth + 1, maxdepth)
            ids_list.append(ids)
        return ids_list


def bert_tokenize(tokenizer, src):
    """
    :param tokenizer: the bert tokenizer
    :param src:
    :return:
    """
    if isinstance(src, str):
        words = re.sub("\d+", "number", src).split(" ")
        words.append("[SEP]")
        words.insert(0, "[CLS]")

        ids = []
        tok2word = []
        total_offset = 0
        for _ in range(len(words)):
            tokens = tokenizer.tokenize(words[_])
            tokens = [t if t in tokenizer.vocab else "[UNK]" for t in tokens]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            if len(token_ids) > 0:
                ids.extend(token_ids)
                positions = [i + total_offset for i in range(len(token_ids))]
                total_offset += len(token_ids)
                tok2word.append(positions)

        return ids, tok2word

    elif isinstance(src, list):
        ids_list = []
        tok2word_list = []
        for s in src:
            ids, tok2word, word_len = bert_tokenize(tokenizer, s)
            ids_list.append(ids)
            tok2word_list.append(tok2word)

        return ids_list, tok2word_list


def merge_vocab(vocab1, vocab2):
    return vocab1 + vocab2


def generate_vocab(path, relation_vocab_size=-1, lower=True, add_bos_eos=False):
    word_rel_vocab = Counter()
    with open(path, 'r', encoding='utf-8') as word_relf:
        for word_rel in word_relf:
            word_rel_vocab.update(word_rel.strip().lower().split())
    if add_bos_eos:
        word_rels = ["<pad>", "<unk>", "<bos>", "<eos>"]
        word_rel2id = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    else:
        word_rels = ["<pad>", "<unk>"]
        word_rel2id = {"<pad>": 0, "<unk>": 1}
    for rel, _ in word_rel_vocab.most_common(relation_vocab_size if relation_vocab_size > 0 else len(word_rel_vocab)):
        word_rels.append(rel)
        word_rel2id[rel] = len(word_rels) - 1
    print('Vocabulary Size: ', len(word_rels))
    return word_rels, word_rel2id


def load_vocab(path):
    toks, tok2id = [], {}
    with codecs.open(path, "r", "utf-8") as fr:
        for symbol in fr:
            symbol = symbol.strip()
            toks.append(symbol)
            tok2id[symbol] = len(toks) - 1
    print('Loaded {} instances from {}'.format(len(toks), path))
    return toks, tok2id


def save_vocab(path, words):
    with codecs.open(path, "w", "utf-8") as fr:
        for symbol in words:
            fr.write(symbol + "\n")


def load_pretrained_emb(emb_path, symbol_idx, word_emb_dim):
    print("using pretrained embedding for initialization...")
    symbol_vec = OrderedDict()
    with codecs.open(emb_path, "r", "utf-8") as fr:
        for line in fr:
            info = line.strip().split(" ")
            word = info[0]
            vec = [float(x) for x in info[1:]]
            if len(vec) != word_emb_dim:
                continue
            symbol_vec[word] = np.array(vec)

    pretrained_embeddings = []
    init_range = np.sqrt(6.0 / word_emb_dim)
    for symbol in symbol_idx:
        if symbol == "<pad>":
            pretrained_embeddings.append(np.zeros(word_emb_dim))
        elif symbol in symbol_vec:
            pretrained_embeddings.append(symbol_vec[symbol])
        else:
            pretrained_embeddings.append(np.random.uniform(-init_range, init_range, word_emb_dim))

    for emb in pretrained_embeddings:
        assert len(emb) == word_emb_dim
    pretrained_embeddings = np.stack(pretrained_embeddings).astype(np.float32)
    return pretrained_embeddings


def cal_max_len(ids, curdepth, maxdepth):
    """calculate max sequence length"""
    assert curdepth <= maxdepth
    if isinstance(ids[0], list):
        res = max([cal_max_len(k, curdepth + 1, maxdepth) for k in ids])
    else:
        res = len(ids)
    return res


def len_to_mask(len_seq, max_len=None):
    """len to mask"""
    if max_len is None:
        max_len = np.max(len_seq)
    mask = np.zeros((len_seq.shape[0], max_len), dtype="float32")
    for i, l in enumerate(len_seq):
        mask[i, :l] = 1
    return mask
