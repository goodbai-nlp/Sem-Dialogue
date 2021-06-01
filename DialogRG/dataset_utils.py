import codecs
import re
from collections import OrderedDict, Counter, defaultdict
import json
import numpy as np
import torch


def generate_vocab(path, vocab_size=-1):
    vocab = Counter()
    with codecs.open(path, "r", "utf-8") as fr:
        for line in fr:
            if len(line.rstrip("\n").split("\t")) < 3:
                continue
            line = line.strip().replace("\1", " ").split()
            vocab.update(line)
    words = ["<pad>", "<bos>", "<eos>", "<unk>", ]
    word2id = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, }
    for word, _ in vocab.most_common(vocab_size if vocab_size > 0 else len(vocab)):
        words.append(word)
        word2id[word] = len(words) - 1
    return words, word2id


def merge_vocab(vocab1, vocab2):
    return vocab1 + vocab2


def generate_vocab_new(path, word_vocab_size=-1, concept_vocab_size=-1, relation_vocab_size=-1, shared_word_concept=False, lower=True):
    word_vocab = Counter()
    concept_vocab = Counter()
    relation_vocab = Counter()
    word_rel_vocab = Counter()

    with open(path + '.src', 'r', encoding='utf-8') as srcf:
        with open(path + '.concept', 'r', encoding='utf-8') as conf:
            with open(path + '.path', 'r', encoding='utf-8') as relf:
                with open(path + '.tgt', 'r', encoding='utf-8') as tgtf:
                    with open(path + '.rel', 'r', encoding='utf-8') as word_relf:
                        for src_tok, src_concept, src_relation, tgt_tok, word_rel in zip(srcf, conf, relf, tgtf, word_relf):
                            src_tok = src_tok.replace('<sep>', ' ')
                            if lower:
                                word_vocab.update(src_tok.strip().lower().split() + tgt_tok.strip().lower().split())
                                concept_vocab.update(src_concept.strip().lower().split())
                                relation_vocab.update(src_relation.strip().lower().split())
                                word_rel_vocab.update(word_rel.strip().lower().split())
                            else:
                                word_vocab.update(src_tok.strip().split() + tgt_tok.strip().split())
                                concept_vocab.update(src_concept.strip().split())
                                relation_vocab.update(src_relation.strip().split())
                                word_rel_vocab.update(word_rel.strip().split())
    if shared_word_concept:
        word_vocab = merge_vocab(word_vocab, concept_vocab)
    words = ["<pad>", "<bos>", "<eos>", "<unk>", ]
    word2id = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, }
    for word, _ in word_vocab.most_common(word_vocab_size if word_vocab_size > 0 else len(word_vocab)):
        words.append(word)
        word2id[word] = len(words) - 1
    if shared_word_concept:
        concepts = words
        concept2id = word2id
    else:
        concepts = ["<pad>", "<bos>", "<eos>", "<unk>", ]
        concept2id = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, }
        for con, _ in concept_vocab.most_common(concept_vocab_size if concept_vocab_size > 0 else len(concept_vocab)):
            concepts.append(con)
            concept2id[con] = len(concepts) - 1
    relations = ["<pad>", "<unk>", ]
    relation2id = {"<pad>": 0, "<unk>": 1, }
    for rel, _ in relation_vocab.most_common(relation_vocab_size if relation_vocab_size > 0 else len(relation_vocab)):
        relations.append(rel)
        relation2id[rel] = len(relations) - 1
    word_rels = ["<pad>", "<unk>", ]
    word_rel2id = {"<pad>": 0, "<unk>": 1, }
    for rel, _ in word_rel_vocab.most_common(relation_vocab_size if relation_vocab_size > 0 else len(word_rel_vocab)):
        word_rels.append(rel)
        word_rel2id[rel] = len(word_rels) - 1
    print('Word vocab size: ', len(words))
    print('Concept vocab size: ', len(concepts))
    print('Concept Relation vocab size: ', len(relations))
    print('Word Relation vocab size: ', len(word_rels))
    return words, word2id, concepts, concept2id, relations, relation2id, word_rels, word_rel2id


def load_file(path):
    data = []
    # kn_num = defaultdict(float)
    with codecs.open(path, "r", "utf-8") as fr:
        for line in fr:
            if len(line.rstrip("\n").split("\t")) < 3:
                continue
            src, tgt, knowledge = line.rstrip("\n").split("\t")[:3]
            src = src.strip()
            tgt = tgt.strip()
            knowledge = knowledge.strip()
            filter_knowledge = []
            for sent in knowledge.split("\1"):
                filter_knowledge.append(" ".join(sent.split()[:500]))
            # kn_num[len(filter_knowledge)] += 1.0

            # concat the knowledge information into input
            # info = src.split(':')
            #
            # goal = info[0]
            # context = info[1]
            # flatted_knowledge = ' '.join(filter_knowledge)
            #
            # src = goal.strip() + ' ' + flatted_knowledge.strip() + ' : ' + context.strip()
            # src = src.strip()

            data.append((src, tgt, filter_knowledge))

    # if len(kn_num) > 0:
    #    total = sum(kn_num.values())
    #    kn_num = sorted((k,100.0*v/total) for k,v in kn_num.items())
    #    print(' '.join('{}:{:.2f}'.format(k,v) for k,v in kn_num))

    return data


def load_json_file(path):
    new_data = []
    ori_data = json.load(open(path, 'r', encoding='utf-8'))
    for item in ori_data:
        src_tok, src_concept, src_relation, tgt_tok = item['src'], item['concept'], item['relation'], item['tgt']
        new_data.append((src_tok, src_concept, src_relation, tgt_tok))
    return new_data


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
            # tokens_ids = (
            #     [vocab_dict.get(tok, vocab_dict.get("<unk>")) for tok in tokens]
            # )
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


def load_vocab(path):
    words = []
    word2id = {}
    with codecs.open(path, "r", "utf-8") as fr:
        for symbol in fr:
            symbol = symbol.strip()
            words.append(symbol)
            word2id[symbol] = len(words) - 1
    return words, word2id


def load_vocab_new(path):
    words, concepts, relations, word_rels = [], [], [], []
    word2id, concept2id, relation2id, word_rel2id = {}, {}, {}, {}
    with codecs.open(path + '/vocab.tok', "r", "utf-8") as fr:
        for symbol in fr:
            symbol = symbol.strip()
            words.append(symbol)
            word2id[symbol] = len(words) - 1
    with codecs.open(path + '/vocab.con', "r", "utf-8") as fr:
        for symbol in fr:
            symbol = symbol.strip()
            concepts.append(symbol)
            concept2id[symbol] = len(concepts) - 1
    with codecs.open(path + '/vocab.rel', "r", "utf-8") as fr:
        for symbol in fr:
            symbol = symbol.strip()
            relations.append(symbol)
            relation2id[symbol] = len(relations) - 1
    with codecs.open(path + '/vocab.rel2', "r", "utf-8") as fr:
        for symbol in fr:
            symbol = symbol.strip()
            word_rels.append(symbol)
            word_rel2id[symbol] = len(word_rels) - 1
    return words, word2id, concepts, concept2id, relations, relation2id, word_rels, word_rel2id


def save_vocab(path, words):
    with codecs.open(path, "w", "utf-8") as fr:
        for symbol in words:
            fr.write(symbol + "\n")


def save_vocab_new(path, words, concepts, relations, word_rels):
    with codecs.open(path + '/vocab.tok', "w", "utf-8") as fr:
        for symbol in words:
            fr.write(symbol + "\n")
    with codecs.open(path + '/vocab.con', "w", "utf-8") as fr:
        for symbol in concepts:
            fr.write(symbol + "\n")
    with codecs.open(path + '/vocab.rel', "w", "utf-8") as fr:
        for symbol in relations:
            fr.write(symbol + "\n")
    with codecs.open(path + '/vocab.rel2', "w", "utf-8") as fr:
        for symbol in word_rels:
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
        max_len = torch.max(len_seq)
    mask = torch.zeros((len_seq.size(0), max_len))
    for i, l in enumerate(len_seq):
        mask[i, :l] = 1
    return mask
