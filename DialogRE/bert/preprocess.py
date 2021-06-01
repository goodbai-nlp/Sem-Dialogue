# coding:utf-8
import os
import torch
import numpy as np
import argparse
from dataset_new import AMRDialogDataSet
from dataset_utils import load_vocab, save_vocab, generate_vocab, tokenize
from torch.utils import data
import time
import pickle

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_path", type=str, help="base train data path")
    argparser.add_argument("--dev_path", type=str, help="base dev data path")
    argparser.add_argument("--test_path", type=str, help="base test data path")
    argparser.add_argument("--devc_path", type=str, help="base dev data path")
    argparser.add_argument("--testc_path", type=str, help="base test data path")
    argparser.add_argument("--save_data", type=str, help="saved data path")
    FLAGS, unparsed = argparser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if not os.path.exists(FLAGS.save_data):
        os.makedirs(FLAGS.save_data)

    print('Building vocabularies ... ')
    words, word2id = generate_vocab(FLAGS.train_path + '.src.tok', add_bos_eos=True)
    concepts, con2id = generate_vocab(FLAGS.train_path + '.concept', add_bos_eos=True)
    relations, relation2id = generate_vocab(FLAGS.train_path + '.rel', relation_vocab_size=5000, add_bos_eos=False)
    paths, path2id = generate_vocab(FLAGS.train_path + '.path', relation_vocab_size=5000, add_bos_eos=True)
    print('Saving vocabularies ... ')
    save_vocab(FLAGS.save_data + '/rel.vocab', relations)
    save_vocab(FLAGS.save_data + '/word.vocab', words)
    save_vocab(FLAGS.save_data + '/con.vocab', concepts)
    save_vocab(FLAGS.save_data + '/path.vocab', paths)

    tokenize_fn = tokenize
    concept_vocab = con2id
    relation_vocab = relation2id
    path_vocab = path2id
    print("Loading train data ...")
    s_time = time.time()
    train_set = AMRDialogDataSet(FLAGS.train_path, tokenize_fn, word_rel_vocab=relation2id, word_vocab=word2id, concept_vocab=con2id, path_vocab=path2id, save_path=FLAGS.save_data+'/train', data_type='std')
    print("Loading trainset takes {:.3f}s".format(time.time() - s_time))

    print("Loading dev data ...")
    s_time = time.time()
    dev_set = AMRDialogDataSet(FLAGS.dev_path, tokenize_fn, word_rel_vocab=relation2id, word_vocab=word2id, concept_vocab=con2id, path_vocab=path2id, save_path=FLAGS.save_data+'/dev', data_type='std')
    print("Loading devset takes {:.3f}s".format(time.time() - s_time))

    print("Loading test data ...")
    s_time = time.time()
    test_set = AMRDialogDataSet(FLAGS.test_path, tokenize_fn, word_rel_vocab=relation2id, word_vocab=word2id, concept_vocab=con2id, path_vocab=path2id, save_path=FLAGS.save_data+'/test', data_type='std')
    print("Loading testset takes {:.3f}s".format(time.time() - s_time))

    print("Loading devc data ...")
    s_time = time.time()
    # devc_set = AMRDialogDataSet(FLAGS.devc_path, tokenize_fn, word_rel_vocab=relation2id, word_vocab=word2id, concept_vocab=con2id, path_vocab=path2id, save_path=FLAGS.save_data+'/devc', data_type='stdc')
    print("Loading devcset takes {:.3f}s".format(time.time() - s_time))

    print("Loading testc data ...")
    s_time = time.time()
    # testc_set = AMRDialogDataSet(FLAGS.testc_path, tokenize_fn, word_rel_vocab=relation2id, word_vocab=word2id, concept_vocab=con2id, path_vocab=path2id, save_path=FLAGS.save_data+'/testc', data_type='stdc')
    print("Loading testcset takes {:.3f}s".format(time.time() - s_time))
