import os
import torch
import random
from apex import amp

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import torch.nn as nn
import numpy as np

np.random.seed(0)
torch.backends.cudnn.deterministic = True
import tqdm
import argparse
import config_utils
from dataset_utils import (
    load_vocab_new,
    tokenize,
    bpe_tokenize,
)
# from model import DualTransformer
# from bert_model import DualTransformer
from model_adapter import DualTransformer
from torch.utils import data
from torch import optim
from bpemb import BPEmb
import math
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def len_to_mask(len_seq, max_len=None):
    """len to mask"""
    len_seq = len_seq.int()
    if max_len is None:
        max_len = torch.max(len_seq).item()
    mask = torch.zeros((len_seq.size(0), max_len))
    for i, l in enumerate(len_seq):
        mask[i, :l] = 1
    return mask


def train(model, data_loader, optimizer, n_gpu, FLAGS):
    model.train()
    train_loss = 0.0
    train_step = 0.0
    train_right, train_total = 0.0, 0.0
    # for i, _data in enumerate(data_loader):
    for i, _data in tqdm.tqdm(enumerate(data_loader)):
        sv, sv_len, cv, cv_len, rv, tv, tv_len, word_rel_mask, wr, tstr = _data
        batch = {}
        batch["tgt_str"] = tstr

        batch["src"] = sv
        batch["src_len"] = sv_len
        batch["src_mask"] = len_to_mask(sv_len)

        batch["con"] = cv
        batch["con_len"] = cv_len
        batch["con_mask"] = len_to_mask(cv_len)

        batch["rel"] = rv

        # batch["word_rel_mask"] = word_rel_mask
        batch["wr"] = wr

        batch["tgt_input"] = tv[:, :-1]    # remove eos
        batch["tgt_ref"] = tv[:, 1:]       # remove bos
        batch["tgt_len"] = tv_len - 1
        batch["tgt_mask"] = len_to_mask(tv_len - 1)

        outputs = model(batch)

        loss = outputs["loss"]
        if n_gpu > 1:
            loss = loss.mean()
        if FLAGS.grad_accum_steps > 1:
            loss = loss / FLAGS.grad_accum_steps
        if FLAGS.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()                     # just calculate gradient

        if i % FLAGS.grad_accum_steps == 0:     # optimizer step
            optimizer.step()
            optimizer.zero_grad()

        train_loss += float(loss.cpu().item())
        train_step += 1.0
        train_right += outputs["counts"][0].mean().cpu().item()
        train_total += outputs["counts"][1].mean().cpu().item()
        torch.cuda.empty_cache()
    
    return train_loss / train_step, train_right / train_total


def validate(model, data_loader, n_gpu):
    model.eval()
    with torch.no_grad():
        dev_loss = 0.0
        dev_step = 0.0
        dev_right, dev_total = 0.0, 0.0
        for i, _data in tqdm.tqdm(enumerate(data_loader)):
            # sv, sv_len, cv, cv_len, rv, tv, tv_len, tstr = _data
            sv, sv_len, cv, cv_len, rv, tv, tv_len, word_rel_mask, wr, tstr = _data
            batch = {}
            batch["tgt_str"] = tstr

            batch["src"] = sv
            batch["src_len"] = sv_len
            batch["src_mask"] = len_to_mask(sv_len)

            batch["con"] = cv
            batch["con_len"] = cv_len
            batch["con_mask"] = len_to_mask(cv_len)

            batch["rel"] = rv

            # batch["word_rel_mask"] = word_rel_mask
            batch["wr"] = wr

            batch["tgt_input"] = tv[:, :-1]  # remove eos
            batch["tgt_ref"] = tv[:, 1:]  # remove bos
            batch["tgt_len"] = (tv_len - 1)
            batch["tgt_mask"] = len_to_mask(tv_len - 1)

            outputs = model(batch)
            loss = outputs["loss"]
            if n_gpu > 1:
                loss = loss.mean()
            dev_loss += float(loss.cpu().item())
            dev_step += 1.0
            dev_right += outputs["counts"][0].mean().cpu().item()
            dev_total += outputs["counts"][1].mean().cpu().item()
            torch.cuda.empty_cache()
    
    return dev_loss / dev_step, dev_right / dev_total


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_path", type=str, help="Configuration file.")
    FLAGS, unparsed = argparser.parse_known_args()

    if FLAGS.config_path is not None:
        print("Loading hyperparameters from " + FLAGS.config_path)
        FLAGS = config_utils.load_config(FLAGS.config_path)

    log_dir = FLAGS.log_dir
    continue_train = False
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        continue_train = True
    path_prefix = log_dir + "/wiki.{}".format(FLAGS.suffix)
    log_file = open(path_prefix + ".log", "w+")
    if not continue_train:
        log_file.write("{}\n".format(str(FLAGS)))
        log_file.flush()
    print("Log file path: {}".format(path_prefix + ".log"))
    config_utils.save_config(FLAGS, path_prefix + ".config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    # n_gpu = 0
    print(
        "device: {}, n_gpu: {}, grad_accum_steps: {}".format(device, n_gpu, FLAGS.grad_accum_steps)
    )
    log_file.write(
        "device: {}, n_gpu: {}, grad_accum_steps: {}\n".format(
            device, n_gpu, FLAGS.grad_accum_steps
        )
        if not continue_train
        else ""
    )
    # exit()
    s_time = time.time()
    words, word2id, concepts, concept2id, relations, relation2id, word_rels, word_rel2id = load_vocab_new(FLAGS.save_data)
    print("Loading vocab takes {:.3f}s".format(time.time() - s_time))
    print("Vocabulary size: {}".format(len(words)))
    if FLAGS.save_data != "":
        tokenize_fn = tokenize
        word_vocab, concept_vocab, relation_vocab, word_rel_vocab = word2id, concept2id, relation2id, word_rel2id
    else:
        print("Not support other vocabulary for now!!")
        exit()

    checkpoint = None
    best_checkpoint_path = path_prefix + "_best.checkpoint.bin"
    last_checkpoint_path = path_prefix + "_last.checkpoint.bin"
    if os.path.exists(last_checkpoint_path):
        print("!!Existing checkpoint. Loading...")
        log_file.write("!!Existing checkpoint. Loading...\n")
        checkpoint = torch.load(last_checkpoint_path)

    if FLAGS.use_bpe_pretrain:
        # word_emb = torch.load(FLAGS.save_data + "/word_emb.pt")
        word_emb = None
        con_emb = torch.load(FLAGS.save_data + "/concept_emb.pt")
    else:
        word_emb = None
        con_emb = None

    model = DualTransformer(FLAGS, word_emb, con_emb, word2id, concept2id, relation2id, word_rel2id)
    model.to(device)
    # model = DualTransformer(FLAGS, None, None, word2id, None, None)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    if FLAGS.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")  # 这里是“欧一”，不是“零一”

    print(model)
    print(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )
    if not continue_train:
        log_file.write(str(model))
        log_file.write(
            "num. model params: {} (num. trained: {})".format(
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )
        )
    if n_gpu > 1:
        model = nn.DataParallel(model)
    if checkpoint:
        if n_gpu <= 1:
            new_pre = {}
            for k, v in checkpoint["model_state_dict"].items():
                name = k[7:] if k.startswith("module") else k
                new_pre[name] = v
            model.load_state_dict(new_pre)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

    best_accu = 0.0
    if checkpoint:
        assert "best_accu" in checkpoint
        best_accu = checkpoint["best_accu"]
        print("Initial accuracy {:.4f}".format(best_accu))
        log_file.write("Initial accuracy {:.4f}\n".format(best_accu))

    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    # for the usage of BertAdam
    # named_params = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # grouped_params = [
    #    {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #    {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
    # ]

    worker_init_fn = lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
    print("Loading train data and making batches")
    log_file.write("Loading data and making batches\n")
    s_time = time.time()
    train_set = torch.load(FLAGS.save_data + "/train_data.pt")
    train_set.max_tok_len = 512
    print("Loading data takes {:.3f}s".format(time.time() - s_time))
    s_time = time.time()
    train_loader = train_set.GetDataloader(
        batch_size=FLAGS.batch_size, shuffle=FLAGS.is_shuffle, num_workers=4
    )
    print("Loading dev data and making batches")
    dev_set = torch.load(FLAGS.save_data + "/dev_data.pt")
    dev_loader = dev_set.GetDataloader(batch_size=FLAGS.batch_size, shuffle=False, num_workers=1)
    print("Loading dev data takes {:.3f}s".format(time.time() - s_time))
    print("Num training examples = {}".format(len(train_set.instance)))
    log_file.write("Num training examples = {}\n".format(len(train_set.instance)))

    max_patience = FLAGS.patience
    patience = 0
    optimizer.zero_grad()
    for iter in range(start_epoch, FLAGS.num_epochs):
        train_loss, train_accu = train(model, train_loader, optimizer, n_gpu, FLAGS)
        val_loss, val_accu = validate(model, dev_loader, n_gpu)

        print(
            "iter: {}, lr:{:.5f} TRAIN loss: {:.4f} accu: {:.4f}; VAL loss: {:.4f} accu: {:.4f}".format(
                iter, optimizer.param_groups[0]["lr"], train_loss, train_accu, val_loss, val_accu
            )
        )
        log_file.write(
            "iter: {}, lr:{:.5f} TRAIN loss: {:.4f} accu: {:.4f}; VAL loss: {:.4f} accu: {:.4f}\n".format(
                iter, optimizer.param_groups[0]["lr"], train_loss, train_accu, val_loss, val_accu
            )
        )
        state = {
            "epoch": iter,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_accu": best_accu,
        }
        # print("saving last model ...")
        torch.save(state, last_checkpoint_path)
        if best_accu < val_accu:
            best_accu = val_accu
            patience = 0
            # save model
            print("saving best model ...")
            log_file.write("saving best model ...\n")
            config_utils.save_config(FLAGS, path_prefix + ".config.json")
            state = {
                "epoch": iter,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_accu": best_accu,
            }
            torch.save(state, best_checkpoint_path)
            os.system("cp {} {}".format(best_checkpoint_path, last_checkpoint_path))
        else:
            patience += 1
            if patience >= max_patience:
                print("Reaching max patience! exit...")
                exit()
        if iter % 10 == 0 and iter > 0:
            # state = {
            #     "epoch": iter,
            #     "model_state_dict": model.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            #     "best_accu": best_accu,
            # }
            # print("saving last model ...")
            tmp_checkpoint_path = path_prefix + "_epoch_{}.checkpoint.bin".format(iter)
            # os.system("cp {} {}".format(last_checkpoint_path, tmp_checkpoint_path))
            os.system("cp {} {}.{}".format(best_checkpoint_path, best_checkpoint_path, iter))
        # torch.cuda.empty_cache()