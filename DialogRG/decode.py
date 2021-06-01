import os
import json
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
import tqdm
import argparse
import config_utils
from dataset import AMRDialogSetNew
from dataset_utils import load_vocab_new, save_vocab_new, generate_vocab_new, tokenize, bpe_tokenize
# from model import DualTransformer
# from bert_model import DualTransformer
from model_adapter import DualTransformer
from torch.utils import data
from torch import optim
from bpemb import BPEmb
import math
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cpu"
__CUDA__ = False
if torch.cuda.is_available():
    device = "cuda"
    __CUDA__ = True


def len_to_mask(len_seq, max_len=None):
    """len to mask"""
    if max_len is None:
        max_len = np.max(len_seq)

    mask = np.zeros((len_seq.shape[0], max_len), dtype="float32")
    for i, l in enumerate(len_seq):
        mask[i, :l] = 1.0

    return mask


def decode(model, data_loader, beam_size, max_decoding_step):
    model.eval()
    results, results2 = [], []
    selected_kn_counts = Counter()
    trg_selected_kn_counts = Counter()
    for i, _data in enumerate(data_loader):
        if i % 100 == 0:
            print(i)
        sv, sv_len, cv, cv_len, rv, tv, tv_len, word_rel_mask, wr, tstr = _data
        batch = {}
        batch["tgt_str"] = tstr

        batch["src"] = torch.from_numpy(sv).to(device)
        batch["src_len"] = torch.from_numpy(sv_len).to(device)
        batch["src_mask"] = torch.from_numpy(len_to_mask(sv_len)).to(device)

        batch["con"] = torch.from_numpy(cv).to(device)
        batch["con_len"] = torch.from_numpy(cv_len).to(device)
        batch["con_mask"] = torch.from_numpy(len_to_mask(cv_len)).to(device)

        batch["rel"] = torch.from_numpy(rv).to(device)

        batch["word_rel_mask"] = torch.from_numpy(word_rel_mask).to(device)
        batch["wr"] = torch.from_numpy(wr).to(device)

        # batch["tgt_input"] = torch.from_numpy(tv[:, :-1]).to(device)    # remove eos
        # batch["tgt_ref"] = torch.from_numpy(tv[:, 1:]).to(device)       # remove bos
        # batch["tgt_len"] = torch.from_numpy(tv_len - 1).to(device)
        # batch["tgt_mask"] = torch.from_numpy(len_to_mask(tv_len - 1)).to(device)

        # batch['target_input'] = torch.from_numpy(tv[:,:-1]).to(device)
        # batch['target_ref'] = torch.from_numpy(tv[:,1:]).to(device)
        # batch['target_len'] = torch.from_numpy(tv_len-1).to(device)
        # batch['target_mask'] = torch.from_numpy(len_to_mask(tv_len-1)).to(device)
        batch["tgt_ref"] = None

        outputs = model(batch)

        # if outputs["selected_kn"] is not None:
        #     selected_kn = outputs["selected_kn"]
        #     selected_kn_counts.update(flatten_list(selected_kn.cpu().tolist()))
        # if outputs["trg_selected_kn"] is not None:
        #     trg_selected_kn = outputs["trg_selected_kn"]
        #     trg_selected_kn_counts.update(flatten_list(trg_selected_kn.cpu().tolist()))

        predictions = model.decode(
            outputs["sent_memory_emb"],
            outputs["graph_memory_emb"],
            outputs["sent_memory_mask"],
            outputs["graph_memory_mask"],
            beam_size,
            max_decoding_step,
        )
        for j, prd in enumerate(predictions):
            input = sv[j].tolist()
            input = model.decode_into_string(input)
            ref = tv[j].tolist()
            # print('ref', ref)
            ref = [int(y) for y in ref[1: ref.index(model.EOS)]]
            ref = model.decode_into_string(ref)
            prd = model.decode_into_string(prd)
            results.append(json.dumps({"input": input, "gold": ref, "pred": prd}))
            results2.append(prd)

    if len(selected_kn_counts) > 0:
        print("selected_kn_counts")
        total = sum(selected_kn_counts.values())
        selected_kn_counts = sorted((k, 100.0 * v / total) for k, v in selected_kn_counts.items())
        print(" ".join("{}:{:.2f}".format(k, v) for k, v in selected_kn_counts))
    if len(trg_selected_kn_counts) > 0:
        print("trg_selected_kn_counts")
        total = sum(trg_selected_kn_counts.values())
        selected_kn_counts = sorted(
            (k, 100.0 * v / total) for k, v in trg_selected_kn_counts.items()
        )
        print(" ".join("{}:{:.2f}".format(k, v) for k, v in selected_kn_counts))
    assert len(results) == len(
        results2
    ), "Error, Length of results and results2 should be consistent!!"
    return zip(results, results2)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--prefix_path", type=str, required=True, help="Prefix path to the saved model"
    )
    argparser.add_argument("--in_path", type=str, required=True, help="Input file.")
    argparser.add_argument("--out_path", type=str, required=True, help="Output file.")
    argparser.add_argument("--beam_size", type=int, default=5)
    argparser.add_argument("--batch_size", type=int, default=20)
    argparser.add_argument("--max_decoding_step", type=int, default=50)
    argparser.add_argument("--checkpoint_step", type=int, default=200)
    args, unparsed = argparser.parse_known_args()
    FLAGS = config_utils.load_config(args.prefix_path + ".config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(
        "device: {}, n_gpu: {}, grad_accum_steps: {}".format(device, n_gpu, FLAGS.grad_accum_steps)
    )

    print("Loading existing vocab from {}...".format(FLAGS.save_data))
    words, word2id, concepts, concept2id, relations, relation2id, word_rels, word_rel2id = load_vocab_new(FLAGS.save_data)

    if FLAGS.use_bpe_pretrain:
        # word_emb = torch.load(FLAGS.save_data + "/word_emb.pt")
        word_emb = None
        con_emb = torch.load(FLAGS.save_data + "/concept_emb.pt")
    else:
        word_emb = None
        con_emb = None
    
    worker_init_fn = lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
    tokenize_fn = tokenize
    word_vocab, concept_vocab, relation_vocab, word_rel_vocab = word2id, concept2id, relation2id, word_rel2id

    print("Loading data and making batches")
    test_set = torch.load(FLAGS.save_data + "/test_data.pt")
    test_loader = test_set.GetDataloader(batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)
    print("Num examples = {}".format(len(test_set.instance)))

    checkpoint_step = "." + str(args.checkpoint_step) if args.checkpoint_step != 200 else ""
    best_checkpoint_path = args.prefix_path + "_best.checkpoint.bin" + checkpoint_step
    if os.path.exists(best_checkpoint_path):
        print("Loading best checkpoint...")
        checkpoint = torch.load(best_checkpoint_path)
    else:
        print('File {} not exist!!!'.format(best_checkpoint_path))
        exit()

    # model = DualTransformer(FLAGS, word_emb, con_emb, word2id, concept2id, relation2id)
    model = DualTransformer(FLAGS, word_emb, con_emb, word2id, concept2id, relation2id, word_rel2id)
    print(model)
    print(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )
    if n_gpu > 1:
        model = nn.DataParallel(model)
    if checkpoint is not None:
        new_pre = {}
        for k, v in checkpoint["model_state_dict"].items():
            name = k[7:] if k.startswith("module") else k
            new_pre[name] = v
        model.load_state_dict(new_pre)
        # model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    best_accu = 0.0 if "best_accu" not in checkpoint else checkpoint["best_accu"]
    print("Current accuracy {:.4f}".format(best_accu))

    fout = open(args.out_path, "w", encoding="utf-8")
    fout2 = open(args.out_path + ".hyp", "w", encoding="utf-8")
    for jobj, trans in decode(model, test_loader, args.beam_size, args.max_decoding_step):
        # print(prd)
        # print(trans)
        # print('============')
        fout.write(jobj + "\n")
        fout2.write(trans + "\n")
    fout.close()
