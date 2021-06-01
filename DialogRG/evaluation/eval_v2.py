# coding:utf-8
import sys
import json
from official_evaluator import calc_f1, calc_bleu, calc_distinct


def load_ne_dict(path):
    return [json.loads(line.strip()) for line in open(path, "r", encoding="utf-8")]


def load(path):
    output_dict = []
    data = [x.strip() for x in open(path, "r", encoding="utf-8").readlines()]
    try:
        for line in data:
            line = json.loads(line)
            output_dict.append(line["pred"])
            ref_dict.append(line["gold"])
    except json.decoder.JSONDecodeError:
        for i, line in enumerate(data):
            out = line.strip()
            output_dict.append(out)
    return output_dict


################
ne_dict = []
"""
for x in ['demo.test', 'demo.dev', 'demo.train']:
    if x in sys.argv[1]:
        ne_dict = load_ne_dict('./data/'+x+'.topic')
        break
"""
output_dict, ref_dict = load(sys.argv[1]), load(sys.argv[2])

data = []
for i, (out, ref) in enumerate(zip(output_dict, ref_dict)):
    ne = ne_dict[i] if len(ne_dict) > 0 else {}
    if "ne" in sys.argv[1:]:
        out = " ".join(ne.get(x, x) for x in out.split())
        ref = " ".join(ne.get(x, x) for x in ref.split())
    if "token" in sys.argv[1:]:
        out = [x for x in out.replace(" ", "")]
        ref = [x for x in ref.replace(" ", "")]
    else:
        out = out.split()
        ref = ref.lower().split()
    data.append((out, ref))

f1 = calc_f1(data)
bleu1, bleu2, bleu3, bleu4 = calc_bleu(data)
distinct1, distinct2 = calc_distinct(data)

print(
    "f1: {:.4f}, bleu1: {:.4f}, bleu2: {:.4f}, bleu3: {:.4f}, bleu4: {:.4f}, distinct1: {:.4f}, distinct2: {:.4f}".format(
        f1, bleu1, bleu2, bleu3, bleu4, distinct1, distinct2
    )
)