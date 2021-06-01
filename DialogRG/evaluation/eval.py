
import sys
import os
import json
from transformers import BertTokenizer


def load_ne_dict(path):
    return [json.loads(line.strip()) for line in open(path,'r')]

def load(path):
    output_dict = []
    ref_dict = []
    data = [x.strip() for x in open(path,'r').readlines()]
    try:
        for line in data:
            line = json.loads(line)
            output_dict.append(line['pred'])
            ref_dict.append(line['gold'])
    except json.decoder.JSONDecodeError:
        for i,line in enumerate(data):
            if i%3 == 1:
                ref = line.strip()
                ref_dict.append(ref)
            elif i%3 == 0:
                out = line.strip()
                output_dict.append(out)
    return output_dict, ref_dict

################

ne_dict = []
for x in ['demo.test', 'demo.dev', 'demo.train']:
    if x in sys.argv[1]:
        ne_dict = load_ne_dict('./data/'+x+'.topic')
        break

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

output_dict, ref_dict = load(sys.argv[1])

fout = open(sys.argv[1]+'.1best','w')
fref = open(sys.argv[1]+'.ref','w')
for i, (out, ref) in enumerate(zip(output_dict, ref_dict)):
    ne = ne_dict[i] if len(ne_dict) > 0 else {}
    if 'ne' in sys.argv[1:]:
        out = ' '.join(ne.get(x,x) for x in out.split())
        ref = ' '.join(ne.get(x,x) for x in ref.split())
    if 'token' in sys.argv[1:]:
        out = tokenizer.decode(tokenizer.encode(out, add_special_tokens=False), clean_up_tokenization_spaces=False)
        out = out.replace('[UNK]', '', 100)
        ref = tokenizer.decode(tokenizer.encode(ref, add_special_tokens=False), clean_up_tokenization_spaces=False)
        ref = ref.replace('[UNK]', '', 100)
    fout.write(out+'\n')
    fref.write(ref+'\n')
fout.close()
fref.close()

os.system('/home/lfsong/ws/mosesdecoder/scripts/generic/multi-bleu.perl %s.ref < %s.1best' \
        %(sys.argv[1], sys.argv[1]))
