#!/bin/bash
dev=1
setting=logs-dual-tiny-coref-split
setting=logs-dual-big-coref-split

model_path=$setting/wiki.dailydiag_bleu10
test_path=../data/semi-coref/test
ref_path=../data/semi-coref/test.tgt
out_file=$setting/eval.log
for((step=50;step<=200;step+=50))
do
hyp_path=$setting/test-$step
CUDA_VISIBLE_DEVICES=$dev python decode.py --prefix_path $model_path --in_path $test_path --out_path $hyp_path --checkpoint_step $step --beam_size 5
echo "================step${step}=============" >> $out_file
python eval_v2.py ${hyp_path}.hyp $ref_path >> $out_file
done
