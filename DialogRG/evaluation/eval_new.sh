#!/bin/bash
dev=2
setting=logs-dual-tiny-coref-split
setting=logs-dual-big-coref-split
setting=logs-dual-big-coref-v2-split-10
setting=logs-dual-big-coref-v2-split-10-g3
setting=logs-dual-big-coref-v2-split-10-g5
setting=logs-dual-big-coref-v2-split-10-sentfirst-addnorm
setting=logs-dual-big-coref-v2-split-10-graphfirst-addnorm

model_path=$setting/wiki.dailydiag_bleu10
test_path=../data/semi-coref-v2/test
ref_path=../data/semi-coref-v2/test.tgt
out_file=$setting/eval.log
for((step=50;step<=200;step+=10))
do
hyp_path=$setting/test-$step
CUDA_VISIBLE_DEVICES=$dev python decode.py --prefix_path $model_path --in_path $test_path --out_path $hyp_path --checkpoint_step $step --beam_size 5 --batch_size 40
echo "================step${step}=============" >> $out_file
python eval_v2.py ${hyp_path}.hyp $ref_path >> $out_file
done
