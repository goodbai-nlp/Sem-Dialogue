save_dir=bert_f1_max-512-Hier-AMR-glayer-2-newvocab
save_dir=$1
cat $save_dir/logits_devc?.txt > $save_dir/logits_devc.txt
cat $save_dir/logits_testc?.txt > $save_dir/logits_testc.txt
python evaluate.py --devdata ../data-v1-coref/dev.re.json --testdata ../data-v1-coref/test.re.json --f1dev $save_dir/logits_dev.txt --f1test $save_dir/logits_test.txt --f1cdev $save_dir/logits_devc.txt --f1ctest $save_dir/logits_testc.txt 2>&1 | tee $save_dir/eval-f1.log
