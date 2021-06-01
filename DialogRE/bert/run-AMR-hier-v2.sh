BERT_BASE_DIR=/public/home/zhangyuegroup/baixuefeng/data/pretrained-model/bert-base-uncased
model=Hier
g_layer=2
dev=6
mode=$1
seed=42
setting=v2
setting=v2-coref
if [ "$mode" == "train" ]
then
echo "Start Training..."
for seed in 1 2 3 42 666
do
save_path=bert_f1_max-512-${model}-AMR-seed-${seed}-glayer-$g_layer-$setting-new
mkdir -p $save_path
CUDA_VISIBLE_DEVICES=$dev python -u run.py --task_name bert --do_train --do_eval --do_evalc \
	--architecture $model \
	--seed $seed \
	--vocab_file $BERT_BASE_DIR/vocab.txt \
	--bert_config_file $BERT_BASE_DIR/bert_config.json  \
	--init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   \
	--max_seq_length 512   \
	--train_batch_size 24   \
	--eval_batch_size 1   \
	--learning_rate 3e-5   \
	--num_train_epochs 30   \
	--output_dir $save_path  \
	--model_type "entity-max" \
	--entity_drop 0.1 \
	--g_num_layer $g_layer \
	--d_concept 512 \
	--save_data workplace/data-bin-$setting \
	--gradient_accumulation_steps 2 2>&1 | tee $save_path/run.log
done
elif [ "$mode" == "test" ]
then
echo "Start Testing..."
for seed in 1 2
do
save_path=bert_f1_max-512-${model}-AMR-seed-${seed}-glayer-$g_layer-$setting-new
CUDA_VISIBLE_DEVICES=$dev python -u run.py --task_name bert --do_eval --do_evalc \
	--architecture $model \
	--seed $seed \
	--vocab_file $BERT_BASE_DIR/vocab.txt \
	--bert_config_file $BERT_BASE_DIR/bert_config.json  \
	--init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   \
	--max_seq_length 512   \
	--train_batch_size 24   \
	--eval_batch_size 1   \
	--learning_rate 3e-5   \
	--num_train_epochs 30   \
	--output_dir $save_path  \
	--model_type "entity-max" \
	--entity_drop 0.1 \
	--g_num_layer $g_layer \
	--d_concept 512 \
	--save_data workplace/data-bin-$setting \
	--gradient_accumulation_steps 2 2>&1 | tee $save_path/eval.log
done
else
	echo "Invalid mode $mode!!!"
fi
