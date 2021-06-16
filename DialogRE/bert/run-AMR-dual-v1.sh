BERT_BASE_DIR=/public/home/zhangyuegroup/baixuefeng/data/pretrained-model/bert-base-uncased
dev=1
model=Dual
g_layer=3
mode=$1
setting=v1-coref
save_path=bert_f1_max-512-${model}-AMR-glayer-$g_layer-$setting
mkdir -p $save_path
if [ "$mode" == "train" ]
then
echo "Start Training..."
CUDA_VISIBLE_DEVICES=$dev python -u run.py --task_name bert --do_train --do_eval --do_evalc \
	--architecture $model \
	--seed 42 \
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
elif [ "$mode" == "test" ]
then
echo "Start Testing..."
CUDA_VISIBLE_DEVICES=$dev python -u run.py --task_name bert --do_eval --do_evalc \
	--architecture $model \
	--seed 42 \
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
else
	echo "Invalid mode $mode!!!"
fi
