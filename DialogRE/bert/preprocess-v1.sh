BASE=../data-v1-coref
save_dir=workplace
setting=data-bin-v1-coref
mkdir -p $save_dir
python preprocess.py \
    --train_path $BASE/train \
    --dev_path $BASE/dev \
    --test_path $BASE/test \
    --devc_path $BASE/devc \
    --testc_path $BASE/testc \
    --save_data $save_dir/$setting
