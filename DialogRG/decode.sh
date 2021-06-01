CUDA_VISIBLE_DEVICES=1 python decode.py --prefix_path $1 \
    --in_path $2 --out_path $3 --checkpoint_step $4 --beam_size 5
