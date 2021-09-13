CUDA_VISIBLE_DEVICES=0 python train.py --gradient_clip_val 1.0 \
                --max_epochs 6 \
                --default_root_dir logs \
                --gpus 1 \
                --batch_size 8 \
                --num_workers 4 --lr 1e-5 \
		--warmup_ratio 0.05
