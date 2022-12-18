python main_view.py \
  --log_dir base_moco_r2d_view_ucf \
  --ckp_dir base_moco_r2d_view_ucf \
  --dataset ucf101 \
  -a r2plus1d_18 \
  --lr 0.01 \
  -cs 112 \
  -fpc 16 \
  -b 40 \
  -j 16 \
  --epochs 201 \
  --schedule 120 160 \
  --aug_plus \
  --mlp \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  dataset/ucf-101
