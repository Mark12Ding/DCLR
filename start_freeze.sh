python -W ignore main_lincls.py \
  --log_dir log_freeze_view_ucf \
  --ckp_dir log_freeze_view_ucf \
  -a r2plus1d_18 \
  --num_class 101 \
  --lr 5 \
  --lr_decay 0.1 \
  --wd 0 \
  -fpc 16 \
  -cpv 10 \
  -b 128 \
  -j 32 \
  --pretrained base_moco_r2d_view_camd_ucf_bs=8_lr=0.01_cs=112_fpc=16/checkpoint_0199.pth.tar \
  --epochs 10 \
  --schedule 6 8 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  dataset/ucf-101
