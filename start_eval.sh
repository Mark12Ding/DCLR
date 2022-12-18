python -W ignore main_lincls.py \
  --log_dir log_finetune_18_ucf \
  --ckp_dir log_finetune_18_ucf \
  -a r2plus1d_18 \
  --num_class 101 \
  --lr 0.1 \
  --lr_decay 0.1 \
  --wd 0.0001 \
  -fpc 16 \
  -cpv 10 \
  -b 64 \
  -j 32 \
  --finetune \
  --pretrained base_simclr_r2d_view_ucf_18_bs=3_lr=0.005_cs=112_fpc=16/checkpoint_0199.pth.tar \
  --epochs 10 \
  --schedule 6 8 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  dataset/ucf-101
