#for var in $(seq 0 1 8);
#do
#  CUDA_VISIBLE_DEVICES=1 python -W ignore inference.py --hparams logs/tb_logs/default/version_4/hparams.yaml --model_binary logs/model_chp2/epoch\=0$var.ckpt --testfile data/test.src
#done

for var in $(seq 3 1 5);
do
  CUDA_VISIBLE_DEVICES=1 python -W ignore inference.py --hparams logs/tb_logs/default/version_5/hparams.yaml --model_binary logs/model_chp3/epoch\=0$var.ckpt --testfile data/test.src
done
