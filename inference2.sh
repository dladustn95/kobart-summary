for sub in economy finance global_economy industry life real_estate small_business
do
  for var in $(seq 0 1 8);
  do
    CUDA_VISIBLE_DEVICES=1 python -W ignore inference.py --hparams logs/tb_logs/default/version_4/hparams.yaml --model_binary logs/model_chp2/epoch\=0${var}.ckpt --testfile data/2020_summ_test_data/${sub}.src --outputfile data/2020_summ_test_data/model2_${var}_${sub}_result.txt
  done

  for var in $(seq 0 1 5);
  do
    CUDA_VISIBLE_DEVICES=1 python -W ignore inference.py --hparams logs/tb_logs/default/version_5/hparams.yaml --model_binary logs/model_chp3/epoch\=0${var}.ckpt --testfile data/2020_summ_test_data/${sub}.src --outputfile data/2020_summ_test_data/model3_${var}_${sub}_result.txt
  done
done
