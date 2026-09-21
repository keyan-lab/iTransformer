export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer_GDE

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_GDE \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'GDE_CFM' \
  --d_model 256 \
  --d_ff 256 \
  --gde_scales 1,2,4 \
  --gde_probe_times 0.5 \
  --gde_flow_layers 2 \
  --gde_flow_d_ff 256 \
  --lambda_cfm 0.1 \
  --itr 1

