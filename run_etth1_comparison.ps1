$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$python = "C:\Users\lenovo\.conda\envs\keyanpytorch\python.exe"

if (-not (Test-Path ".\dataset\ETT-small\ETTh1.csv")) {
    throw "Missing dataset\ETT-small\ETTh1.csv"
}

# Official iTransformer baseline. Parameters match the official ETTh1 script.
& $python run.py `
  --is_training 1 `
  --root_path .\dataset\ETT-small\ `
  --data_path ETTh1.csv `
  --model_id ETTh1_96_96_official `
  --model iTransformer `
  --data ETTh1 `
  --features M `
  --seq_len 96 `
  --pred_len 96 `
  --e_layers 2 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --des Official `
  --d_model 256 `
  --d_ff 256 `
  --seed 2023 `
  --itr 1 `
  --num_workers 0

# Same official backbone/configuration plus the GDE-CFM adapter.
& $python run.py `
  --is_training 1 `
  --root_path .\dataset\ETT-small\ `
  --data_path ETTh1.csv `
  --model_id ETTh1_96_96_GDE `
  --model iTransformer_GDE `
  --data ETTh1 `
  --features M `
  --seq_len 96 `
  --pred_len 96 `
  --e_layers 2 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --des GDE_CFM `
  --d_model 256 `
  --d_ff 256 `
  --seed 2023 `
  --gde_scales 1,2,4 `
  --gde_probe_times 0.5 `
  --gde_flow_layers 2 `
  --gde_flow_d_ff 256 `
  --lambda_cfm 0.1 `
  --itr 1 `
  --num_workers 0

