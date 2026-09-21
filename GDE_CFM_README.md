# GDE-CFM integration into the official iTransformer repository

This branch starts from `thuml/iTransformer` and keeps the official data loaders, experiment class, normalization, inverted embedding, Transformer encoder, projection head, early stopping, and evaluation metrics.

## Added files

- `model/gde_cfm_modules.py`: multi-scale velocity, velocity encoder, conditional vector field, CFM loss, flow probe, GDE projection, and gated injection.
- `model/iTransformer_GDE.py`: the official iTransformer backbone plus the GDE-CFM adapter.
- `scripts/multivariate_forecasting/ETT/iTransformer_GDE_ETTh1.sh`: Linux ETTh1 command.
- `run_etth1_comparison.ps1`: Windows script that runs the official baseline and GDE model with matched settings.

## Minimal modifications to official files

- `experiments/exp_basic.py`: registers `iTransformer_GDE`.
- `experiments/exp_long_term_forecasting.py`: accepts the model's optional CFM loss while leaving official models unchanged.
- `run.py`: adds GDE arguments and a configurable random seed.
- `utils/tools.py`: replaces removed NumPy alias `np.Inf` with `np.inf` for NumPy 2.x compatibility.

The original `model/iTransformer.py` is intentionally unchanged.

## Data

Place the public ETTh1 file at:

```text
dataset/ETT-small/ETTh1.csv
```

The dataset and generated checkpoints/results are excluded by `.gitignore`.

## Run on Windows

```powershell
conda activate keyanpytorch
cd "C:\Users\lenovo\Documents\ChatGPT\flow matching\iTransformer-official-gde"
powershell -ExecutionPolicy Bypass -File .\run_etth1_comparison.ps1
```

The two runs use the same official ETTh1 96-to-96 backbone settings. Do not compare results from unmatched smoke-test configurations.

