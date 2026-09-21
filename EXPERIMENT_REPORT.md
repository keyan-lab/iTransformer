# Official iTransformer + GDE-CFM: ETTh1 experiment report

## 1. Purpose

This experiment starts from the official THUML iTransformer repository and adds
the proposed GDE-CFM module without replacing the official iTransformer
backbone. The goal of this first experiment is to verify that the new module can
be trained and evaluated in the official benchmark pipeline and to obtain a
controlled baseline comparison.

## 2. Repository structure

- Official backbone (kept unchanged): `model/iTransformer.py`
- Added GDE-CFM components: `model/gde_cfm_modules.py`
- Added combined model: `model/iTransformer_GDE.py`
- Minimal training-pipeline integration:
  `experiments/exp_basic.py`, `experiments/exp_long_term_forecasting.py`,
  and `run.py`
- Reproduction command: `run_etth1_comparison.ps1`

The earlier lightweight prototype is not used for the results in this report.

## 3. Controlled experimental setting

Both models use exactly the same official ETTh1 benchmark setup:

| Item | Setting |
|---|---|
| Dataset | ETTh1 |
| Task | Multivariate long-term forecasting |
| Input length | 96 |
| Prediction length | 96 |
| Number of variables | 7 |
| iTransformer encoder layers | 2 |
| `d_model` / `d_ff` | 256 / 256 |
| Batch size | 32 |
| Maximum epochs | 10 |
| Learning rate | 0.0001 |
| Early-stopping patience | 3 |
| Random seed | 2023 |

The enhanced model additionally uses scales `1,2,4`, probe time `0.5`, two
flow blocks, and `lambda_cfm = 0.1`.

## 4. Results

| Model | Test MSE | Test MAE | Relative MSE change | Relative MAE change |
|---|---:|---:|---:|---:|
| Official iTransformer | **0.385807** | **0.404540** | reference | reference |
| iTransformer + GDE-CFM | 0.392986 | 0.409789 | +1.86% | +1.30% |

Lower is better. Therefore, the present GDE-CFM configuration does not improve
forecast accuracy over the official baseline. It nevertheless verifies that
the complete official data loading, training, validation, checkpointing, and
test pipeline works with the added module.

The official baseline contains 841,568 trainable parameters. The current
enhanced model contains 2,794,738 parameters (1,953,170 additional parameters),
so its increased computational cost must be considered when interpreting the
result.

## 5. What can and cannot be concluded

The experiment supports the following conclusions:

1. The module has been integrated into the official repository rather than a
   separately reimplemented iTransformer.
2. The official baseline is reproducible in the local environment.
3. The enhanced model can complete end-to-end training and testing.
4. Under this single initial configuration, the added module slightly worsens
   ETTh1 96-to-96 prediction accuracy and incurs substantially more computation.

It does **not** yet show that the GDE-CFM idea is ineffective. Only one loss
weight, one fusion design, one seed, and one prediction horizon have been
tested. A defensible next stage is an ablation study of `lambda_cfm`, the GDE
residual gate, probe times, and derivative scales, followed by repeated runs
with several random seeds.

## 6. Reproduction

From PowerShell in the repository root:

```powershell
conda activate keyanpytorch
.\run_etth1_comparison.ps1
```

The script first runs the unchanged official model and then the enhanced model
with matched benchmark parameters.
