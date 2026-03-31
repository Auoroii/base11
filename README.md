# MLKD for Speech Emotion Recognition (Noisy Conditions)

This repository reproduces MLKD-style SER under noisy conditions with PyTorch + HuggingFace Transformers.
The current recommended student setup is:
`dual_path_gate_pred` (MLKD + dual-path + enhancement frontend + predicted-condition adaptive gate).

## Environment
- Python >= 3.10
- Dependencies in `requirements.txt`

## Label Mapping
We keep four classes and merge `exc` into `hap`:
- `0 = ang`
- `1 = neu`
- `2 = sad`
- `3 = hap` (includes `exc`)

## Data: IEMOCAP Metadata
Assume you already have `IEMOCAP_full_release` with the official structure.

Build metadata CSV:
```bash
python tools/build_iemocap_metadata.py \
  --iemocap_root E:/IEMOCAP_full_release \
  --output_csv data/metadata.csv
```

The CSV columns are:
`path,label_str,label_id,speaker_id,session_id,utt_id`

## Data: RAVDESS Metadata (4-class mapping)
Build metadata CSV from RAVDESS filenames:
```bash
python tools/build_ravdess_metadata.py \
  --ravdess_root E:/RAVDESS \
  --output_csv data/ravdess_metadata.csv
```

Default label mapping:
- `05 -> ang`
- `01/02 -> neu` (`02` calm merged into neutral)
- `04 -> sad`
- `03 -> hap`

## Noise Data
Prepare `noise_root/` with long noise wavs:
`babble, f16, factory, hf, volvo` etc. Filenames should include these keywords.

Example:
```
E:/noise_root/babble_1.wav
E:/noise_root/factory_2.wav
```

## Main Method: Predicted-Condition Dual-Path Gate
### `dual_path_gate_pred` (recommended)
- Student extracts pooled raw and pooled enhanced representations.
- `ConditionEstimator` predicts:
  - `cond_feat`
  - `snr_logits_pred`
  - `noise_logits_pred`
  - `snr_value_pred`
- Adaptive gate uses **predicted** condition (`condition_source=predicted`) instead of oracle labels.
- Main model selection should use `eval_condition_mode=deploy`.

### Oracle upper bound
- `dual_path_gate_oracle` keeps oracle labels in gate at evaluation/training for upper-bound reference.

### Hybrid transition
- `dual_path_gate_hybrid` uses oracle-to-predicted smooth transition with:
  - `training.hybrid_warmup_epochs`
  - `training.hybrid_transition_epochs`

## Key Config Options
- `model.ablation`: `baseline|enhancement_only|dual_path_no_gate|dual_path_gate_oracle|dual_path_gate_pred|dual_path_gate_pred_reg|dual_path_gate_hybrid`
- `model.gate_condition_mode`: `none|oracle|predicted|hybrid`
- `training.eval_condition_mode`: `deploy|oracle`
- `model.condition_feat_dim`
- `model.use_snr_regression`
- `loss.lambda_snr_reg`
- `training.hybrid_warmup_epochs`
- `training.hybrid_transition_epochs`

## Ablations (A0-A6)
- `A0 baseline` -> `baseline`
- `A1 enhancement_only` -> `enhancement_only`
- `A2 dual_path_no_gate` -> `dual_path_no_gate`
- `A3 dual_path_gate_oracle` -> `dual_path_gate_oracle`
- `A4 dual_path_gate_pred` -> `dual_path_gate_pred` (main recommendation)
- `A5 dual_path_gate_pred_reg` -> `dual_path_gate_pred_reg`
- `A6 dual_path_gate_hybrid` -> `dual_path_gate_hybrid`

## Training: Single Fold (LOSO)
Teacher (clean):
```bash
python train_teacher.py \
  --metadata_csv data/metadata.csv \
  --fold_speaker Ses01F \
  --output_dir exp/fold_Ses01F/teacher
```

Student (recommended A4):
```bash
python train_student_mlkd.py \
  --metadata_csv data/metadata.csv \
  --fold_speaker Ses01F \
  --teacher_ckpt exp/fold_Ses01F/teacher/best_teacher.pt \
  --output_dir exp/fold_Ses01F/student_pred \
  --ablation dual_path_gate_pred \
  --gate-condition-mode predicted \
  --eval-condition-mode deploy \
  --max-epochs 60 \
  --scheduler cosine \
  --warmup-epochs 5 \
  --warmup-start-lr 5e-6 \
  --lr 5e-5 \
  --min-lr 5e-6 \
  --early-stop-patience 12 \
  --monitor full_mean_UA \
  --full-val-every 5 \
  --grad-clip 1.0 \
  --loss-schedule staged \
  --save-best-full \
  --save-best-proxy
```

Student (A5, with SNR regression):
```bash
python train_student_mlkd.py \
  --metadata_csv data/metadata.csv \
  --fold_speaker Ses01F \
  --teacher_ckpt exp/fold_Ses01F/teacher/best_teacher.pt \
  --output_dir exp/fold_Ses01F/student_pred_reg \
  --ablation dual_path_gate_pred_reg \
  --use-snr-regression \
  --lambda-snr-reg 0.02 \
  --eval-condition-mode deploy
```

Resume examples:
```bash
python train_student_mlkd.py ... --resume last
python train_student_mlkd.py ... --resume best_full
python train_student_mlkd.py ... --resume best_proxy
python train_student_mlkd.py ... --resume exp/fold_Ses01F/student/last.pt
```

## Evaluation
### Deploy evaluation (main)
`deploy` forbids direct oracle condition injection into gate.

```bash
python evaluate.py \
  --metadata_csv data/metadata.csv \
  --fold_speaker Ses01F \
  --model_type student \
  --checkpoint exp/fold_Ses01F/student_pred/best_full.pt \
  --eval_condition_mode deploy \
  --all_conditions
```

### Oracle evaluation (upper bound)
```bash
python evaluate.py \
  --metadata_csv data/metadata.csv \
  --fold_speaker Ses01F \
  --model_type student \
  --checkpoint exp/fold_Ses01F/student_pred/best_full.pt \
  --eval_condition_mode oracle \
  --all_conditions
```

### Clean / single condition evaluation
```bash
python evaluate.py \
  --metadata_csv data/metadata.csv \
  --fold_speaker Ses01F \
  --model_type student \
  --checkpoint exp/fold_Ses01F/student_pred/best_full.pt

python evaluate.py \
  --metadata_csv data/metadata.csv \
  --fold_speaker Ses01F \
  --model_type student \
  --checkpoint exp/fold_Ses01F/student_pred/best_full.pt \
  --noise_type babble \
  --snr_db 10 \
  --eval_condition_mode deploy
```

UA (Unweighted Accuracy) is computed as mean per-class recall from the confusion matrix.

## RAVDESS Training (8:1:1 random split)
```bash
python train_teacher.py \
  --config configs/ravdess.yaml \
  --output_dir exp_ravdess/teacher

python train_student_mlkd.py \
  --config configs/ravdess.yaml \
  --teacher_ckpt exp_ravdess/teacher/best_teacher.pt \
  --output_dir exp_ravdess/student

python evaluate.py \
  --config configs/ravdess.yaml \
  --model_type student \
  --checkpoint exp_ravdess/student/best_full.pt \
  --eval_condition_mode deploy \
  --all_conditions
```

## Recommended Experiment Flow
1. Run A0/A1/A2 to verify enhancement and dual-path gains.
2. Run A3 (`dual_path_gate_oracle`) as upper-bound reference.
3. Run A4 (`dual_path_gate_pred`) as main deployable method.
4. Run A5 (`dual_path_gate_pred_reg`) to check stability/robustness.
5. Optionally run A6 (`dual_path_gate_hybrid`) for smoother transition training.
6. Report main comparisons with `eval_condition_mode=deploy`, and add `oracle` eval as analysis.

## Full 10-Speaker LOSO
```bash
python run_loso.py --config configs/default.yaml
```

Outputs:
- `exp/loso_results.csv` (all folds, all conditions)
- `exp/loso_summary.csv` (mean UA per noise type and SNR)

## Checkpoints
- Teacher: `best_teacher.pt`
- Student: `last.pt`, `best_proxy.pt`, `best_full.pt` (`best_student.pt` kept for compatibility)

Checkpoints store `model_state`, `model_config`, and `model_meta` for reproducibility and reevaluation.
