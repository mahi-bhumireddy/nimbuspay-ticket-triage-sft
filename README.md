# NimbusPay Ticket Triage — Qwen2.5-3B Fine-Tuning

Supervised fine-tuning (SFT) of **Qwen2.5-3B-Instruct** using 4-bit QLoRA and 16-bit LoRA to convert multi-channel customer support tickets (English & Hinglish) into validated 8-field triage JSON objects.

## Repository Structure
- `notebook.ipynb`: End-to-end data audit, training, evaluation, and inference notebook
- `data/train.jsonl`, `data/dev.jsonl`, `data/test.jsonl`: Train, 200-row dev, and 400-row blind test splits
- `score.py`: Official schema validation and slice evaluation harness
- `predictions_test.jsonl`: Final 400/400 schema-valid test predictions
- `outputs/experiment_results.csv`: Summary metrics across all 5 experiments
- `outputs/exp1_raw_qlora_r16/`: Best model adapter (`95.0%` EM, `98.88%` Mean Field Accuracy) and predictions
- `outputs/exp2_clean_qlora_r16/` through `outputs/exp5_hard_slice_aug/`: Ablation adapters and predictions
- `report.md`: Full experimental analysis, slice breakdown, and error analysis
- `README.md`: Project overview and reproduction instructions

## Results Summary (`dev.jsonl`, n = 200)

| Experiment | Setup | JSON Valid | Exact Match | Mean Field Acc | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`exp1_raw_qlora_r16`** | **Raw train data, 4-bit QLoRA (`r=16`), 3 epochs** | **100.0%** | **95.0%** | **98.88%** | **5.42 GB** |
| `exp2_clean_qlora_r16` | Cleaned train data, 4-bit QLoRA (`r=16`), 3 epochs | 100.0% | 91.0% | 98.25% | 5.43 GB |
| `exp3_clean_fp16_r16` | Cleaned train data, 16-bit LoRA (`r=16`), 2 epochs | 100.0% | 89.0% | 97.75% | 13.72 GB |
| `exp4_clean_qlora_r32` | Cleaned train data, 4-bit QLoRA (`r=32`), 2 epochs | 100.0% | 89.5% | 98.13% | 3.56 GB |
| `exp5_hard_slice_aug` | 2x Hard-slice oversampling (`r=16`), 2 epochs | 100.0% | 90.0% | 98.06% | 3.33 GB |

## Reproduction & Evaluation
1. **Environment:** Google Colab T4 GPU (16 GB VRAM) with `transformers`, `peft`, `bitsandbytes`, `trl`, and `datasets`.
2. **Pipeline:** Open `notebook.ipynb` and run Cells 1–8 sequentially. Saved adapters and predictions persist under `outputs/`.
3. **Evaluate Dev Predictions:** Run `python score.py --gold data/dev.jsonl --pred outputs/exp1_raw_qlora_r16/dev_preds.jsonl`.
