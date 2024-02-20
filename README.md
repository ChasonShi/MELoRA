# MELoRA
Mini-Ensemble Low-Rank Adapter for Parameter-Efficient Fine-Tuning


1. Install dependencies

   ```bash
   conda create -n MELoRA python=3.10
   conda activate MELoRA
   pip install torch==2.0.1
   pip install -r requirements.txt

   ```

   ```bash
   cd peft-0.5.0
   pip install -e .
   ```

2. Run experiments

fill in the `--model_name_or_path` `--wandb_project` and `--output_dir` in `llama_finetune.sh` and `glue_finetune.sh` with the path to the model and the output directory.

### Instruction Tuning
```bash
bash llama_finetune.sh
```

### NLU

```bash
bash glue_finetune.sh
```
