# Fine-Tuning Qwen2.5-Coder-3B-Instruct with LoRA

## Learning

This repository documents how I fine-tune the Qwen2.5-Coder-3B-Instruct model using LoRA (Low-Rank Adaptation) on the CoNaLa dataset.

## Documents

- `CheckData.ipynb` - Simple sanity check of the prepared data.
- `Evaluate.ipynb` - Evalute the fine tuned model on openai-humanevl dataset.
- `scr/PrepareData.py` - Data preparation script
- `scr/Train.py` - Training script with LoRA implementation

## Summaries

- **Vary LoRA rank settings**: Common values 8 and 16 are considered. No big performance differences observed.
- **Experiment with different fine-tuning datasets**: One possible reason for poorer performance is that the CoNaLa dataset does not import packages, whereas the OpenAI HumanEval tasks often require them. 
- **Broaden evaluation metrics**: It is possible that Qwen2.5 was already fine-tuned on HumanEval before release. To verify this, we could consider evaluating the fine-tuned model on an alternative benchmark.
