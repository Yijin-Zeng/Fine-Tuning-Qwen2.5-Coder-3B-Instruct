# scripts/train.py
import os, argparse, random, torch, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fmt_to_chat(tokenizer, example):
    inst = example.get("instruction","")
    inp  = example.get("input","")
    out  = example.get("output","")
    user = inst if not inp else f"{inst}\n\nInput:\n{inp}"
    messages = [
        # this is the prompt used by the Qwen team
        {"role": "system", "content": "You are an intelligent Python coding assistant. Write only valid Python code for the function, no explanations."},
        {"role": "user", "content": user},
        {"role": "assistant","content": out},
    ]
    # return text format here
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-3B-Instruct") 
    ap.add_argument("--train_path", default="data/train.jsonl")
    ap.add_argument("--val_path",   default="data/val.jsonl")
    ap.add_argument("--output_dir", default="models/qwen25coder3b_codealpaca_lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per_device_bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4) # small learning rate
    ap.add_argument("--lora_r", type=int, default=16) # rank 
    ap.add_argument("--lora_alpha", type=int, default=16) 
    ap.add_argument("--lora_dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    # tokenize
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # load 4-bit for accelerating training
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    model.config.use_cache = False

    # load datasets
    data_files = {"train": args.train_path, "validation": args.val_path}
    ds = load_dataset("json", data_files=data_files)
    train_ds = ds["train"].map(lambda ex: fmt_to_chat(tok, ex),
                               remove_columns=ds["train"].column_names,
                               desc="Formatting train")
    val_ds = ds["validation"].map(lambda ex: fmt_to_chat(tok, ex),
                                    remove_columns=ds["validation"].column_names,
                                    desc="Formatting val")
    # configure LoRA 
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    # configure SFTTrainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        eval_strategy="steps",
        eval_steps=50,                # run val every N steps
        save_strategy="steps",
        save_steps=50,                # save at same cadence as eval
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,    
        gradient_checkpointing=True,
        fp16=True      
        )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_cfg,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # early stopping
    )
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    tok.save_pretrained(os.path.join(args.output_dir, "adapter"))
    print(f"Saved LoRA adapter to: {args.output_dir}/adapter")

if __name__ == "__main__":
    main()