# infer.py
import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load(base, adapter):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok

def chat(model, tok, user_prompt, max_new_tokens=400):
    sys_msg = "You are a helpful Python coding assistant. Return only Python code unless asked to explain."
    msgs = [{"role":"system","content":sys_msg},
            {"role":"user","content":user_prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = tok.decode(out[0], skip_special_tokens=True)
    return gen

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--adapter", default="qwen25coder3b_codealpaca_lora/adapter")
    args = ap.parse_args()
    model, tok = load(args.base_model, args.adapter)
    print("Loaded. Type a task (Ctrl+C to quit).")
    while True:
        try:
            prompt = input("\n>>> ")
            out = chat(model, tok, prompt)
            out = out.replace("```python","").replace("```","").strip()
            print(out)
        except KeyboardInterrupt:
            break
