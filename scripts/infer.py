"""推理示例：使用 KV cache 进行增量生成。"""
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer

from llm_lab import TritonMindConfig, TritonMindForCausalLM
import llm_lab

TOKENIZER_DIR = Path(llm_lab.__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/diy_checkpoint.pth",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = TritonMindConfig(**checkpoint["config"])
    model = TritonMindForCausalLM(config)
    model = model.to(config.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    max_new_tokens = 50  # 最大生成 token 数

    while True:
        prompt_text = input("Enter a prompt (回车退出): ")
        if not prompt_text.strip():
            break

        # 编码 prompt
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        prompt_ids = prompt_ids.to(config.device)

        # 第一次 forward：处理整个 prompt，得到 KV cache
        logits, presents = model(prompt_ids, use_cache=True)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated_ids = torch.cat([prompt_ids, next_token.unsqueeze(1)], dim=1)

        # 循环生成后续 token（使用 KV cache）
        for _ in range(max_new_tokens - 1):
            # 只送最后一个 token，用 KV cache
            new_token = next_token.unsqueeze(1)
            logits, presents = model(new_token, past_key_values=presents, use_cache=True)
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)

            # 遇到 eos_token 就停止
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break

        # 解码并打印
        generated_text = tokenizer.decode(generated_ids.squeeze(0).tolist(), skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        print("--------------------------------")

if __name__ == "__main__":
    main()
