"""
DPO 微调脚本：基于 egpo_weak_dpo.py 生成的偏好数据训练模型。
使用 trl.DPOTrainer + LoRA 进行高效微调。

用法示例：
  python generate_html/dpo_train.py \
    --model-path ./your_model \
    --data-path data/egpo_weak_dpo.jsonl \
    --output-dir output/dpo_html \
    --num-epochs 3 \
    --lr 5e-6
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer


def _detect_device() -> str:
    """检测可用设备：cuda > mps > cpu"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dpo_dataset(data_path: str, max_samples: int | None = None) -> Dataset:
    """从 egpo_weak_dpo.py 输出的 JSONL 加载 DPO 数据集。"""
    records: List[Dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                {
                    "prompt": obj["prompt"],
                    "chosen": obj["chosen"],
                    "rejected": obj["rejected"],
                }
            )
            if max_samples and len(records) >= max_samples:
                break

    print(f"[data] loaded {len(records)} DPO pairs from {data_path}")
    return Dataset.from_list(records)


def build_model_and_tokenizer(
    model_path: str,
    use_qlora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
):
    """加载模型 + tokenizer，并挂载 LoRA。"""
    device = _detect_device()
    print(f"[model] detected device: {device}")

    quantization_config = None
    if use_qlora:
        if device != "cuda":
            print("[model] WARNING: QLoRA 需要 CUDA，当前设备不支持，已跳过量化")
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

    # MPS 上用 float32 更稳定；CUDA 用 bfloat16
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        device_map="auto" if device == "cuda" else None,
    )

    # 非 CUDA 且无量化时，手动搬到设备上
    if device != "cuda" and quantization_config is None:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    target_modules = [m.strip() for m in lora_target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description="DPO 微调：HTML 生成偏好对齐")
    ap.add_argument("--model-path", required=True, help="基座模型路径（HF 格式）")
    ap.add_argument("--data-path", default="data/egpo_weak_dpo.jsonl", help="DPO JSONL 数据路径")
    ap.add_argument("--output-dir", default="output/dpo_html", help="输出目录")
    ap.add_argument("--max-samples", type=int, default=None, help="最多使用多少条数据（调试用）")

    # 训练超参
    ap.add_argument("--num-epochs", type=int, default=3, help="训练轮数")
    ap.add_argument("--lr", type=float, default=5e-6, help="学习率")
    ap.add_argument("--batch-size", type=int, default=2, help="per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=8, help="梯度累积步数")
    ap.add_argument("--max-length", type=int, default=4096, help="最大序列长度")
    ap.add_argument("--max-prompt-length", type=int, default=2048, help="prompt 最大长度")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta（KL 惩罚系数）")
    ap.add_argument("--warmup-ratio", type=float, default=0.1, help="warmup 比例")

    # LoRA
    ap.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    ap.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    ap.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    ap.add_argument("--lora-target-modules", default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--use-qlora", action="store_true", help="启用 QLoRA（4bit 量化）")

    # 其他
    ap.add_argument("--save-steps", type=int, default=100, help="每 N 步保存 checkpoint")
    ap.add_argument("--logging-steps", type=int, default=10, help="每 N 步打印日志")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--bf16", action="store_true", default=True, help="使用 bf16 训练")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True, help="梯度检查点（省显存）")

    args = ap.parse_args()

    # 加载数据
    dataset = load_dpo_dataset(args.data_path, max_samples=args.max_samples)

    # 加载模型
    model, tokenizer = build_model_and_tokenizer(
        model_path=args.model_path,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )

    # ref_model: DPOTrainer 在 peft 模式下自动处理，无需手动加载
    device = _detect_device()
    use_bf16 = args.bf16 and device == "cuda"
    use_fp16 = device == "mps"  # MPS 上用 fp16 比 fp32 快，且比 bf16 兼容性好

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        remove_unused_columns=False,
        save_total_limit=3,
        report_to="none",
        lr_scheduler_type="cosine",
        dataloader_pin_memory=device == "cuda",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("[train] starting DPO training...")
    trainer.train()

    # 保存最终 LoRA 权重
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[train] done. model saved to {final_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
