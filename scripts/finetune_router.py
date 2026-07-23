"""Fine-tune a small open-source model to BE the strict-JSON support router.

Run this in a CUDA GPU environment (your own GPU box, a rented instance, or
Colab). It LoRA-fine-tunes a small base model on the chat-format JSONL produced
by ``scripts/prepare_dataset.py`` and exports a GGUF you can serve with Ollama.

NOTE: This is intentionally NOT run on the demo's CPU machine. The heavy imports
(unsloth/torch/trl) live inside main() so this file still byte-compiles anywhere.

Quickstart (on the GPU box):
    pip install "unsloth[colab-new]" "trl<0.10" datasets
    python scripts/finetune_router.py \
        --data data/generated/router_train.jsonl \
        --base-model unsloth/Qwen2.5-1.5B-Instruct \
        --epochs 3 --export-gguf

Then import the GGUF into Ollama (see docs/FINETUNING.md) and set
APP_LLM_PROVIDER=local + APP_LOCAL_ROUTER_MODEL=<your-model>.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune the JSON router.")
    parser.add_argument("--data", default="data/generated/support_routing.jsonl")
    parser.add_argument(
        "--base-model",
        default="unsloth/Qwen2.5-1.5B-Instruct",
        help="Apache-2.0, strong at JSON. Alternatives: ...0.5B (faster), google/gemma-3-1b-it (multilingual).",
    )
    parser.add_argument("--output", default="outputs/router-lora")
    parser.add_argument("--gguf-dir", default="router-gguf")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 = use epochs.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--export-gguf", action="store_true", help="Export a q4_k_m GGUF for Ollama.")
    args = parser.parse_args()

    # Heavy, GPU-only imports — kept local so the file is importable anywhere.
    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported

    print(f"[finetune] Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    # LoRA: train a small adapter instead of the whole model (cheap + fast).
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Our JSONL already has a "messages" field (system/user/assistant). Render it
    # to a single training string with the model's own chat template.
    def formatting_prompts_func(examples: dict) -> dict:
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["messages"]
        ]
        return {"text": texts}

    print(f"[finetune] Loading dataset: {args.data}")
    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"[finetune] {len(dataset)} training examples")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )

    print("[finetune] Training...")
    trainer.train()

    print(f"[finetune] Saving LoRA adapter -> {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    if args.export_gguf:
        print(f"[finetune] Exporting GGUF (q4_k_m) -> {args.gguf_dir}")
        model.save_pretrained_gguf(args.gguf_dir, tokenizer, quantization_method="q4_k_m")
        print(
            "[finetune] Done. Next: import into Ollama and point the app at it.\n"
            "  See docs/FINETUNING.md (step 4 onwards)."
        )
    else:
        print("[finetune] Done (LoRA saved). Re-run with --export-gguf to produce an Ollama model.")

    # Free reference so a notebook session releases VRAM promptly.
    del trainer, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
