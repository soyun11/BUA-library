# finetune_vl.py
# Qwen2.5-VL-32B-Instruct QLoRA 파인튜닝 스크립트

import json
import os
import random
from pathlib import Path
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    data_path: str = "/NHNHOME/WORKSPACE/26moe002_H/sypark/browser-use/data/ft_train_vl_gemini_v5.jsonl"
    output_dir: str = "./output/qwen2.5-vl-32b-cnu-qlora-v7"
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 64   # 메모리 절약
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 32768
    save_steps: int = 50
    logging_steps: int = 10
    eval_ratio: float = 0.05


def load_data(data_path: str) -> list:
    data = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"총 {len(data)}개 데이터 로드 완료")
    return data


class VLDataset(TorchDataset):
    def __init__(self, data: list, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length
        self.samples = []

        print("데이터 전처리 중...")
        skipped = 0
        for item in data:
            messages = item["messages"]
            result = self._process(messages)
            if result is not None:
                self.samples.append(result)
            else:
                skipped += 1
        print(f"전처리 완료: {len(self.samples)}개 사용, {skipped}개 스킵")

    def _process(self, messages):
        parsed_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        content = parsed
                except Exception:
                    pass
            parsed_messages.append({"role": msg["role"], "content": content})

        try:
            text = self.processor.apply_chat_template(
                parsed_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            return None

        images = []
        for msg in parsed_messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        img_url = part.get("image_url", {}).get("url", "")
                        if img_url.startswith("data:image"):
                            try:
                                import base64
                                from PIL import Image
                                import io
                                header, data = img_url.split(",", 1)
                                img_bytes = base64.b64decode(data)
                                images.append(Image.open(io.BytesIO(img_bytes)))
                            except Exception:
                                pass

        try:
            if images:
                inputs = self.processor(
                    text=text,
                    images=images,
                    truncation=False,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=text,
                    truncation=False,
                    return_tensors="pt",
                )
        except Exception as e:
            return None

        if inputs["input_ids"].shape[1] > self.max_length:
            return None

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        assistant_token = self.processor.tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )
        end_token = self.processor.tokenizer.encode(
            "<|im_end|>", add_special_tokens=False
        )

        labels_list = labels.tolist()
        mask = True
        i = 0
        while i < len(labels_list):
            if labels_list[i:i+len(assistant_token)] == assistant_token:
                # <|im_start|>assistant 헤더 토큰은 마스킹
                for j in range(len(assistant_token)):
                    labels_list[i + j] = -100
                mask = False
                i += len(assistant_token)
                continue
            if labels_list[i:i+len(end_token)] == end_token:
                # system/user 구간의 <|im_end|>는 마스킹, assistant 구간은 유지
                if mask:
                    for j in range(len(end_token)):
                        labels_list[i + j] = -100
                mask = True
                i += len(end_token)
                continue
            if mask:
                labels_list[i] = -100
            i += 1
        labels = torch.tensor(labels_list)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            result["pixel_values"] = pv.squeeze(0) if pv.dim() == 5 else pv
        if "image_grid_thw" in inputs:
            # reshape(-1, 3) 으로 항상 2D 보장: (1,3)→(1,3), (N,3)→(N,3), (3,)→(1,3)
            result["image_grid_thw"] = inputs["image_grid_thw"].reshape(-1, 3)

        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def custom_collate_fn(batch):
    input_ids = pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True, padding_value=0
    )
    labels = pad_sequence(
        [b["labels"] for b in batch],
        batch_first=True, padding_value=-100
    )
    attention_mask = pad_sequence(
        [b["attention_mask"] for b in batch],
        batch_first=True, padding_value=0
    )

    result = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

    pv_list = [b["pixel_values"] for b in batch if "pixel_values" in b]
    if pv_list:
        try:
            result["pixel_values"] = torch.cat(pv_list, dim=0)
        except Exception as e:
            print(f"pixel_values cat 실패: {e}")

    thw_list = [b["image_grid_thw"] for b in batch if "image_grid_thw" in b]
    if thw_list:
        try:
            result["image_grid_thw"] = torch.cat(thw_list, dim=0)
        except Exception as e:
            print(f"image_grid_thw cat 실패: {e}")

    return result


def train(cfg: Config):
    print(f"\n{'='*50}")
    print(f"모델: {cfg.model_name}")
    print(f"데이터: {cfg.data_path}")
    print(f"출력: {cfg.output_dir}")
    print(f"QLoRA 4bit: {cfg.use_4bit}")
    print(f"max_seq_length: {cfg.max_seq_length}")
    print('='*50)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Processor 로드 (이미지 크기 줄임)
    print("\n📦 Processor 로드 중...")
    processor = AutoProcessor.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        min_pixels=128*28*28,
        max_pixels=512*28*28,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 2. 데이터 로드
    print("\n📂 데이터 로드 중...")
    raw_data = load_data(cfg.data_path)

    random.shuffle(raw_data)
    eval_size = max(1, int(len(raw_data) * cfg.eval_ratio))
    train_data = raw_data[eval_size:]
    eval_data = raw_data[:eval_size]
    print(f"Train: {len(train_data)}개 / Eval: {len(eval_data)}개")

    print("\n🔧 데이터셋 전처리 중...")
    train_dataset = VLDataset(train_data, processor, cfg.max_seq_length)
    eval_dataset = VLDataset(eval_data, processor, cfg.max_seq_length)

    # 3. 4bit 양자화
    print("\n🤖 모델 로드 중 (4bit QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.use_double_quant,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    # 4. LoRA 적용
    print("\n🔩 LoRA 적용 중...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 학습 설정
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 6. 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_collate_fn,
    )

    print("\n🚀 학습 시작!")
    trainer.train()

    # 7. 저장
    print("\n💾 모델 저장 중...")
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    print(f"✅ 저장 완료: {cfg.output_dir}")


if __name__ == "__main__":
    cfg = Config()
    train(cfg)