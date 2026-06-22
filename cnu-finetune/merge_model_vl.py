from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

BASE_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
LORA_PATH = "./output/qwen2.5-vl-32b-cnu-qlora-v7"
MERGED_PATH = "./output/qwen2.5-vl-32b-cnu-merged-v7"

print("베이스 모델 로드 중...")
from transformers import Qwen2_5_VLForConditionalGeneration
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)

print("LoRA merge 중...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()

print("저장 중...")
model.save_pretrained(MERGED_PATH)

processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
processor.save_pretrained(MERGED_PATH)
print(f"완료! → {MERGED_PATH}")
