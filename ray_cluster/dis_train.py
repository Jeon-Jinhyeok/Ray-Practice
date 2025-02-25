import ray
import os
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# ✅ Ray 클러스터 연결
ray.init(
    address="ray://34.214.222.28:10001",
    runtime_env={"storage_path": "/tmp/ray_results"}  # ✅ 저장소 경로 변경
)

# ✅ Hugging Face Dataset 경로 설정
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface_datasets_cache"

# ✅ 학습 설정
training_args = TrainingArguments(
    output_dir="/tmp/ray_results/checkpoints",  # ✅ 실험 결과 저장 경로 변경
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="/tmp/ray_results/logs",  # ✅ 로그 저장 경로 변경
)

def train_bart():
    model_name = "facebook/bart-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset = load_dataset("xsum", cache_dir="/tmp/huggingface_datasets_cache")

    def preprocess_data(examples):
        inputs = tokenizer(examples["document"], max_length=1024, truncation=True, padding="max_length")
        labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_datasets = dataset.map(preprocess_data, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    trainer.train()
    return "✅ Training Completed!"

trainer = TorchTrainer(
    train_bart,
    scaling_config=ScalingConfig(num_workers=3, use_gpu=True)
)

trainer.fit()
