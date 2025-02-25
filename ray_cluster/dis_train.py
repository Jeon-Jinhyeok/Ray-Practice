import ray
import os
import torch

# 환경 변수 설정 - GPU 사용 및 메모리 관리 최적화
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 사용 가능한 GPU 명시
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # 메모리 단편화 방지

from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 데이터셋 전처리를 위한 클래스
class DataProcessor:
    def __init__(self, tokenizer, max_input_length, max_target_length):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
    def process_data(self, examples, field_document, field_summary):
        inputs = self.tokenizer(
            examples[field_document], 
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length"
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples[field_summary], 
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length"
            )
            
        inputs["labels"] = labels["input_ids"]
        
        # ignore_index=-100으로 패딩 토큰 마스킹
        labels_with_ignore_index = []
        for label in labels["input_ids"]:
            labels_with_ignore_index.append([
                -100 if token == self.tokenizer.pad_token_id else token 
                for token in label
            ])
            
        inputs["labels"] = labels_with_ignore_index
        return inputs

# Ray 클러스터 연결
ray.init(
    address="auto",
    runtime_env={
        "pip": ["transformers", "datasets", "torch", "accelerate", "tensorboardX"],
        "env_vars": {
            "HF_DATASETS_CACHE": "/tmp/huggingface_datasets_cache",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3"
        }
    }
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="/tmp/ray_results/bart_finetuning",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="/tmp/ray_results/logs",
    save_total_limit=2,
    fp16=True,
    fp16_full_eval=True,
    dataloader_num_workers=4,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    no_cuda=False,
    torch_compile=False
)

# Trainer를 상속한 커스텀 클래스 정의
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 모델의 디바이스 가져오기
        model_device = model.device
        
        # 모든 입력이 모델과 같은 디바이스에 있는지 확인
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.device != model_device:
                inputs[k] = v.to(model_device)
                
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch=num_items_in_batch)

def train_bart_fn():
    # GPU 확인 및 설정
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU instead.")
    
    # 데이터셋 캐싱 디렉토리 설정
    cache_dir = "/tmp/huggingface_datasets_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # BART 모델 및 토크나이저 로드
    model_name = "facebook/bart-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    print(f"Model loaded and moved to {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        print("XSUM 데이터셋 로드 중...")
        dataset = load_dataset("xsum", cache_dir=cache_dir)
        field_document = "document"
        field_summary = "summary"
    except Exception as e:
        print(f"XSUM 데이터셋 로드 중 오류 발생: {e}")
        print("CNN/Daily Mail 데이터셋으로 대체...")
        dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir=cache_dir)
        field_document = "article"
        field_summary = "highlights"
    
    print(f"데이터셋 로드 완료: {len(dataset['train'])} 학습 샘플")
    
    # 시퀀스 길이 설정
    MAX_INPUT_LENGTH = 1024
    MAX_TARGET_LENGTH = 256

    # 데이터 프로세서 인스턴스 생성
    processor = DataProcessor(tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
    def process_function(examples):
        return processor.process_data(examples, field_document, field_summary)

    print("데이터셋 전처리 시작...")
    tokenized_datasets = dataset.map(
        process_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="토큰화 및 전처리 중",
        num_proc=8
    )
    print("데이터셋 전처리 완료")
    
    # 데이터셋 형식 확인
    print("Train dataset format check:")
    print(f"Keys: {list(tokenized_datasets['train'][0].keys())}")
    
    # Trainer 초기화
    print("학습 준비 중...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    
    # 모델 학습
    print("모델 학습 시작...")
    trainer.train()
    
    # 최종 모델 저장
    print("최종 모델 저장 중...")
    trainer.save_model("/tmp/ray_results/bart_finetuning/final_model")
    
    # 정리
    print("학습 완료!")
    return {"status": "완료", "output_dir": "/tmp/ray_results/bart_finetuning"}

# Ray Train을 사용한 분산 학습 설정
trainer = TorchTrainer(
    train_bart_fn,
    scaling_config=ScalingConfig(
        num_workers=4,  # 4개의 GPU 모두 활용
        use_gpu=True,
        resources_per_worker={
            "CPU": 7, 
            "GPU": 1 
        }
    ),
    run_config=ray.train.RunConfig(
        storage_path="/tmp/ray_results",
        name="bart_finetuning_expanded"
    )
)

# 학습 실행
print("분산 학습 시작...")
results = trainer.fit()
print(f"학습 결과: {results}")