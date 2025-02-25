import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import prepare_model, prepare_data_loader
import time

# GPU 사용 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Ray 클러스터 초기화
ray.init(
    address="auto",
    runtime_env={
        "pip": ["torch", "torchvision", "tensorboard"],
        "env_vars": {"PYTHONWARNINGS": "ignore"}
    }
)

# 학습 함수 정의
def train_resnet_fn():
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # 데이터 전처리 및 증강
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # CIFAR-10 데이터셋 로드
    try:
        print("CIFAR-10 데이터셋 로드 중...")
        # 학습 데이터셋
        train_dataset = torchvision.datasets.CIFAR10(
            root='/tmp/data',
            train=True,
            download=True,
            transform=transform_train
        )
        
        # 검증 데이터셋
        val_dataset = torchvision.datasets.CIFAR10(
            root='/tmp/data',
            train=False,
            download=True,
            transform=transform_test
        )
        
        print(f"데이터셋 로드 완료: {len(train_dataset)} 학습 샘플, {len(val_dataset)} 검증 샘플")
        
        # 데이터 로더 생성
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4
        )
        
        # Ray Train을 위한 분산 데이터 로더 준비
        train_loader = prepare_data_loader(train_loader)
        val_loader = prepare_data_loader(val_loader)
        
        # ResNet50 모델 로드 (전이 학습)
        print("ResNet50 모델 로드 중...")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # CIFAR-10에 맞게 분류기 부분 수정 (10개 클래스)
        num_classes = 10
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # 모델을 GPU로 이동
        model = model.to(device)
        
        # Ray Train을 위한 분산 모델 준비
        model = prepare_model(model)
        
        # 손실 함수 및 옵티마이저 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        
        # 학습 설정
        num_epochs = 20
        best_acc = 0.0
        
        print("=== 학습 시작 ===")
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 학습 모드
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # 학습 루프
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 그래디언트 초기화
                optimizer.zero_grad()
                
                # 순전파 + 역전파 + 최적화
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # 통계
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_loader)}, "
                          f"Loss: {running_loss/(batch_idx+1):.4f}, "
                          f"Acc: {100.*correct/total:.2f}%")
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # 검증 모드
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # 학습률 스케줄러 업데이트
            scheduler.step(val_loss)
            
            # 결과 출력
            epoch_time = time.time() - start_time
            print(f"Epoch: {epoch+1}/{num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"Time: {epoch_time:.2f}s, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 최고 성능 모델 저장
            if val_acc > best_acc:
                print(f"Validation accuracy improved from {best_acc:.2f}% to {val_acc:.2f}%, saving model...")
                best_acc = val_acc
                
                # 저장 디렉토리 경로 생성
                save_dir = "/tmp/ray_results/resnet50_cifar10"
                os.makedirs(save_dir, exist_ok=True)
                print(f"저장 디렉토리 확인/생성: {save_dir}")
                
                # 모델 저장
                save_path = os.path.join(save_dir, "best_model.pth")
                torch.save(model.state_dict(), save_path)
                print(f"모델 저장 완료: {save_path}")
        
        print(f"학습 완료! 최고 정확도: {best_acc:.2f}%")
        return {"status": "완료", "best_accuracy": best_acc}
    
    except Exception as e:
        print(f"학습 중 오류 발생: {e}")
        raise e

# 결과 저장 디렉토리 미리 생성
result_dir = "/tmp/ray_results"
model_dir = os.path.join(result_dir, "resnet50_cifar10")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
print(f"결과 저장 디렉토리 생성 완료: {model_dir}")

# Ray Train을 사용한 분산 학습 설정
trainer = TorchTrainer(
    train_resnet_fn,
    scaling_config=ScalingConfig(
        num_workers=4,  # 4개의 GPU 모두 활용
        use_gpu=True,
        resources_per_worker={
            "CPU": 7, 
            "GPU": 1 
        }
    ),
    run_config=RunConfig(
        storage_path=result_dir,
        name="resnet50_cifar10",
        failure_config=ray.train.FailureConfig(max_failures=3)  # 최대 3번까지 재시도
    )
)

# 학습 실행
print("분산 학습 시작...")
try:
    results = trainer.fit()
    print(f"학습 결과: {results}")
except Exception as e:
    print(f"학습 실행 중 오류 발생: {e}")