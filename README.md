# LungCa_detection
This project provides an end-to-end pipeline for preprocessing, augmentation, and training to perform reliable lesion detection on lung cancer diagnostic imaging data

## 1. Overview 
  본 프로젝트는 폐암 진단용 의료영상(CT 및 X-ray) 데이터를 대상으로 신뢰성 높은 병변(lesion) 탐지 모델을 개발하기 위한 [전처리 → 증강 → 학습 → 검증 → 슬라이싱 추론(SAHI)]의 딥러닝 파이프라인을 구현한 프로젝트입니다.
  
  1) 목표 : 임상현장에서 활용 가능한 AI 기반의 폐 병변 자동 탐지

## 2. Model Architecture 

    1) Data Preprocessing
        HU Windowing, CLAHE 적용 (대비 향상)
        OpenCV, Pydicom
    2) Data Augmentation
        구조 보존 중심 변환 (Rotate, Flip, Blur 등)
        Albumentations
    3) Fine-tuning
        Pretrained YOLOv12m 모델 미세조정
        Ultralytics 8.3.221
    4) Validation
        mAP, Precision, Recall 계산
        YOLO val()
    4) SAHI Inference
        Slice-Aided Hyper Inference로 고해상도 영상 탐지
        SAHI
    5) Report Generation
        CSV + 그래프 기반 자동 리포트 생성
        pandas, matplotlib
   
## 3. Project Structure
    1) Lung-Cancer-Detection-(Model)-1/
       ├── train/
       ├── valid/
       ├── test/
       ├── data.yaml
       └── labels/
    
    2) yolov12m_finetune2/
       └── weights/best.pt
    
    3) outputs/
       ├── sahi_results/
       ├── metrics_report/
       │   ├── yolov12m_metrics.csv
       │   └── yolov12m_performance.png
       └── yolov12m_test_pred/

## 4. Pipeline       
	1) 환경 준비 및 데이터 로드 
    2) 전처리 (Preprocessing)
        
		(1) CT:
    	  •	HU 변환: window = (-1200, 400) (lung window)
    	  •	정규화: 0–1 또는 0–255
    	  •	voxel spacing 보정 필요 시 resample (1×1×1 mm)
    
        (2) X-ray:
    	  •	CLAHE(clip=2.0, tileGridSize=(8,8))
    	  •	Gaussian Denoise (σ≤1)
    	  •	intensity normalization
    
        (3) 의료영상 전처리 주의사항:
    	  •	Hue/Saturation 변경 (의학적 의미 상실)
    	  •	VerticalFlip (해부학적 방향 왜곡)
    	  •	Elastic 변형 강하게 적용 (병변 형태 손상)
        
```python
        import cv2, albumintations as A    
        transform = A.Compose([
            A.RandomRotate90(p=0.2),
            A.HorizontalFlip(p=0.3),
            A.Blur(p=0.1, blur_limit=(3,7)),
            A.CLAHE(p=0.2, clip_limit=(1.0,4.0), tile_grid_size=(8,8)),
            A.RandomBrightnessContrast(p=0.2)
        ])
        
        image = cv2.imread("sample_ct.png")
        augmented = transform(image=image)["image"]
        cv2.imwrite("augmented_ct.png", augmented)
```
    3) 데이터 증강 (Augmentation)
      • Albumentations로 shift, scale, brightness, noise 등 적용
      • 의료영상 특화 구조보존(Structure-preserving) 고려하여 이미지 증강 적용
      
    4) Fine tuning
      • yolov12m.pt (pretrained weights)를 일반 객체탐지용에서 의료 도메인에 맞게 finetuning
      • Fine-tuned YOLOv12m 모델은 폐암 병변 자동 탐지에서 90% 이상의 탐지 정확도 달성
      
    5) 학습 (Training)
      • CT/X-ray용 전처리·증강 반영 후 학습
      • 작은 learning rate (1e-3~1e-4) 권장

```python 
        from ultralytics import YOLO
        
        model = YOLO("yolov12m.pt")  # Pretrained weights
        
        model.train(
            data="/home/alpaco/homework/LungCa detection /Lung-Cancer-Detection-(Model)-1/data.yaml",
            epochs=10,
            imgsz=640,
            batch=16,
            lr0=0.001,
            optimizer="SGD",
            device="cuda",
            project="runs_lung",
            name="yolov12m_finetune2"
        )
```      
    6) 검증 (Validation)
      • mAP, Precision, Recall 평가
      • 과적합 여부 확인
      
```python
        results = model.val(data="data.yaml", imgsz=640)

        print("mAP50-95:", results.box.map)
        print("mAP50:", results.box.map50)
        print("Precision:", results.box.p)
        print("Recall:", results.box.r)
```
    7) 슬라이싱 추론 (SAHI)
      • 큰 영상(>2048px)을 슬라이스별 탐지 후 병합
      • 작은 결절 검출률 향상에 도움
      
```python
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path="/home/alpaco/homework/LungCa detection /yolov12m_finetune2/weights/best.pt",
            confidence_threshold=0.3,
            device="cuda"
        )
        
        result = get_sliced_prediction(
            image="sample_ct.png",
            detection_model=detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        result.export_visuals(export_dir="outputs/sahi_results/")
```

    8) 테스트 (Test Set Evaluation)
      • unseen 데이터에 대해 결과 검증
      • 병변 탐지 정확도, 누락 여부 평가
        
    9) 결과 (Visualization)
      • mAP50 = 0.9016 : 병변 탐지 정확도 우수
      • mAP50–95 = 0.4701 : 경계 정밀도 중간 수준
      • Precision = 0.879 : False positive 낮음
      • Recall = 0.835 : 대부분의 병변 탐지 성공
      
## Result & 

    1) mAP50 > 0.9 실제 연구 데이터셋 기반 R&D용 모델로 사용 가능하다. 

    2) mAP50–95 = 0.47으로 병변 위치 오차는 존재하지만, 탐지 누락률은 낮다. 

[graph 1. Performance]()

    3) 폐 결절 자동 탐지 및 GGO(ground-glass opacity) 인식에 충분히 유효한 성능을 보인다.
[img 1. Traing result1](https://github.com/seirah-yang/LungCa_detection/blob/main/prediction_visual.png)
[img 2. Traing result2](https://github.com/seirah-yang/LungCa_detection/blob/main/prediction_visual(2).png)
    
    4) Fine-tuned YOLOv12m 모델은 폐암 병변 자동 탐지에서 90% 이상의 탐지 정도를 보이며 SAHI를 통해 고해상도 이미지 탐지 성능을 향상시켰다. 
	
## Summary

본 프로젝트는 “의료영상의 AI기반 병변 탐지 파이프라인”을 완성한 사례로 전처리–증강–학습–평가–시각화의 전 과정을 End-to-End로 통합했으며,
폐암 조기 진단 AI 연구의 기초 데이터셋 및 모델 실험용으로 활용 가능합니다.
      
## Author 
**양 소 라 | RN, BSN, MSN** 

    Clinical Data Science Researcher
    
    AI Developer (End-to-End Clinical AI Bootcamp, AlphaCo)
    
    Domain Focus: Clinical Data Management & Digital Medicine
    				- DCT
    				- CDISC/CDASH
    				- AI for eCRF & NLP-based Document Automation
