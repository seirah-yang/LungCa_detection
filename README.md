# LungCa_detection
This project provides an end-to-end pipeline for preprocessing, augmentation, and training to perform reliable lesion detection on lung cancer diagnostic imaging data


## 파이프라인
    ① 이미지 수집 (Data Collection)
      폐 CT / X-ray 영상 확보 (DICOM, PNG, JPG 등)
      환자정보 제거(De-ID), 라벨 품질 중요
      
    ② 전처리 (Preprocessing)
      HU 윈도우링, intensity normalization, CLAHE 등
      병변 형태 보존 필수 — 강한 blur, rotation, hue 변환 금지
      
    ③ 데이터 증강 (Augmentation)
      Albumentations로 shift, scale, brightness, noise 등 적용
      의료영상 특화 구조보존(Structure-preserving) 원칙 준수
      
    ④ 데이터셋 분리
      train / val / test (예: 70/20/10%)
      환자 단위로 split (같은 환자 이미지가 여러 셋에 섞이면 안 됨)
      
    ⑤ 모델 로드 (Pretrained)
      yolov12m.pt (pretrained weights)
      일반 객체탐지용 → 의료 도메인에 맞게 finetuning
      
    ⑥ 학습 (Training)
      CT/X-ray용 전처리·증강 반영 후 학습
      작은 learning rate (1e-3~1e-4) 권장
      
    ⑦ 검증 (Validation)
      mAP, Precision, Recall 평가
      과적합 여부 확인, early stopping 적용
      
    ⑧ 슬라이싱 추론 (SAHI)
      큰 영상(>2048px)을 슬라이스별 탐지 후 병합
      작은 결절 검출률 향상에 도움
      
    ⑨ 테스트 (Test Set Evaluation)
      unseen 데이터에 대해 결과 검증
      병변 탐지 정확도, 누락 여부 평가
      
    ⑩ 결과 분석 (Visualization)
      bbox, heatmap, confusion matrix
      False Negative 최소화 중점
