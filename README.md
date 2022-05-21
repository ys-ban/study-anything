# study-anything
아무거나 공부한 거 그때마다 기록하기

# Computer Vision
- [competition-1](./competition-1-anon/):
  - 512\*384 크기의 RGB 이미지
  - multiclass classification 또는 multilabel classification
  - 사용수단: DL 전이학습(EfficientNet, ResNet, Inception)
  - 그 외 사용가능수단: Conv2dLSTM, Tranformer
  - 기타 메모:
    - oversampling과 focal loss, loss with weights의 장단점
    - K-fold validation에서 data leakage
    - multiclass와 multilabel의 차이
    - shallow feature와 dense feature를 어떻게 추출할지
    - label dependency
- [competition-3](./competition-3/):
  - Taco Dataset의 일부 sample로 다양한 사이즈의 이미지
  - semantic segmentation task
  - 사용수단: UNet3+ with Deepsupervision and Class Guide Module
  - 그 외 사용가능수단: HRNet, SwinTransformer 등 다양한 SOTA 모델
  - 기타 메모:
    - loss 변화를 이용한 성능향상을 시도했음(focal tversky loss, jaccard loss 등)
    - loss 선정 기준은 semantic segmentation loss 관련 논문
    - 하지만 작동하는 모델이 거의 없었음(PSPNet, FPN에는 확실한 성능향상이 있었으나 본 모델에는 그렇지 않음)
    - CGM을 customizing할 때, GAP에 비해 GMP가 잘 작동한 이유는 무엇인가?
- [Neural Style Transfer](./CV-model/)
  - Gram Matrix를 고차원 대상에 대한 distance 혹은 metric으로 사용했다는 것이 중요한 idea
- [ArcFace & Image Retrieval](./CV-model/ArcFace/)
  - class imbalance가 심할 때 image classification task를 image retrieval로 치환하여 접근
  - 마진에 대한 고찰
    - 일반적으로 2차원 평면 내에서 각도에 대한 고려를 한다면 사실 마진은 2pi/#(classes)를 넘지 않아야 함
    - 하지만 N차원 입장에서 볼 때, 마진은 생각보다 커도 됨
    - 성급하지만 3차원에서만 생각을 하자면 $4\pi \fallingdotseq {2\pi \sin^2{\theta}} \over {#(classes)}$ 정도까지 가능
    - 차원을 높힐 수록 마진이 넉넉해짐을 수학적으로 증명 가능
  - 장점:
    - class imbalance나 적은 데이터의 한계를 극복할 때 적용 가능
    - 모델의 복잡도가 높지 않아도 준수한 성능을 보임
  - 단점:
    - CPU 자원의 소모가 큼
    - 한 image를 분류할 때, 기존 추출한 데이터 수의 자승만큼 simularity 연산이 필요
    - 즉, 계산량이 기하급수적으로 증가하기에 실제 적용을 위해서 parellel computing을 사용할 필요 있음


# Tabular Data
- [competition-2](./competition-2/):
  - 구매로그 데이터
  - binary - 마지막 달의 다음 달 기준 고객의 구매 총액이 300이 남을지 예측
  - 사용수단: 의사결정나무, lightGBM
  - 그외 사용가능수단: 시계열(prophet, auto_arima, MLP, TabNet)
  - 기타 메모:
    - pycaret과 AutoML
    - Information Value
    - fine classing
    - coarse classing
    - seach the best imputing value
