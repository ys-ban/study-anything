# study-anything
아무거나 공부한 거 그때마다 기록하기

# Computer Vision
- [competition-1](./competition-1-anon/):
  - 512\*384 크기의 RGB 이미지
  - multiclass classification 또는 multilabel classification
  - 사용수단: DL 전이학습(EfficientNet, ResNet, Inception)
  - 그외 사용가능수단: Conv2dLSTM, Tranformer
  - 기타 메모:
    - oversampling과 focal loss, loss with weights의 장단점
    - K-fold validation에서 data leakage
    - multiclass와 multilabel의 차이
    - shallow feature와 dense feature를 어떻게 추출할지
    - label dependency


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
