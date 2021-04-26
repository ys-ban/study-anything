# Wrap Up
- 워크플로우
  1. 주피터노트북 기반 코딩, 학습
     - 장점
       1. EDA에 적합
       2. 빠른 피보팅, 학습, data wrangling, visualization에 용이
       3. 데이터에 대한 감, 특징을 파악하기 용이
       4. 짧은 코드인 상황에서는 생산성인 매우 높음
     - 단점
       1. 코드가 길어질수록 생산성이 낮아짐
       2. 재사용성 하락
  2. 모듈화 + 주피터노트북 기반 코딩, 학습
     - 장점
       1. 모델, 코드 복잡도가 올라간 상황에서 생산성이 높아짐
       2. 재사용성 증가
     - 단점 
       1. 생산성, 확장성을 고려한 코딩 필요
       2. 초기 문제 상황에 대한 고려를 충분히 해봐야 함
  3. 피드백
     - 초기 EDA와 리서치가 부족했음
     - EDA와 리서치를 통해 문제를 이해하고 접근방법에 대한 고민이 충분히 고민해야 함
     - 모든 과정을 터미널로 실행하는 모듈화 시도 필요

- 데이터셋과 모델
  1. 데이터셋
     1. 특성1(3가지), 특성2(3가지), 특성3(2가지)을 조합한 18개의 레이블
     2. 특성이 조합되면서 특정 데이터 세그멘트가 최대 25배 정도의 수량 차이
     3. 직접 확인해본 결과 이미지의 형태는 유사(대상의 위치, 사이즈 등)
     4. 모든 이미지는 512\*384의 RGB 데이터
  2. 모델
     1. 단일 multiclass 모델
        1. 총 18개의 레이블을 유추하는 모델
        2. 각 특성을 잘 유추하는지 확실히 파악하기 어려움
        3. 예측해야 하는 레이블이 많으므로 수렴이 어려울 수 있음
        4. 데이터의 부족으로 특정 레이블의 학습이 어려울 수 있음
     2. 특성에 따른 3개의 multiclass 모델
        1. 3개, 3개, 2개를 유추하는 모델
        2. 각 특성에 따라서 적합 전략은 다양화할 수 있음
        3. 3개의 예측값을 이용해 최종 레이블을 유추하는 객체가 필요
        4. 비교적 문제가 간단해지고 데이터 또한 더 많이 사용하여 학습할 수 있음
        5. 각 모델을 $n$번 학습할 경우 최종 레이블을 산출할 수 있는 모델의 조합은 $n^3$
     3. 7개의 label을 유추하는 multilabel 모델
        1. 모든 문제를 $n$개의 이진 분류 문제로 치환
        2. 각 레이블 간의 의존성을 학습시킬 수 있음

- 실제 학습 모델
  1. 특성에 따른 3개의 multiclass 모델 1
     - 데이터를 7:3으로 나눠 train/validation
     - augmentation
       - random rotation
       - random perspective
       - random gray scale
       - random color jitter
       - random affine
     - backbone으로 inception v3를 사용해 학습
     - lr = 1e-2, optimizer = SGD, lr scheduler는 미사용, cross entropy loss 사용
     - 특성2와 특성3을 위한 모델은 5 epoch 안으로 충분히 수렴
     - 특성1을 위한 모델에 따라 최종 스코어가 결정
       - LB상 accuracy: 74.48%, f1 score: 0.68
     - 각 모델 5 epoch 학습 후 특성1을 위한 모델에 data oversampling 적용 후 재학습
     - augmentation에 normalize 추가
       - LB상 accuracy: 77.35%, f1 score: 0.71
     - 특성1의 레이블을 3개가 아닌 10개로 세분화, data oversampling 적용 후 재학습
     - k fold를 활용하여 cross validation
       - LB상 accuracy: 77.98%, f1 score: 0.73
  2. 특성에 따른 3개의 multiclass 모델 2
     - 데이터를 8:2으로 나눠 train/validation, oversampling
     - augmentation
       - center crop 384\*384
       - horizontal flip
       - shift scale rotate
       - hue, saturation value
       - random brightness contrast
       - normalize
       - random grid shuffle
       - coarse dropout
       - cutout
     - backbone으로 efficientnet b3를 사용해 학습
     - lr = 1e-5, optimizer = Adam, lr scheduler = cosine annealing warmup restart, cross entropy loss 사용
     - 특성2와 특성3을 위한 모델은 2 epoch 안으로 충분히 수렴
     - 특성1을 위한 모델에 따라 최종 스코어가 결정
     - 특성1의 기존 3개의 레이블을 10개로 세분화하여 학습
       - LB상 accuracy: 77.15%, f1 score: 0.7226
  3. 7개의 label을 유추하는 multilabel 모델
     - 데이터를 8:2으로 나눠 train/validation, oversampling
     - augmentation
       - center crop 384\*384
       - horizontal flip
       - shift scale rotate
       - hue, saturation value
       - random brightness contrast
       - normalize
       - random grid shuffle
       - coarse dropout
       - cutout
     - backbone으로 efficientnet b3를 사용해 학습
     - lr = 1e-4, optimizer = AdamW, lr scheduler = cosine annealing warmup restart, multilabel margin soft loss 사용
     - 특성2와 특성3을 예측하는 f1 score는 5 epoch 안으로 0.98 달성
     - 특성1의 label 0 또한 f1 score가 10 epoch 안으로 0.98 달성
     - 특성1의 label 1, label 2의 f1 score는 60 epoch의 학습 동안 평균 0.8을 달성하지 못함
     - 어떤 제출도 하지 않음
  4. 특성에 따른 3개의 multiclass 모델 3
     - 데이터를 8:2으로 나눠 train/validation, oversampling
     - augmentation
       - center crop 384\*384
       - horizontal flip
       - shift scale rotate
       - hue, saturation value
       - random brightness contrast
       - normalize
       - random grid shuffle
       - coarse dropout
       - cutout
     - backbone으로 efficientnet b3를 사용해 학습
     - 특성3의 예측 값이 1일 때를 위한 특성1 분류기와 특성3의 예측 값이 0일 때를 위한 특성1 분류기
     - lr = 1e-5, optimizer = Adam, lr scheduler = cosine annealing warmup restart, cross entropy loss 사용
     - 특성2와 특성3을 위한 모델은 2 epoch 안으로 충분히 수렴
     - 특성3의 예측값이 0일 때 특성1 분류기의 성능은 평균 f1 score가 0.8에 근사함
     - 특성3의 예측값이 1일 때 특성1 분류기의 성능은 평균 f1 score가 0.7을 넘지 못함
     - LB상 accuracy: 76.2698%, f1 score: 0.7117

- 그 외 기법
  - 앙상블
    - 단순 voting 적용

- 학습과정에서 교훈
  - 피어세션을 여러 사람과 하는 것은 매우 좋았음, 매일 같은 조가 된 사람들의 성과를 보고 비슷하거나 높은 점수를 갖고 있는 사람에게 질문할 것들을 준비해 갔었고 이런 질문들 덕분에 늘 배울 수 있는 시간이었음
  - 여러 사람의 이야기를 듣다보면 여러 기법이 짬뽕이 되는데 오히려 이게 독이 될 수 있음, 어떤 기법이 서로 호응하거나 또는 서로를 저해할 수 있는지 여러 실험이 필요함
  - 당장의 문제를 해결하는 것이 아닌 해결을 위한 시도가 편한 코딩 스타일이 필요함, 시간이 많음에도 불구하고 시간에 좇기듯 코딩을 했고 때문에 후반부 거의 제출을 하지 못할만큼 리팩토링에 시간을 많이 씀

- 차후 실험해볼 과제
  - focal loss
  - f1 loss
  - weight averaging
  - various drop out
  - shallow feature extractor, sparse feature extractor 함께 사용
  - transformer
  - convolutional LSTM
  - cutmix
  - mixup
  - batch size = 1


- 스스로에 대한 비판
  - 왜 inception v3, efficientnet b3를 사용했는가?
    - 당시 이유:
      - 단순히 image task이기에 benchmark가 높은 모델 중 비교적 사이즈가 작은 모델 선택
    - 비판:
      - face deep learning만 검색해도 face image로 미리 학습된 모델을 찾을 수 있는데 실제 이런 사전 검색 없이 선택한 건 상황에 대한 충분한 고려가 없어 보임
  - 왜 f1 loss나 focal loss를 사용 안했는가?
    - 당시 이유:
      - 이미 over sampling을 적용했기 때문에 쓸 필요가 없다고 판단함
    - 비판:
      - 그러면 학습시간이 상당히 늘었을텐데 꾸준한 피보팅과 수정을 위해서라면 오히려 다양한 loss를 사용하는 것이 나음
  - 생각보다 많은 시도를 한 것 같은데 왜 결과가 안 나왔다고 생각하는가?
    - 초기 모든 코드를 주피터노트북 상에 작성함
    - 코드가 길어질수록 작업 시간이 길어짐
    - debugging에 시간을 상당히 많이 씀
    - 문제점을 깨닫고 모듈화를 진행했는데 이로 인해 초기에 작성한 코드와 현재 코드가 잘 호환되지 않음
    - 초기 코드와 호환이 되도록 코드를 수정하기에는 대회 마감이 며칠 안 남아서 초기 코드를 모두 폐기함
  - 왜 transformer나 ConvLSTM을 안 사용했는가?
    - 당시 이유:
      - 사용하기 위해 논문과 여러 article을 살펴보았으나 스스로가 이 개념을 이해하고 있지 않음을 깨달음 
    - 비판:
      - 이해하고 있지 않더라도 여러 github이나 example을 보면 충분히 사용 가능했을텐데
  - random grid shuffle을 쓰면 모델이 얼굴 위치에 따른 그 relation을 충분히 학습하지 못했을 것 같은데?
    - 당시 이유:
      - 얼굴이 어디에 있더라도 모든 피처를 찾아낼 수 있도록 학습하는 것이기에 그 relation을 학습할 필요는 없음
    - 비판:
      - 오히려 많은 그리드로 자른 이미지와 듬성하게 자른 이미지를 각각 학습시켜 concat하여 결과를 추론했다면 성능이 더 좋아질 수 있었을 것임
  - 왜 drop out p는 0.5인가?
    - 당시 이유:
      - 그냥 별 생각 없었음
    - 비판:
      - 데이터가 적을수록 과적합이 쉽게 발생하고 train은 되지만 validation 지표들이 하락하기 쉬워짐, 그러므로 오히려 일반화 성능을 위해 drop out p를 0.7이나 0.8처럼 높게 적용하여 전반적인 성능향상을 도모하는 것을 추천
  - weight averaging은 알고 있었는데 왜 안 썼는가?
    - 당시 이유:
      - 마감 당일에 알게 되어 포기함
    - 비판:
      - 사실 load_state_dict하고 반복문 사용해서 평균만 내면 되는건데 시간이 모자라서 안 했다는 건 핑계임
  - 왜 SGD로 초기 학습을 진행하다 Adam으로 바꿨는가?
    - 당시 이유:
      - SGD는 초기 시간이 많다고 판단하여 일반화성능을 보장하면서 학습하기 위해 사용했으나 시간이 촉박해질수록 빠른 수렴을 위해 Adam으로 바꿨음
    - 비판:
      - 여러 학습률 스케쥴러를 사용하거나 penalty를 사용하면 Adam으로도 충분히 과적합 없이 학습이 가능한데 본인이 몰라서 그냥 SGD 쓴 걸 합리화중임

