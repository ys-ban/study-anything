# ArcFace
- https://arxiv.org/abs/1801.07698
- ArcFaceClassifier를 이용한 classification accuracy 향상 도모
- ArcFaceClassifier에 적합한 ArcFaceLoss를 이용한 학습
- 기존 ArcFaceLoss에는 Cross entropy를 이용해 학습하나 이에 더불에 f1 loss와 focal loss를 결합해 학습 가능하도록 구현

# Results
- ArcFaceClassifier는 과적합이 빠르게 일어남
- ArcFaceClassifier + ArcFaceLoss로 학습한 후 backbone을 freeze한 후 classifier를 변경 후 feature extraction할 경우 실질적 성능향상을 이끌어낼 수 있음
- (ArcFace training) : LB score 하락
- (ArcFace training) -> (backbone freeze) -> (feature extraction for the last linear layer) : LB f1 score 0.0001 이상 향상