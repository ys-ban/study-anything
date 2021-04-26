# 개요
- 본 코드는 보안 상의 문제가 있어서 변수명과 그 외 자세한 사항을 공개하지 않았습니다.
- 이미지를 입력으로 받아 라벨을 알아내는 것이 본 모델의 과제였습니다.
- 전이학습을 활용했으며 문제는 multiclass classification 또는 multilabel classification
- 주피터노트북을 통해 학습 및 평가를 진행했기에 모듈화된 내용은 모델, 데이터셋, 비용함수입니다.


# 폴더 및 파일 설명
- [code](./code/)에는 본 과제에 사용된 모델, 데이터셋, 비용함수 코드가 포함되어 있습니다.
  - [models.py](./code/models.py): 전이학습을 활용한 모델 정의
  - [dataset.py](./code/dataset.py): 문제 정의에 따라 다른 label을 반환하는 데이터셋 정의
  - [loss.py](./code/loss.py): 본 과제에 사용할 수 있는 다양한 비용함수(loss) 정의
- [inference.ipynb](./inference.ipynb)는 본 모델을 통해 실제 예측을 수행하는 예시 코드입니다.
- [daily_note.md](./daily_note.md)는 10일간 본 과제를 해결하기 위한 그 날의 고민 및 실험이 기록되어 있습니다.
- [wrap_up.md](./wrap_up.md)는 10일간의 스스로에 대한 소회가 담겨있습니다.