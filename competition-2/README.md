# 개요
- 본 대회는 인터넷쇼핑몰 2년간의 기록(2009.12~2011.11)을 통해 2011년 12월 총 구매액이 300이 넘을 고객을 찾는 것이 목표이다.
- 주문번호, 상품번호, 수량, 개당 가격, 총 금액, 상품설명, 주문일시, 고객번호가 각 컬럼으로 주어졌다.
- 기본적으로 시계열을 기반으로한 prophet이나 auto_arima를 사용해 예측하려고 했지만 undersampling을 해도 데이터의 수가 부족해 DT를 통해 접근했다.

# 폴더 및 파일 설명
- [src](./src/)는 모듈화된 코드가 담겨있다.
  - [evaluation.py](./src/evaluation.py): 예측 결과의 score를 계산하는 모듈
  - [features.py](./src/features.py): DT모델을 위한 피쳐를 추출하는 모듈
  - [inference.py](./src/inference.py): 실험환경을 포함한 json파일의 경로를 입력받아 예측을 수행하는 모듈
  - [utils.py](./src/utils.py): 기타 학습, 테스트를 위한 코드
- [notebook](./notebook/)는 초기 실험용 코드, EDA를 수행한 노트북 파일이 담겨있다.
- [output](./output/)는 실제 inference.py를 사용해 예측을 수행한 결과가 담기는 폴더이다.
- [daily_note.md](./daily_note.md)는 본 과제 수행을 위해 2주간 고민하고 실험한 내용이 담겨있다.
- [HOW_TO_WORK.ipynb](./HOW_TO_WORK.ipynb)는 실제 inference.py를 사용하는 예시가 담겨있다.
- [FINAL_TEST.ipynb](./FINAL_TEST.ipynb)는 모듈화 직전 마지막으로 코드를 테스트한 주피터노트북이다.