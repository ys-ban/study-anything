# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta
import argparse

import json

import time

# check out intermediate variables
import pickle

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

# Custom library
from utils import seed_everything, print_score
from features import *



# data_dir = '../input' # os.environ['SM_CHANNEL_TRAIN']
# model_dir = '../model' # os.environ['SM_MODEL_DIR']
# output_dir = '../output' # os.environ['SM_OUTPUT_DATA_DIR']


def make_lgb_oof_prediction(train, y, test, features, categorical_features='auto', model_params=None, folds=10):
    x_train = train[features]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # LightGBM 데이터셋 선언
        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dvalid = lgb.Dataset(x_val, label=y_val)
        
        # LightGBM 모델 훈련
        clf = lgb.train(
            model_params,
            dtrain,
            valid_sets=[dtrain, dvalid], # Validation 성능을 측정할 수 있도록 설정
            categorical_feature=categorical_features,
            verbose_eval=200
        )

        # Validation 데이터 예측
        val_preds = clf.predict(x_val)
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 측정
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(x_test) / folds
        
        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold+1}'] = clf.feature_importance()

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
        
    # 폴드별 피처 중요도 평균값 계산해서 저장 
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi


if __name__ == '__main__':

    # 인자 파서 선언
    parser = argparse.ArgumentParser()
    
    # baseline 모델 이름 인자로 받아서 model 변수에 저장
    parser.add_argument('param_json', type=str, default='no_input', help="set path of parameter json file")
    args = parser.parse_args()
    param_json = args.param_json
    
    # param_json:
    #    data_dir
    #    output_dir
    #    model_params_dir
    #    feature_rule

    with open(param_json, 'r') as f:
        param_json = json.load(f)
    
    TOTAL_THRES = 300 # 구매액 임계값
    if param_json=='no_input':
        data_dir = "/opt/ml/code/input"
        output_dir = "/opt/ml/code"
        model_params = {
            'objective': 'binary', # 이진 분류
            'boosting_type': 'gbdt',
            'metric': 'auc', # 평가 지표 설정
            'feature_fraction': 0.7, # 피처 샘플링 비율
            'bagging_fraction': 0.6, # 데이터 샘플링 비율
            'bagging_freq': 1,
            'n_estimators': 10000, # 트리 개수
            'early_stopping_rounds': 300,
            'seed': 42,
            'verbose': -1,
            'n_jobs': -1,    
        }
        period_rule = [
            ["d", 7, 0], ["d", 14, 7], ["d", 30, 14], ["m", 2, 1],
            ["m", 3, 2], ["m", 6, 3], ["m", 6, 0], ["m", 9, 6],
            ["m", 12, 9], ["m", 15, 12], ['m', 22, 0]
        ]
    else:
        data_dir = param_json['data_dir']
        output_dir = param_json['output_dir']
        model_params_dir = param_json['model_params_dir']
        period_rule = param_json['period_rule']

        with open(model_params_dir, "r") as f:
            model_params = json.load(f)  

        SEED = model_params['seed'] # 랜덤 시드
    
    seed_everything(SEED) # 시드 고정
    
    print('baseline model:\n\tLightGBM with k-fold cross validation')
    print(f'seed:\n\t{SEED}')
    print('#'*80)
    print(f'data_dir:\n\t{data_dir}')
    print(f'output_dir:\n\t{output_dir}')
    print(f'model_params_dir:\n\t{model_params_dir}')
    print('period_rule:')
    for i, period in enumerate(period_rule):
        time_unit = "day" if period[0]=='d' else 'month'
        print(f'\tperiod {i}:\t[{time_unit}] {period[1]} ~ {period[2]}')
    print('#'*80)
    
    
    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])

    # 예측할 연월 설정
    year_month = '2011-12'
    agg_basic = ['mean','max','min','sum','count','std','skew']
    agg_custom = ['mean','max','min','sum','count','std','skew', 'median', range_func, iqr_func]
    agg_custom_2 = ['mean','max','min','sum','count','std','skew', 'median', range_func]
    agg_dict = {
        'quantity': agg_custom,
        'price': agg_custom,
        'total': agg_custom,
        'cumsum_total_by_cust_id': agg_custom,
        'cumsum_quantity_by_cust_id': agg_custom,
        'cumsum_price_by_cust_id': agg_custom,
        'cumsum_total_by_prod_id': agg_custom,
        'cumsum_quantity_by_prod_id': agg_custom,
        'cumsum_price_by_prod_id': agg_custom,
        'cumsum_total_by_order_id': agg_custom,
        'cumsum_quantity_by_order_id': agg_custom,
        'cumsum_price_by_order_id': agg_custom,
        'order_id': ['nunique'],
        'product_id': ['nunique'],
        "order_ts":["first", "last"],
        "order_ts_diff":agg_custom_2,
        "quantity_diff":agg_custom_2,
        "price_diff":agg_custom_2,
        "total_diff":agg_custom_2
    }

    feature_rule = [
        agg_dict for _ in range(len(period_rule))
    ]
    

    
    
    # 피처 엔지니어링 실행
    train, test, y, features = feature_engineering(data, year_month, "2011-10", period_rule, feature_rule, TOTAL_THRES)

    # Cross Validation Out Of Fold로 LightGBM 모델 훈련 및 예측
    y_oof, test_preds, fi = make_lgb_oof_prediction(train, y, test, features, model_params=model_params)
    # 테스트 결과 제출 파일 읽기
    sub = pd.read_csv(data_dir + '/sample_submission.csv')
    
    # 테스트 예측 결과 저장
    sub['probability'] = test_preds
    
    
    os.makedirs(output_dir, exist_ok=True)
    # 제출 파일 쓰기
    sub.to_csv(os.path.join(output_dir , 'output_'+"_".join(time.ctime().split())+'.csv'), index=False)