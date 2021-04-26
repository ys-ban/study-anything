import pandas as pd
import numpy as np
import os, sys, gc, random
from datetime import date
import datetime
import dateutil.relativedelta


# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Custom library
from utils import seed_everything, print_score


TOTAL_THRES = 300 # 구매액 임계값
SEED = 42 # 랜덤 시드
seed_everything(SEED) # 시드 고정

data_dir = '../input/train.csv' # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model' # os.environ['SM_MODEL_DIR']


'''
    입력인자로 받는 year_month에 대해 고객 ID별로 총 구매액이
    구매액 임계값을 넘는지 여부의 binary label을 생성하는 함수
'''

TOTAL_THRES = 300

'''
    입력인자로 받는 year_month에 대해 고객 ID별로 총 구매액이
    구매액 임계값을 넘는지 여부의 binary label을 생성하는 함수
'''
def generate_label(df, year_month, total_thres=TOTAL_THRES, print_log=False):
    df = df.copy()
    
    # year_month에 해당하는 label 데이터 생성
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    df.reset_index(drop=True, inplace=True)

    # year_month 이전 월의 고객 ID 추출
    cust = df[df['year_month']<year_month]['customer_id'].unique()
    # year_month에 해당하는 데이터 선택
    df = df[df['year_month']==year_month]
    
    # label 데이터프레임 생성
    label = pd.DataFrame({'customer_id':cust})
    label['year_month'] = year_month
    
    # year_month에 해당하는 고객 ID의 구매액의 합 계산
    grped = df.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    
    # label 데이터프레임과 merge하고 구매액 임계값을 넘었는지 여부로 label 생성
    label = label.merge(grped, on=['customer_id','year_month'], how='left')
    label['total'].fillna(0.0, inplace=True)
    label['label'] = (label['total'] > total_thres).astype(int)

    # 고객 ID로 정렬
    label = label.sort_values('customer_id').reset_index(drop=True)
    if print_log: print(f'{year_month} - final label shape: {label.shape}')
    
    return label



def feature_preprocessing(train, test, features, do_imputing=True, impute_type='constant', fill_value=-2):
    x_tr = train.copy()
    x_te = test.copy()
    
    # 범주형 피처 이름을 저장할 변수
    cate_cols = []

    # 레이블 인코딩
    for f in features:
        if x_tr[f].dtype.name == 'object': # 데이터 타입이 object(str)이면 레이블 인코딩
            cate_cols.append(f)
            le = LabelEncoder()
            # train + test 데이터를 합쳐서 레이블 인코딩 함수에 fit
            le.fit(list(x_tr[f].values) + list(x_te[f].values))
            
            # train 데이터 레이블 인코딩 변환 수행
            x_tr[f] = le.transform(list(x_tr[f].values))
            
            # test 데이터 레이블 인코딩 변환 수행
            x_te[f] = le.transform(list(x_te[f].values))

    print('categorical feature:', cate_cols)

    if do_imputing:
        # 중위값으로 결측치 채우기, impute_type에 따라 다름
        imputer = SimpleImputer(strategy=impute_type, fill_value=(None if impute_type!='constant' else fill_value))

        x_tr[features] = imputer.fit_transform(x_tr[features])
        x_te[features] = imputer.transform(x_te[features])
    
    return x_tr, x_te





def devide(x):
    answer = ''
    if x<0: answer = 'minus'
    elif x<20: answer = 'very low'
    elif x<40: answer = 'low'
    elif x<70: answer = 'middle-low'
    elif x<100: answer = 'middle'
    elif x<150: answer = 'middle-high'
    elif x<200: answer = 'almost'
    elif x<250: answer = 'high'
    elif x<270: answer = 'higgh'
    elif x<300: answer = 'hiigh'
    elif x<325: answer = 'veeery high'
    elif x<350: answer = 'very high'
    elif x<500: answer = 'very hiigh'
    elif x<1000: answer = 'super high'
    else: answer = 'epic high'
    return answer



def count_over(df,year_month):
    cid = 'customer_id'
    cust = df[df['year_month'] < year_month][cid].unique()
    count = pd.DataFrame({'customer_id':cust}).sort_values(cid)
    
    count['count'] = 0
    count = count.set_index(cid)
    while True:
        year_month = date(int(year_month.split('-')[0]),int(year_month.split('-')[1]),1) - dateutil.relativedelta.relativedelta(months=1)
        year_month = year_month.strftime('%Y-%m')
        total = df[df['year_month'] == year_month].groupby('customer_id').sum().reset_index()
        if total.empty: break
        cust = total[total['total']>300][cid].unique()
        for x in cust:
            count.at[x,'count']+=1
    return count.sort_values('customer_id').reset_index()['count']



def mean_during_usage(df, year_month):
    df = df.copy()
    cid = 'customer_id'
    cust = df[df['year_month'] < year_month][cid].unique()
    count = pd.DataFrame({'customer_id':cust}).sort_values(cid)
    
    count['count'] = 0
    count = count.set_index(cid)
    while True:
        year_month = date(int(year_month.split('-')[0]),int(year_month.split('-')[1]),1) - dateutil.relativedelta.relativedelta(months=1)
        year_month = year_month.strftime('%Y-%m')
        total = df[df['year_month'] == year_month].groupby('customer_id').sum().reset_index()
        if total.empty: break
        cust = total[total['total']>300][cid].unique()
        for x in cust:
            count.at[x,'count']+=1
    return count.sort_values('customer_id').reset_index()['count']
    

def range_func(x):
    max_val = np.max(x)
    min_val = np.min(x)
    range_val = max_val - min_val
    return range_val


def iqr_func(x):
    q3, q1 = np.percentile(x, [75, 25])
    iqr = q3 - q1
    return iqr


def do_anything(df, features):
    df = df.copy()
    for f in features:
        if not pd.api.types.is_numeric_dtype(df[f]):
            continue
        if f=='mean_cat':
            continue
        new_colname = f+"_squared_n_log"
        df[new_colname] = [x*2 for x in df[f]]
        df[new_colname] = [np.log(x+0.5) for x in df[new_colname]]
    return df

def make_feature(df, ref_date, period, feature, col_prefix):
    df = df.copy()
    ref_date = datetime.datetime.strptime(ref_date, "%Y-%m")
    if period[0]=='d':
        # period[1] > period[2]
        date_s = ref_date - dateutil.relativedelta.relativedelta(days=period[1])
        date_e = ref_date - dateutil.relativedelta.relativedelta(days=period[2])
    if period[0]=='m':
        # period[1] > period[2]
        date_s = ref_date - dateutil.relativedelta.relativedelta(months=period[1])
        date_e = ref_date - dateutil.relativedelta.relativedelta(months=period[2])
    date_s = date_s.strftime("%Y-%m-%d")
    date_e = date_e.strftime("%Y-%m-%d")
    
    df = df[(date_s<df['order_date'])&(date_e>df['order_date'])]

    # customer_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_cust_id'] = df.groupby(['customer_id'])['total'].cumsum()
    df['cumsum_quantity_by_cust_id'] = df.groupby(['customer_id'])['quantity'].cumsum()
    df['cumsum_price_by_cust_id'] = df.groupby(['customer_id'])['price'].cumsum()

    # product_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_prod_id'] = df.groupby(['product_id'])['total'].cumsum()
    df['cumsum_quantity_by_prod_id'] = df.groupby(['product_id'])['quantity'].cumsum()
    df['cumsum_price_by_prod_id'] = df.groupby(['product_id'])['price'].cumsum()
    
    # order_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_order_id'] = df.groupby(['order_id'])['total'].cumsum()
    df['cumsum_quantity_by_order_id'] = df.groupby(['order_id'])['quantity'].cumsum()
    df['cumsum_price_by_order_id'] = df.groupby(['order_id'])['price'].cumsum()

    # diff를 활용한 새로운 feature
    df['order_ts'] = df['order_date'].astype(np.int64)//1e9
    df['order_ts_diff'] = df.groupby(['customer_id'])['order_ts'].diff()
    df['quantity_diff'] = df.groupby(['customer_id'])['quantity'].diff()
    df['price_diff'] = df.groupby(['customer_id'])['price'].diff()
    df['total_diff'] = df.groupby(['customer_id'])['total'].diff()
    
    ret_data = pd.DataFrame()
    
    # group by aggretation 함수로 데이터 피처 생성
    df_agg = df.groupby(['customer_id']).agg(feature)

    # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
    new_cols = []
    cnt = 0
    for col in feature.keys():
        for stat in feature[col]:
            if type(stat) is str:
                new_cols.append(f'{col_prefix}-{col}-{stat}')
            else:
                new_cols.append(f'{col_prefix}-{col}-custom_{cnt}')
                cnt += 1
            

    df_agg.columns = new_cols
    df_agg.reset_index(inplace = True)

    df_agg['year_month'] = ref_date
    df_agg['year_month'] = df_agg['year_month'].dt.strftime('%Y-%m')

    ret_data = ret_data.append(df_agg)
    
    features = ret_data.drop(columns=['customer_id', 'year_month']).columns
    
    print('ret_data.shape', ret_data.shape)
    
    return ret_data, features


def make_fixed_period_feature(df, ref_date, period, feature, col_prefix):
    df = df.copy()
    date_s = period[0]
    date_e = period[1]
    
    df = df[(date_s<df['order_date'])&(date_e>df['order_date'])]

    # customer_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_cust_id'] = df.groupby(['customer_id'])['total'].cumsum()
    df['cumsum_quantity_by_cust_id'] = df.groupby(['customer_id'])['quantity'].cumsum()
    df['cumsum_price_by_cust_id'] = df.groupby(['customer_id'])['price'].cumsum()

    # product_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_prod_id'] = df.groupby(['product_id'])['total'].cumsum()
    df['cumsum_quantity_by_prod_id'] = df.groupby(['product_id'])['quantity'].cumsum()
    df['cumsum_price_by_prod_id'] = df.groupby(['product_id'])['price'].cumsum()
    
    # order_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_order_id'] = df.groupby(['order_id'])['total'].cumsum()
    df['cumsum_quantity_by_order_id'] = df.groupby(['order_id'])['quantity'].cumsum()
    df['cumsum_price_by_order_id'] = df.groupby(['order_id'])['price'].cumsum()

    # new baseline
    df['order_ts'] = df['order_date'].astype(np.int64)//1e9
    df['order_ts_diff'] = df.groupby(['customer_id'])['order_ts'].diff()
    df['quantity_diff'] = df.groupby(['customer_id'])['quantity'].diff()
    df['price_diff'] = df.groupby(['customer_id'])['price'].diff()
    df['total_diff'] = df.groupby(['customer_id'])['total'].diff()
    
    ret_data = pd.DataFrame()
    
    # group by aggretation 함수로 데이터 피처 생성
    df_agg = df.groupby(['customer_id']).agg(feature)

    # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
    new_cols = []
    cnt = 0
    for col in feature.keys():
        for stat in feature[col]:
            if type(stat) is str:
                new_cols.append(f'{col_prefix}-{col}-{stat}')
            else:
                new_cols.append(f'{col_prefix}-{col}-custom_{cnt}')
                cnt += 1
            

    df_agg.columns = new_cols
    df_agg.reset_index(inplace = True)

    df_agg['year_month'] = datetime.datetime.strptime(ref_date, "%Y-%m")
    df_agg['year_month'] = df_agg['year_month'].dt.strftime('%Y-%m')

    ret_data = ret_data.append(df_agg)
    
    # ret_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    features = ret_data.drop(columns=['customer_id', 'year_month']).columns
    
    # ret_data 데이터 전처리 -> 수정 후 나중에 해도 될 듯?
    # x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    
    print('ret_data.shape', ret_data.shape)
    
    return ret_data, features



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


def feature_engineering(df, test_ref_date, train_ref_date, period_rule, feature_rule, total_thres):
    df = df.copy()
    
    # 1. customer_id 추출
    train_label = generate_label(df, train_ref_date, total_thres)[['customer_id','year_month','label']]
    test_label = generate_label(df, test_ref_date, total_thres)[['customer_id','year_month','label']]
    
    # 2. period_rule과 feature_rule에 따라서 위 customer_id에 대응하는 값을 train_ref_date에서 추출
    # 3. 2에서 추출한 dataframe을 병합하여 train_data
    train_data = train_label.copy()
    # train_df_list = []
    train_feat_list = None
    if len(period_rule)!=len(feature_rule):
        raise("Error:\n\tlen(period_rule)!=len(feature_rule)")
    for i in range(len(period_rule)):
        res_tuple = make_feature(df, train_ref_date, period_rule[i], feature_rule[i], i)
        train_data = train_data.merge(res_tuple[0], on=['customer_id', 'year_month'], how='left')
        if train_feat_list is None:
            train_feat_list = res_tuple[1].copy()
        else:
            train_feat_list = train_feat_list.append(res_tuple[1].copy())
    #res_tuple = make_fixed_period_feature(df, train_ref_date, ("2010-11", "2011-11"), agg_dict, len(period_rule))
    #train_data = train_data.merge(res_tuple[0], on=['customer_id', 'year_month'], how='left')
    #train_feat_list = train_feat_list.append(res_tuple[1].copy())
    
    
    df['year_month'] = [str(x)[:7] for x in df['order_date'].copy()[:]]
    train = df[(df['year_month']<'2011-11')&(df['year_month']>'2010-12')][:]
    train_month_count = [len(set(train[train['customer_id']==cust]['year_month'])) for cust in train_data['customer_id'].copy()]
    train_month_sum = [train[train['customer_id']==cust]['total'].sum() for cust in train_data['customer_id'].copy()]
    train_month_mean = [y/x if x>0 else 0 for x, y in zip(train_month_count, train_month_sum)]
    train_mean_cat = [devide(x) for x in train_month_mean]
    train_data['count_m'] = train_month_count
    train_data['sum_m'] = train_month_sum
    train_data['mean_m'] = train_month_mean
    train_data['mean_cat'] = train_mean_cat
    train_feat_list = train_feat_list.append(pd.Index(['count_m', 'sum_m', 'mean_m', 'mean_cat']))

    
    # 4. period_rule과 feature_rule에 따라서 위 customer_id에 대응하는 값을 test_ref_date에서 추출
    # 5. 4에서 추출한 dataframe을 병합하여 test_data
    test_data = test_label.copy()
    # test_df_list = []
    test_feat_list = None
    for i in range(len(period_rule)):
        res_tuple = make_feature(df, test_ref_date, period_rule[i], feature_rule[i], i)
        test_data = test_data.merge(res_tuple[0], on=['customer_id', 'year_month'], how='left')
        if test_feat_list is None:
            test_feat_list = res_tuple[1].copy()
        else:
            test_feat_list = test_feat_list.append(res_tuple[1].copy())
    #res_tuple = make_fixed_period_feature(df, test_ref_date, ("2010-11", "2011-11"), agg_dict, len(period_rule))
    #test_data = test_data.merge(res_tuple[0], on=['customer_id', 'year_month'], how='left')
    #test_feat_list = test_feat_list.append(res_tuple[1].copy())
    
    test = df[(df['year_month']<'2011-12')&(df['year_month']>'2011-01')][:]
    test_month_count = [len(set(test[test['customer_id']==cust]['year_month'])) for cust in test_data['customer_id'].copy()]
    test_month_sum = [test[test['customer_id']==cust]['total'].sum() for cust in test_data['customer_id'].copy()]
    test_month_mean = [y/x if x>0 else 0 for x, y in zip(test_month_count, test_month_sum)]
    test_mean_cat = [devide(x) for x in test_month_mean]
    test_data['count_m'] = test_month_count
    test_data['sum_m'] = test_month_sum
    test_data['mean_m'] = test_month_mean
    test_data['mean_cat'] = test_mean_cat
    test_feat_list = test_feat_list.append(pd.Index(['count_m', 'sum_m', 'mean_m', 'mean_cat']))
    
    train_feat_list = train_data.drop(columns=['customer_id', 'label', 'year_month']).columns # 한번 해봄
    test_feat_list = test_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    # 6. train_data, train_label, test_data, test_label, features 최종 반환
    if (train_feat_list!=test_feat_list).sum()!=0: # len(train_feat_list)!=len(test_feat_list)
        raise("Error:\n\t(train_feat_list!=test_feat_list).sum()!=0")
    x_tr, x_te = feature_preprocessing(train_data, test_data, train_feat_list, do_imputing=True)
    
    # x_tr = do_anything(x_tr, train_feat_list)
    # x_te = do_anything(x_te, train_feat_list)
    # train_feat_list = x_tr.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, train_label['label'], train_feat_list



if __name__ == '__main__':
    
    print('data_dir', data_dir)
