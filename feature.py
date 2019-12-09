import pandas as pd
import numpy as np
import datetime
import time
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

off_train = pd.read_csv('ccf_offline_stage1_train.csv')
off_test = pd.read_csv('ccf_offline_stage1_test_revised.csv')
off_train.columns = ['uid','mid','cid','dr','dist','dtr','dt']
off_train.dist=off_train.dist.replace(np.nan,-1)
off_train.fillna(0,inplace=True)
off_test.columns= ['uid','mid','cid','dr','dist','dtr']
off_test.dist=off_test.dist.replace(np.nan,-1)

off_test.fillna(0,inplace=True)
#为什么不能在读入的地方使用dtype
off_train['uid']=off_train['uid'].astype(np.int32)
off_train['mid']=off_train['mid'].astype(np.int32)
off_train['cid']=off_train['cid'].astype(np.int32)
off_train['dr']=off_train['dr'].astype(str)
off_train['dist']=off_train['dist'].astype(np.int8)
off_train['dtr']=off_train['dtr'].astype(str)
off_train['dt']=off_train['dt'].astype(str)

off_test['uid']=off_test['uid'].astype(np.int32)
off_test['mid']=off_test['mid'].astype(np.int32)
off_test['cid']=off_test['cid'].astype(np.int32)
off_test['dr']=off_test['dr'].astype(str)
off_test['dist']=off_test['dist'].astype(np.int8)
off_train['dtr']=off_train['dtr'].astype(str)


#优惠券特征提取
##优惠率
def getdiscounttype(row):
    if row == '0':
        return 0 #没有优惠券
    elif ':' in row:
        return 1  #满减
    else:
        return 2 #打折
def convert_rate(row):
    if row == '0':
        return 0 #
    elif ':' in row:
        rows = row.split(':')
        return 1.0-float(rows[1])/float(rows[0])
    else:
        return float(row)
def convert_rate_enough(row):
    if row == '0':
        return 0
    elif ':' in row:
        rows = row.split(':')
        return rows[0]
    else:
        return 0
##星期类型
def weekday(row):
    if row == '0':
        return -1
    else:
        return pd.to_datetime(row,format='%Y%m%d').weekday()+1
def weekday_type(row):
    if row<6:
        return 0
    else:
        return 1

def coupon(df):
    df['discount_type']=df['dr'].apply(getdiscounttype)
    df['discount_rate']=df['dr'].apply(convert_rate)
    df['discount_enough']=df['dr'].apply(convert_rate_enough)
    df['weekday']=df['dtr'].apply(weekday)
    df['weekday_type']=df['weekday'].apply(weekday_type)
    weekdaycols=['weeday_'+str(i) for i in range(1,8)]
    tmpdf=pd.get_dummies(df['weekday'].replace(-1,np.nan))
    tmpdf.columns=weekdaycols
    df[weekdaycols]=tmpdf
    df=df.drop(['weekday'],axis=1)
    return df


#提取用户和优惠券的交互特征
def cal_rate(x,y):
    return x/y
def user_coupon(df):
    res=df[['uid']].drop_duplicates(subset='uid',keep='first')
    t0=df[['uid','dtr']]
    t0=t0[(t0.dtr>'0')]
    t0=t0.groupby(['uid'])['dtr'].agg(lambda x:':'.join(x)).reset_index()
    t0['received_number']=t0.dtr.apply(lambda s:len(s.split(':')))
    res=pd.merge(res,t0[['uid','received_number']],on='uid',how='left')
    
    t1=df[['uid','dtr','dt']]
    t1=t1[(t1.dtr>'0')&(t1.dt>'0')]
    t1=t1.groupby(['uid'])['dtr'].agg(lambda x:':'.join(x)).reset_index()
    t1['used_number']=t1.dtr.apply(lambda s:len(s.split(':')))
    res=pd.merge(res,t1[['uid','used_number']],on=['uid'],how='left')
    res['used_rate']=res.apply(lambda x:cal_rate(x['used_number'],x['received_number']),axis=1)
    t1=df[['uid','dtr','dt','dist']]
    t1=t1[(t1.dtr>'0')&(t1.dt>'0')]
    t1=t1.groupby('uid')['dist'].max().reset_index()
    t1.rename(columns={'dist':'max_dist'}, inplace = True)
    res=pd.merge(res,t1[['uid','max_dist']],on=['uid'],how='left')
    res.fillna(0,inplace=True)
    res.to_csv('data/user_coupon.csv',index=None)
    return res



#提取用户和商户的交互特征
def user_merchant(df):
    res=df[['uid','mid']].drop_duplicates(subset=['uid','mid'],keep='first')
    t0=df[['uid','mid','dtr']]
    t0=t0[(t0.dtr>'0')]
    t0=t0.groupby(['uid','mid'])['dtr'].agg(lambda x:':'.join(x)).reset_index()
    t0['received_mct_number']=t0.dtr.apply(lambda s:len(s.split(':')))
    res=pd.merge(res,t0[['uid','mid','received_mct_number']],on=['uid','mid'],how='left')

    t1=df[['uid','mid','dtr','dt']]
    t1=t1[(t1.dtr>'0')&(t1.dt>'0')]
    t1=t1.groupby(['uid','mid'])['dtr'].agg(lambda x:':'.join(x)).reset_index()
    t1['used_mct_number']=t1.dtr.apply(lambda s:len(s.split(':')))
    res=pd.merge(res,t1[['uid','mid','used_mct_number']],on=['uid','mid'],how='left')
    res['used_mct_rate']=res.apply(lambda x:cal_rate(x['used_mct_number'],x['received_mct_number']),axis=1)
    t1=df[['uid','mid','dtr','dt','dist']]
    t1=t1[(t1.dtr>'0')&(t1.dt>'0')]
    t1=t1.groupby(['uid','mid'])['dist'].max().reset_index()
    t1.rename(columns={'dist':'max_mct_dist'}, inplace = True)
    res=pd.merge(res,t1[['uid','mid','max_mct_dist']],on=['uid','mid'],how='left')
    res.fillna(0,inplace=True)
    return res

def merchant(df):
    res=df[['mid']].drop_duplicates(subset='mid',keep='first')
    t0=df[['mid','dtr']]
    t0=t0[(t0.dtr>'0')]
    t0=t0.groupby(['mid'])['dtr'].agg(lambda x:':'.join(x)).reset_index()
    t0['mct_sent_number']=t0.dtr.apply(lambda s:len(s.split(':')))
    res=pd.merge(res,t0[['mid','mct_sent_number']],on='mid',how='left')
    
    t1=df[['mid','dtr','dt']]
    t1=t1[(t1.dtr>'0')&(t1.dt>'0')]
    t1=t1.groupby(['mid'])['dtr'].agg(lambda x:':'.join(x)).reset_index()
    t1['mct_used_number']=t1.dtr.apply(lambda s:len(s.split(':')))
    res=pd.merge(res,t1[['mid','mct_used_number']],on=['mid'],how='left')
    
    res['mct_used_rate']=res.apply(lambda x:cal_rate(x['mct_used_number'],x['mct_sent_number']),axis=1)
    
    t1=df[['mid','dtr','dt','dist']]
    t1=t1[(t1.dtr>'0')&(t1.dt>'0')]
    t1=t1.groupby('mid')['dist'].max().reset_index()
    t1.rename(columns={'dist':'mct_max_dist'}, inplace = True)
    res=pd.merge(res,t1[['mid','mct_max_dist']],on=['mid'],how='left')
    
    t1=df[['mid','dt']]
    t1=t1[(t1.dt>'0')]
    t1=t1.groupby(['mid'])['dt'].agg(lambda x:':'.join(x)).reset_index()
    t1['bought_number']=t1.dt.apply(lambda s:len(s.split(':')))
    res=pd.merge(res,t1[['mid','bought_number']],on='mid',how='left')
    
    t1=df[['uid','mid','dt']]
    t1=t1[(t1.dt>'0')].drop_duplicates(subset=['mid','uid'],keep='first')
    t1=t1.groupby(['mid'])['dt'].agg(lambda x:':'.join(x)).reset_index()
    t1['bought_user_number']=t1.dt.apply(lambda s:len(s.split(':')))
    res=pd.merge(res,t1[['mid','bought_user_number']],on='mid',how='left')
    
    t1=df[['mid','dr']]
    t1=t1[(t1.dr>'0')]
    t1['mean_dr']=t1['dr'].apply(convert_rate)
    t1=t1.groupby('mid')['mean_dr'].mean().reset_index()
    res=pd.merge(res,t1[['mid','mean_dr']],on='mid',how='left')
    
    res.fillna(0,inplace=True)
    return res

#标记样本
def label(row):
    if row['dtr']=='0':
        return -1
    if row['dt']>'0':
        td = pd.to_datetime(row['dt'],format='%Y%m%d')-pd.to_datetime(row['dtr'],format='%Y%m%d')
        if td<=pd.Timedelta(15,'D'):
            return 1
    return 0

def give_label(df):
    df['label']=df.apply(label,axis=1)
    return df

#此处只有一个dataset产生的代码，因为在实际中为了避免内存使用过大，通过更改变量名运行三次，得到三个数据集
dataset1 = off_test
feature1 = off_train[((off_train.dt>='20160315')&(off_train.dt<='20160630'))|((off_train.dt=='null')&(off_train.dtr>='20160315')&(off_train.dtr<='20160630'))]

dataset1= coupon(dataset1)
dataset1=pd.merge(dataset1,user_coupon(feature1[['uid','dtr','dt','dist']]),on='uid',how='left')
dataset1=pd.merge(dataset1,user_merchant(feature1[['uid','mid','dtr','dt','dist']]),on=['uid','mid'],how='left')
dataset1=pd.merge(dataset1,merchant(feature1[['uid','mid','dr','dtr','dt','dist']]),on='mid',how='left')
#dataset1= give_label(dataset1)
dataset1.max_dist = dataset1.max_dist.replace(np.nan,-1)
dataset1.max_mct_dist = dataset1.max_mct_dist.replace(np.nan,-1)
dataset1.mct_max_dist = dataset1.mct_max_dist.replace(np.nan,-1)
dataset1.fillna(0,inplace=True)
dataset1.to_csv('data/dataset3.csv',index=None)



