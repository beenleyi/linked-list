import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

dataset1=pd.read_csv('data/dataset1.csv')
dataset2=pd.read_csv('data/dataset2.csv')
dataset3=pd.read_csv('data/dataset3.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)

dataset12=pd.concat([dataset1,dataset2],axis=0)

dataset12_y=dataset12.label
dataset12_x=dataset12.drop(['uid','mid','cid','dr','dtr','dt','label'],axis=1)
dataset3_preds=dataset3[['uid','cid','dtr']]
dataset3_x=dataset3.drop(['uid','mid','cid','dr','dtr'],axis=1)

dataset12=xgb.DMatrix(dataset12_x,label=dataset12_y)
dataset3=xgb.DMatrix(dataset3_x)

params={'booster':'gbtree',
	    'objective': 'rank:pairwise',#二分类
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':7
	    }

watchlist = [(dataset12,'train')]
model = xgb.train(params,dataset12,num_boost_round=3500,evals=watchlist)

#predict test set
dataset3_preds['label'] = model.predict(dataset3)
dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label.values.reshape(-1, 1))
dataset3_preds.sort_values(by=['cid','label'],inplace=True)
dataset3_preds.to_csv("xgb_preds.csv",index=None,header=None)
print(dataset3_preds.describe())

feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)

fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))
 
with open('xgb_feature_score.csv','w') as f:
    f.writelines("feature,score\n")#添加列名字
    f.writelines(fs)

