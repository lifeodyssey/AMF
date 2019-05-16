import autosklearn.regression
from sklearn.model_selection import train_test_split

import pandas as pd
import sklearn.metrics
#TODO install these packages
rawdata=pd.read_excel('dataformodel.xlsx')#输入要读入文件的路径
Y=rawdata[['cpue']]#单引号内写要预测的变量名
X=rawdata[['lon','lat','sst','chla','doy']]#单引号内输入要使用的变量名
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=7)#test size代表了测试集在整个数据集中的占比
# x_train=pd.read_csv('xtrain.csv')[['lon','lat','sst','chla','doy']]
# y_train=pd.read_csv('ytrain.csv')[['cpue']]
# x_test=pd.read_csv('xtest.csv')[['lon','lat','sst','chla','doy']]
# y_test=pd.read_csv('ytest.csv')[['cpue']]

#TODO add other variables and auto detect variables
automl = autosklearn.regression.AutoSklearnRegressor(
    include_estimators=["random_forest","xgradient_boosting",],#TODO which model
    exclude_estimators=None,
    include_preprocessors=["no_preprocessing", ],
    exclude_preprocessors=None,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 10},
    )
automl.fit(x_train, y_train.values.ravel())
print(automl.show_models())
automl.refit(x_train, y_train.values.ravel())
y_pre = automl.predict(x_test)#TODO add this result to original dataset
print("R2 score", sklearn.metrics.accuracy_score(y_test.values.ravel(), y_pre))
df=pd.DataFrame(y_pre)
#TODO print different modle result(alt,but remain a button)
#import  xlsxwriter
#xls=xlsxwriter.workbook('pre')
#sht1=xls.add_worksheet()
#sht1.write(df)
df.to_csv('pre.csv')