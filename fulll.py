import autosklearn.regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#TODO install these packages
rawdata=pd.read_excel('rawdata.xlsx')#输入要读入文件的路径
Y=rawdata[['cpue']]#单引号内写要预测的变量名
#由于这里用的是连续数据，默认进行对数变换，离散数据可以去掉这里
Y=np.log10(Y+1)
X=rawdata[['lon','lat','sst','chla','doy']]#单引号内输入要使用的变量名
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=7)#test size代表了测试集在整个数据集中的占比


#TODO add other variables and auto detect variables
automl = autosklearn.regression.AutoSklearnRegressor(
    include_estimators=["random_forest","decision_tree","gradient_boosting","xgradient_boosting"],#这里只放了几个我认为效果比较好的模型，模型种类参见https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/regression
    exclude_estimators=None,
    include_preprocessors=["no_preprocessing", ],
    exclude_preprocessors=None,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 10},#这里选了10 fold cross validation
    )
automl.fit(x_train, y_train.values.ravel())
print(automl.cv_results_)
automl.sprint_statistics()
automl.show_models()
automl.refit(x_train, y_train.values.ravel())
y_pre = automl.predict(X)#这里X是你想要的预测结果对应的自变量，我这里在原来的结果上进行预测
ypre=np.power(10,ypre)-1#变换回来
ypre=pd.DataFrame(ypre)
result=pd.concat([rawdata,ypre])
result.to_excel('result.xlsx')
#ypre.to_excel(writer,'Sheet1',startcol=rawdata.shape[0]+1)