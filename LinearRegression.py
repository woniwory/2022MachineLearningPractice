import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

#help(sm.add_constant)

#현재경로 확인
#print(os.getcwd())

#데이터불러오기
corolia = pd.read_csv("C:\\Users\\woniw\\OneDrive\\바탕 화면\\ToyotaCorolla.csv")

#데이터확인
#print(corolia.head())

#데이터 수와 변수의 수 확인하기
nCar = corolia.shape[0]
nVar = corolia.shape[1]

#print(nCar,nVar)

#범주형 변수를 이진형 변수( 0 ,1 )로 변환

#변수안에 범주형 변수가 있는지 확인
#print(corolia.Fuel_Type.unique())

#가변수 생성
dummy_p = np.repeat(0,nCar)
dummy_d = np.repeat(0,nCar)
dummy_c = np.repeat(0,nCar)

#인덱스 슬라이싱 후 1 대입
p_idx = np.array(corolia.Fuel_Type=="Petrol")
d_idx = np.array(corolia.Fuel_Type=="Diesel")
c_idx = np.array(corolia.Fuel_Type=="CNG")

#print(p_idx,d_idx,c_idx)

dummy_p[p_idx] = 1
dummy_d[d_idx] = 1
dummy_c[c_idx] = 1

#불필요한 변수 제거 및 가변수 추가
Fuel = pd.DataFrame({"Petrol":dummy_p, "Diesel": dummy_d, "CNG":dummy_c})

print(Fuel)

corolia_ = corolia.drop(["Id","Model","Fuel_Type"],axis=1,inplace=False)
mlrData = pd.concat((corolia_,Fuel),1)
print(mlrData.head())


#bias 추가

#상수항 추가 ( 상수항 추가는 한번 더 실행시키면 추가가 더 된다 주의!!)
mlrData = sm.add_constant(mlrData,has_constant="add")
#print(mlrData.head())

#설명변수(X) ,타겟변수(Y) 분리 및 학습데이터와 평가데이터 분할

featureColums = list(mlrData.columns.difference(["Price"]))

x = mlrData[featureColums]
y = mlrData.Price

trainX ,testX , trainY , testY = train_test_split(x,y,train_size=0.7,test_size=0.3)
#print(trainX.shape,testX.shape,trainY.shape,testY.shape)

#Train the MLR / 회귀모델적합
fullModel = sm.OLS(trainY,trainX)
fittedFullModel = fullModel.fit()

#R-Squre 가 높고 , 대부분의 변수들이 유의함.
print(fittedFullModel.summary())

#VIF를 통한 다중공선성 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(mlrData.values,i)
                     for i in range(mlrData.shape[1])]
vif["features"]=mlrData.columns

print(vif)

#학습데이터의 잔차 확인

res = fittedFullModel.resid
import matplotlib.pyplot as plt
#Q-Q plot # 정규분포확인
fig = sm.qqplot(res, fit=True, line='45')
plt.show()
# residual pattern 확인

predY = fittedFullModel.predict(trainX)

fig = plt.scatter(predY,res,s=4)
plt.xlim(4000,30000)
plt.xlim(4000,30000)
plt.xlabel('Fitted values')
plt.ylabel('Residual')
plt.show()

#검증 데이터에 대한 예측

predY2 = fittedFullModel.predict(testX)

plt.plot(np.array(testY-predY2),label="predFull")
plt.legend()
plt.show()

#MSE 값 구하기
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_true=testY,y_pred=predY2)
print(MSE)


