import pandas as pd

df = pd.read_csv('https://bit.ly/perch_csv')
#pandas 내부에 numpy를 import 하고 있어서
#numpy를 따로 import하지 않고 사용했음
perch_full = df.to_numpy()
#데이터 출력
#print(perch_full)


#6주차 데이터 가져옴
import numpy as np

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

#다항 특성 만들기
from sklearn.preprocessing import PolynomialFeatures

#PolynomialFeatures: 다항 회귀
#다항 회귀: 데이터들간의 형태가 비선형 일때 데이터에
#              각 특성의 제곱을 추가해주어서 특성이 추가된
#              비선형 데이터를 선형 회귀 모델로 훈련시키는 방법
#degree default = 2, 따로 5로 설정
poly = PolynomialFeatures(degree = 2, include_bias=False)
#fit안 [2, 3] 값이 1(bias), 2, 3, 2**2, 2*3, 3**2 를 만들어냄
#poly.fit([[2, 3]])

#위 poly에 include_bias = False로 설정 시
#출력할 때 맨 앞 1(bias)가 출력되지 않음
#print(poly.transform([[2, 3]]))

poly.fit(train_input)

#훈련 데이터
train_poly = poly.transform(train_input)

#shape로 데이터를 다차원 배열형태로 바꾸어 얼마나 행, 열이 있는지 출력
#print(train_poly.shape) #(42, 9) 출력
#print(train_input.shape) #(42, 3) 출력

#특성이 어떻게 만들어졌는지 출력
#print(poly.get_feature_names())

#검증 데이터
test_poly = poly.transform(test_input)

from sklearn.linear_model import LinearRegression

lr = LinearRegression() #선형 회귀 객체 생성
lr.fit(train_poly, train_target) #훈련
#훈련 데이터 정확도 검증
print(lr.score(train_poly, train_target)) #degree = 2일때 99.03%
print(lr.score(test_poly, test_target)) #degree = 2일때 97.14%

#규제: 조건이 너무 많아지면 예측 결과가 떨어질 수 있어 적절히 제한하는 것

#규제 적용 전에 표준화(범위 scale을 줄여줌)
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

#릿지 회귀 - 과대 적합을 없애기 위해 너무 많은 특성 수를 줄여줌
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print("릿지 회귀")
print(ridge.score(train_scaled, train_target)) #98.57%
print(ridge.score(test_scaled, test_target))   #98.35%

#적절한 규제 강도 찾기
import matplotlib.pyplot as plt

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    #릿지 모델 생성
    #alpha_list.에서 지정한 scale 6개로 Ridge 제한을 걸고 생성
    #alpha = 0.1처럼 한 가지 값만 넣을 수 있음
    ridge = Ridge(alpha=alpha)
    #릿지 모델 훈련
    ridge.fit(train_scaled, train_target)
    #훈련 점수와 테스트 점수 저장
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

#alpha_list 값 6개를 넣은 정확도 출력 (각 6개씩 출력)
#출력시 train_score, test_score 값이 가장 근접한 값은 0.1임
#print(train_score)
#print(test_score)

#alpha_list를 10의 지수값으로 뽑아내기위해 log10 사용
#값이 10^-3, 10^-2 처럼 됨
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

#라쏘 회귀
from sklearn.linear_model import Lasso

#alpha값을 넣어 설정할 수 있음
#lasso = Lasso(alpha = 10)
lasso = Lasso()
lasso.fit(train_scaled, train_target) #훈련
print("라쏘 회귀")
print(lasso.score(train_scaled, train_target)) #98.65%
print(lasso.score(test_scaled, test_target))   #98.46%

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

#lasso의 기울기(coef)가 0이 되는 값의 갯수 sum값 출력
#출력값: 3 || 머신러닝하고 관계없는 단순 값 확인 기법
#print(np.sum(lasso.coef_ == 0))














