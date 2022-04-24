import numpy as np

#데이터 준비
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

    
#훈련 세트 준비
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

#선형 회귀(LinearRegression)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
#선형 회귀로 훈련시킴
lr.fit(train_input, train_target)

print(lr.predict([[50]]))
#coef: 기울기
#intercept: 절편(X or Y축과 맞닿는 부분의 점 좌표) - 여기선 Y절편
#각 [39.01714496], -709.0186449535474 출력
print(lr.coef_, lr.intercept_)

import matplotlib.pyplot as plt

plt.scatter(train_input, train_target)
#x값(길이)를 15~50까지 표시
#y값은 절편과 기울기를 고려해 표시할 길이의 최솟값, 최댓값과 곱해 표시
plt.plot([15, 50], [15*lr.coef_ + lr.intercept_, 50*lr.coef_ + lr.intercept_])
#점 하나 임의로 표시 (길이, 무게)
plt.scatter(50, 1241.8, marker='^')
plt.show()

#두 세트의 정확도가 너무 낮으므로 과소적합
print(lr.score(train_input, train_target)) #93.98%
print(lr.score(test_input, test_target))   #82.47%


print("\n다항 회귀\n")

#다항 회귀(곡선) - 2차 방정식

#원래 train_input(test_input)의 제곱값을 stack에 쌓아줌
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

#데이터가 바뀌었으므로 다시 훈련시킴
lr2 = LinearRegression()
lr2.fit(train_poly, train_target)

#예측값 출력
print(lr2.predict([[50**2, 50]]))
#기울기, 절편값 출력
#출력값은 [1.01~~ -21.55~~], 116.05~~가 출력됨
#무게 = 1.01 x 길이^2 - 21.6 x 길이 + 116.05 로 예측
print(lr2.coef_, lr2.intercept_)

#구간별 직선을 그리기 위해 15에서 49까지 정수 배열 생성
point = np.arange(15, 50)

#훈련 세트의 산점도 그리기
plt.scatter(train_input, train_target)

#15에서 49까지 2차 방정식 그래프 그리기
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
#50cm 농어 데이터를 임의로 넣기
plt.scatter([50], [1574], marker='^')
plt.show()

#아까 선형 회귀때 했던 것보다 정확도가 오르고 오차율이 떨어짐
print(lr2.score(train_poly, train_target)) #97.06%
print(lr2.score(test_poly, test_target))   #97.75%






