#회귀 모델 훈련

from cgi import test
import numpy as np

#농어의 길이, 무게 데이터 가져오기
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

import matplotlib.pyplot as plt

#위의 데이터 X, Y 라벨 설정하고 출력
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#훈련 세트 준비
from sklearn.model_selection import train_test_split

#random state: 호출할 때마다 동일한 학습/테스트용 데이터 세트를 생성하기 위해 주어지는 난수값
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

#reshape는 배열의 모양을 [1,2,3] -> [[1],
#                                               [2],
#                                               [3]] 처럼 바꿔줌
#전자는 크기가 3, 후자는 (3, 1)이 된다.
#reshape안 -1은 size를 기반으로 row개수를 선정해줌
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
#총 갯수가 56개이며, 출력은 (42, 1) (14, 1)로 된다.
print(train_input.shape, test_input.shape)


# test_array = np.array([1, 2, 3, 4])
# print(test_array)
# #출력시 4가 아닌 (4, )로 출력된다.
# #해당 자료가 튜플로 만들어졌다는 것을 알려주는 것임.
# print(test_array.shape)

# test_array = test_array.reshape(2, 2)
# print(test_array)
# print(test_array.shape)

#예측, 평균은 타깃에 대한 예측값, 평균값임
#R^2 = 1 -   (타깃 - 예측)^2 의 합
#               --------------------------
#                (타깃 - 평균)^2 의 합

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
#fit 메소드를 사용해 train_input, train_target 데이터를 훈련시킴
knr.fit(train_input, train_target)
#score로 정확도를 출력함 (1에 가까울수록 높은 정확도)
#과대적합과 과소적합
#과대적합: score확인시 train 점수가 test보다 높은 경우
#과소적합: score확인시 train 점수가 test보다 낮거나 둘 다 낮은 경우
#             과소적합은 데이터 세트 크기가 작을 때 주로 발생
print(knr.score(train_input, train_target)) #96.98%
print(knr.score(test_input, test_target))   #99.28%
from sklearn.metrics import mean_absolute_error
#입력한 test_input을 결과로 내줌
test_prediction = knr.predict(test_input)
#test_input을 결과로 낸 값 test_prediction과 test_target을 서로 비교
mae = mean_absolute_error(test_target, test_prediction)
#비교한 결과 - 19.~~~ 출력됨 <- 오차값이 19g정도 차이난다는 의미
#만약 score가 1이면 0이 출력됨
print(mae)

#KNeighborsClassifier 클래스의 기본값이자,
#knr의 이웃값을 변경해줌, 기본값은 5
#주변 이웃이 많아질수록 길이가 어떻게 되었든 결과가 동일해짐
#이웃 개수가 적을 수록 과대적합이, 많을 수록 과소적합이 나타남 
knr.n_neighbors = 3

#다시 훈련후 score 출력
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target)) #98.04%
print(knr.score(test_input, test_target))   #97.46%