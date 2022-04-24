from sklearn.neighbors import KNeighborsClassifier

#도미 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

#빙어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

#도미, 빙어 데이터의 무게, 길이를 각각 합쳐 fish_data 리스트에 넣어줌
t_length = bream_length + smelt_length
t_weight = bream_weight + smelt_weight
fish_data = [ [l, w] for l, w in zip(t_length, t_weight) ]

#도미 값을 5, 빙어 값을 7로 설정
fish_target = [5] * 35 + [7] * 14

train_input = fish_data[:35]
train_target = fish_target[:35]

test_input = fish_data[35:]
test_target = fish_target[35:]

kn = KNeighborsClassifier()
kn = kn.fit(train_input, train_target)
#score 메소드로 데이터 정확도 확인
print(kn.score(test_input, test_target))

import numpy as np

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
#print(input_arr)

#seed: 난수를 생성하기 위한 정수 초깃값 지정
np.random.seed(42)
#arange: 일정한 간격의 정수 또는 실수 배열 생성 (기본 간격은 1),
#매개변수가 하나이면 종료 숫자를 의미. 49면 0~48까지 생성
index = np.arange(49)
print(index)

#shuffle: 주어진 배열을 랜덤하게 섞음.
#          다차원 배열이면 첫 번째 행에 대해서만 섞음
np.random.shuffle(index)
print(index)

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

import matplotlib.pyplot as plt

kn = kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))

#첫 번째 열(0)
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel("length")
plt.ylabel("weight")
#파란색은 훈련 데이터, 주황색은 테스트 데이터
plt.show()




