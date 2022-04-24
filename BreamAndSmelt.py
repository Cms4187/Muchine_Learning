#4주차 파일

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

import matplotlib.pyplot as plt

# #폰트 설정해야 xlabel, ylabel을 한글로 사용 가능
# plt.rcParams.update({'font.family':'malgun gothic', 'font.size':12})

# #차트 제목
# plt.title("물고기의 길이와 무게")

# #가로, 세로를 각각 설정한 스캐터(산점도) 차트 생성
# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.xlabel('길이')
# plt.ylabel('무게')

# #좌측 하단의 주황색 데이터가 빙어
# plt.show()

t_length = bream_length + smelt_length
t_weight = bream_weight + smelt_weight

#리스트 내포(큰 리스트 안에 작은 리스트를 만들어주는 기법)
#bream과 smelt의 length, weight를 각각 더해주어 새로 만든 변수인
#t_length, t_weight 값을 fish_data 리스트에 zip을 사용해 넣어준다.
fish_data = [ [l, w] for l, w in zip(t_length, t_weight) ]

#도미 값을 5, 빙어 값을 7로 설정
fish_target = [5] * 35 + [7] * 14
print(fish_target)


from sklearn.neighbors import KNeighborsClassifier

#KNeighborsClassifier 객체를 만들어줌
kn = KNeighborsClassifier()

#kn 변수 안에 fish_data, fish_target 데이터를 넣어줌
kn.fit(fish_data, fish_target)

#score: 사이킷 런에서 모델을 평가하는 메소드
#0에서 1.0까지 정확도가 높을 수록 1.0에 가까운 값이 출력된다.
#kn안의 데이터와 fish_data, fish_target의 데이터가 전부 일치하므로 1.0 출력
print(kn.score(fish_data, fish_target))

#result 데이터 값이 도미 값에 가까우므로 fish_target에서 설정한 5출력
result = kn.predict([ [30, 600] ])
print(result)


plt.rcParams.update({'font.family':'malgun gothic', 'font.size':12})

plt.title("물고기의 길이와 무게")
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
#marker로 표시되는 모양 변경 가능
plt.scatter(30, 600, marker='^')
plt.xlabel('길이')
plt.ylabel('무게')

#좌측 하단의 주황색 데이터가 빙어
plt.show()


