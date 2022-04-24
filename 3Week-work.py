#1. 가장 오른쪽에 총합, 평균 열을 추가하고 각 학생의 총합, 평균 값을 입력한다.
#2. 총합을 기준으로 순위를 결정한다.
#3. 순위를 기준으로 1등부터 오름차순으로 나열한다.
#4. 평균을 기준으로 90이상은 A, 80이상은 B, 나머지는 C인 등급을 학점 열에 추가한다.

import pandas as pd

data = pd.read_csv('sample.csv')

#성적 부분의 열만 따로 뽑아 변수만들고
#만든 변수값의 각 행마다 총합를 구함
data_score = data[['운영체제', '논리회로', 'DB']]
sums = data_score.sum(axis=1)
#소수점 첫째 자리까지만 표시
avgs = (sums/3).round(1)

#1. 만든 총합, 평균을 원본 데이터의 가장 오른쪽에 열을 추가해 넣음
data['총합'] = sums
data['평균'] = avgs

#2. 총합을 기준으로 순위를 정함
#ascending=False로 높은 총합이 1순위에 가까움
data['순위'] = data['총합'].rank(ascending=False)

#3. 순위를 기준으로 1등부터 오름차순으로 나열함
data_2 = data.sort_values('총합')
#print(data_2)

#4. 평균을 기준으로 90이상은 A, 80이상은 B, 나머지는 C인 등급을 학점 열에 추가한다.

def func(row):
    if row > 90:
        return "A"
    if row > 80:
        return "B"
    else:
        return "C"

#논리회로 점수에 맞춰 함수에 작성한 대로
#성적을 판단하여 성적 열을 만들어 줌
data_2['학점'] = data_2['평균'].apply(func)
print(data_2)


