import matplotlib.pyplot as plt
import pandas as pd

#판다스에서 사용했던 sample 데이터 가공
data = pd.read_csv('sample.csv')

#윈도우 제목
plt.figure(num="성적 그래프")

#plt.title의 글꼴 설정
plt.rcParams.update({'font.family':'malgun gothic', 'font.size':12})

#차트 제목
plt.title("학번별 성적")

#x, y축 제목 설정
plt.xlabel("학번")
plt.ylabel("성적")

#1, 3, 4는 x축, 2, 4, 2는 y축 좌표값이며, marker 사용시 각 좌표의 점 표현
#plt.plot([1, 3, 4], [2, 4, 2], marker='o')

#pandas의 DataFrame을 차트로 표현하기
#x값, y값은 각각 x축, y축으로 설정할 열 이름이다. 값을 넣지 않으면
#인덱스가 x축, 나머지가 y축으로 인식된다.
#kind는 차트 종류이며, 라인 차트이면 kind='line' 처럼 작성한다.
#DataFrame.plot(kind='line', x=column, y=columns, color=color, ax=None)

#plt적용, 한개의 화면에서 나타나도록 하기 위한 axes 설정
#아래에서 계속 사용됨
ax = plt.gca()

#1. 
#columns = ['운영체제', '논리회로', 'DB']
#data.plot(kind='line', x='학번', y=columns, ax=ax, marker='o')

#2. DataFrame은 따로 구문 추가로 작성후 테스트 해주어야 함
#라인 차트와 유사한 막대 차트, stacked는 하나의 막대에 누적 여부
#DataFrame.plot(kind='bar', x=column, y=columns, color=color, ax=None, stakced=False)

#3. DataFrame은 따로 구문 추가로 작성후 테스트 해주어야 함
#Scatter 차트: x, y축 좌표 값을 점으로 표현. x, y값은 필수이며 x, y축의 사이즈가 같아야함
#DataFrame.plot(kine='scatter', x=columns, y=columns, color=color, ax=None, s=area)

#4.
#xcolumns = ['학번', '학번', '학번']
#ycolumns = ['운영체제', '논리회로', 'DB']
#data.plot(kind='scatter', x=xcolumns, y=ycolumns, ax=ax)

#5. 다수의 Scatter 차트
#color = ['#209FDF','#99CA53','#F6A625', '#6D5FD5','#BF593E', '#FF0000', '#0000FF']
#data.plot(kind='scatter', x='학번', y='운영체제', ax=ax, color=color[0], s=10)
#data.plot(kind='scatter', x='학번', y='논리회로', ax=ax, color=color[1], s=20)
#data.plot(kind='scatter', x='학번', y='DB', ax=ax, color=color[2], s=30)



#6. 파이 차트로 출력
#data = pd.DataFrame({'mass': [0.330, 4.87, 5.97], 'radius': [2439.7, 6051.8, 6378.1]}, index=['Mercury', 'Venus', 'Earth'])
#plt.rcParams.update({'font.family': 'malgun gothic', 'font.size': 12})
#plt.title("행성 크기 비교")
#data.plot(kind='pie', y='mass', ax=ax)

#7. 히스토그램: 데이터 분석에서 변수의 분포, 중심 경향, 퍼짐 정도, 치우침 정도등을 파악할 때 사용
#도수 분포의 상태를 막대 모양으로 표현. 인수 중 bins는 x축이자 히스토그램으로 나타낼 막대 수  의미
#bins의 기본값은 10, 세분화해서 보려면 기본값보다 크게 설정
#alpha는 투명도 설정, 0.5로 설정시 반투명

#학번 열 삭제
#data = data.drop('학번', axis=1)
# plt.figure(num="히스토그램")
# plt.rcParams.update({'font.family': 'malgun gothic', 'font.size': 12})
# plt.title("과목별 점수 빈도")
# ax=plt.gca()
# data.plot(kind='hist', by=None, bins=20, ax=ax, alpha=0.5, edgecolor='black', linewidth=1)
# # x, y 축 제목
# plt.xlabel("과목별 성적")
# plt.ylabel("빈도수")

#8. 단일 히스토그램

data = data.drop('학번', axis=1)
# 창사코 데이터를 data1에 series로 저장
data1 = data['운영체제']
# 차트 윈도우 제목
plt.figure(num="히스토그램")
# 차트 글꼴
plt.rcParams.update({'font.family': 'malgun gothic','font.size': 12})
# 차트 제목
plt.title('운영체제 점수 빈도')
# plt 적용, 한개의 화면에서 나타나도록 하기 위한 axes 설정
ax = plt.gca()
data1.plot(kind='hist', by=None, bins=20, ax=ax, alpha=0.5, edgecolor='black', linewidth=1)
# x, y 축 제목
plt.xlabel("과목별 성적")
plt.ylabel("빈도수")

plt.show()