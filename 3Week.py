#cmd(관리자 권한 실행)에서 먼저 pip install pandas 해줘야함
#데이터 분석에 가장 많이 쓰이는 라이브러리
#데이터를 리스트나 배열로 변환해 사용
#1차원 배열 Series, 다차원 배열 DatFrame이라는 객체 생성


import pandas as pd

#시리즈의 인덱스 값은 0~??로 정수이며,
#a~i로 강제로 인덱스값을 바꿔줌
datas = pd.Series([1, 2, 3, 4, 5, "A", "B", "가", "나"],
                    ['a','b','c','d','e','f','g','h','i'])

print(datas)
print(type(datas))

#A열에 1~4, B열에 5~8 데이터를 넣음
#
data2 = pd.DataFrame({'A':[1, 2, 3, 4], 'B':[5, 6, 7, 8]})

print(data2)

