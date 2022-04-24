def sum(a, b):
    c = a+b+10
    return c

def lamF():
    print("나는 함수")

class lamC():
    def lamMethod():
        print("lam 클래스 내부 lam 메소드")

x = 1
su = sum(5, x)
print(su)

lamF()

lamC.lamMethod()