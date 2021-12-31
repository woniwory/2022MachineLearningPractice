# -*- coding: utf-8 -*-
# 한글 포멧팅 오류를 줄이기 위한 1번줄 코드

from konlpy.tag import Kkma
import numpy as np
from numpy import dot
from numpy.linalg import norm

import requests
from bs4 import BeautifulSoup
import re
import time

#konlpy, numpy 모듈 import

print("="* 34)
print("    Cosine Similarity Program") # 메인 타이틀.
print("="* 34)
text_list=[]

def get_text():
    headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"}
    url=str(input("url 입력: "))
    request=requests.get(url, headers=headers)
    request.raise_for_status()
    soup=BeautifulSoup(request.text,"lxml")
    texts=soup.find("div",attrs={"class":re.compile("^content")}).find("div").get_text()
    text_list.append(texts)


    return text_list

'''
https://www.nocutnews.co.kr/news/5451176
https://www.nocutnews.co.kr/news/5451646
https://www.nocutnews.co.kr/news/5457500
'''

class Cosine_Similarity:

    def __init__(self,pos_list,vector_cases):
        self.__pos_list = pos_list
        self.__vector_cases = vector_cases

    def getVec(self):
        for i in range(3):
            vector_cases[i] = np.zeros(len(pos_list))
            for pos in new_pos_cases[i]:
                idx = 0
                while (pos_list[idx] != pos):
                    idx += 1
                vector_cases[i][idx] += 1
            print(f"Doc{i+1}의 Vector 빈도수 : {vector_cases[i]}")
            print()
        print(f"Vector의 차원 수 (공통된 단어의 개수) : {len(pos_list)}")
        return vector_cases


while True:
    print()
    print("    1. Settings")
    print("    2. getVec 실행")
    print("    3. 제작자 및 module 정보")
    print("    4. 문서 출력")
    print("    5. 종료")
    print()
    print("=" * 34)
    print()
    choice = int(input("원하시는 기능을 선택해주세요 : "))


    if choice == 1:

        print()
        print("이 버전의 Cosine Similarity 프로그램은, 기본적으로 3개의 문서 벡터를 구하여 각각의 유사도를 구하게 설정되어 있습니다.")
        print("Setiings에서는 3개의 문서를 직접 설정할 수 있습니다.")

        for x in range(3):
            get_text()

        print("문서 Setting이 완료되었습니다.")
        print(f"( 현재 문서의 개수 : {len(text_list)} )")
        print()
        print("=" * 34)

    elif choice == 2:

        if len(text_list) == 3:

            kkma = Kkma()

            print()
            print('kkma loaded')
            print()

            start = time.time()  # time 모듈을 이용하여 Runtime 측정시작
            pos1 = kkma.pos(text_list[0])
            pos2 = kkma.pos(text_list[1])
            pos3 = kkma.pos(text_list[2])  # pos1,2,3에 문서1,2,3의 형태소 토큰들을 각각 저장

            pos_cases = [pos1, pos2, pos3]  # pos1,2,3을 담을 pos_cases

            new_pos1 = []  # 조사,어미,문장부호를 제외한 토큰들을 new_pos1,2,3에 저장
            new_pos2 = []
            new_pos3 = []
            new_pos_cases = [new_pos1, new_pos2, new_pos3]  # new_pos1,2,3을 new_pos_cases

            pos_list = []  # 공통된 단어토큰들을 저장할 pos_list

            vector1 = []  # vector의 값을 저장할 vector1,2,3
            vector2 = []
            vector3 = []
            vector_cases = [vector1, vector2, vector3]  # vector1,2,3을 저장할 vector_cases

            for i in range(len(pos_cases)):
                for pos in range(len(pos_cases[i])):
                    if pos_cases[i][pos][1][0] != 'J' and pos_cases[i][pos][1][0] != 'E' and pos_cases[i][pos][1][
                        0] != 'S':  # 조사,어미,문장부호가 아니라면, new_pos1,2,3에 각각저장
                        new_pos_cases[i].append(pos_cases[i][pos])
                        if not pos_cases[i][pos] in pos_list:  # pos1,2,3에 pos_list에 일치하는 요소가 없다면 pos_list에 저장
                            pos_list.append(pos_cases[i][pos])

            # print("new_pos1 : ", new_pos1)
            # print("new_pos2 : ", new_pos2)
            # print("new_pos3 : ", new_pos3)

            # print(len(pos1))
            # print(len(pos2))
            # print(len(pos3))

            # print("pos_list : ", pos_list)
            # print("len(pos_list) : ", len(pos_list))

            c = Cosine_Similarity(pos_list, vector_cases)  # 인스턴스 생성
            vector_cases = c.getVec()  # getVec 메소드를 이용해 벡터 생성


            def cos_sim(A, B):
                return (dot(A, B) / (norm(A) * norm(B))) * 100  # 코사인 값을 계산. 벡터의 내적과 norm을, numpy 모듈을 이용하여 정리.


            Similarity1 = cos_sim(vector_cases[0], vector_cases[1])
            Similarity2 = cos_sim(vector_cases[1], vector_cases[2])
            Similarity3 = cos_sim(vector_cases[2], vector_cases[0])
            cs_cases = [Similarity1, Similarity2, Similarity3]

            print()
            print("=" * 68)
            print()
            for i in range(3):
                print("    Doc1과 Doc2의 코사인 유사도는 %0.2f%%입니다." % cs_cases[i])  # 3개의 코사인 유사도 출력
            print()
            print("    (Doc1, Doc2), (Doc2, Doc3)의 차이는 %0.2f%%입니다." % (
                abs(Similarity1 - Similarity2)))  # 코사인 유사도의 차이 비교, 차이가 가장 작은 두 문서가 제일 유사함.
            print("    (Doc1, Doc2), (Doc3, Doc1)의 차이는 %0.2f%%입니다." % (abs(Similarity1 - Similarity3)))
            print("    (Doc2, Doc3), (Doc3, Doc1)의 차이는 %0.2f%%입니다." % (abs(Similarity2 - Similarity3)))
            print()
            print("    Runtime: %0.2f Minutes" % ((time.time() - start) / 60))  # time 모듈을 이용하여  Runtime 측정완료
            print()
            print("=" * 68)


        else:
            print()
            print("프로그램을 실행하기 위해서는, Settings가 필요합니다.")
            print()
            print("=" * 34)






    elif choice == 3: # 제작자 및 module 정보 출력
        print()
        print("제작자 : 심수민, 이승민, 이창수, 정성원")
        print("konlpy          0.5.2")
        print("numpy           1.19.1")
        print("requests        2.25.0")
        print("beautifulsopa4  4.6.0")
        print()
        print("Ver1 : 2020-11-28")
        print("Ver2 : 2020-12-01")
        print("Ver3 : 2020-12-02")
        print("Ver4 : 2020-12-03")
        print("Ver5 : 2020-12-07")
        print()
        print("=" * 34)
        print()

    elif choice == 4: # 문서 출력
        if len(text_list) != 3: # Settings 미완료시
            print()
            print("프로그램을 실행하기 위해서는, Settings가 필요합니다.")
            print()
            print("=" * 34)

        else:
            for i in range(3): # Settings 완료시
                print()
                print(f" 문서{i+1} : {text_list[i]}")
                print()
                print("=" * 300)

    elif choice == 5: # 프로그램 종료
        print("프로그램을 종료합니다.")
        print()
        print("=" * 34)
        break

    else:
        continue
