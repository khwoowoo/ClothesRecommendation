import random

import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

# 보색 구하기
def complementary_color(clothes):
    if clothes.rgb[0] == clothes.rgb[1] == clothes.rgb[2]:
        max_rgb = 255
    else: max_rgb = max(clothes.rgb)
    return (max_rgb-clothes.rgb[0], max_rgb-clothes.rgb[1], max_rgb-clothes.rgb[2])

def cos_sim(A, B):
    if A == [0, 0, 0]:
        A = [1,1,1]
    if B == [0, 0, 0]:
        B = [1,1,1]

    A = np.array(A)
    B = np.array(B)
    return dot(A, B)/(norm(A)*norm(B))

#옷의 정보가 들어가는 옷 클래스
class Clothes:
    def __init__(self, sex, type, color, name, season, rgb, url):
        #성별, 타입(상의, 하의), 색상, 이름, 시즌, rgb, url, 선호도 딕션너리
        self.sex = sex
        self.type = type
        self.color = color
        self.name = name
        self.season = season
        self.rgb = rgb
        self.url = url
        self.preferenceDict = {}
        self.selected = 0

    #출력을 위해
    def __str__(self):
        return 'sex: ' + self.sex + ', type: ' + self.type + ', color: ' + self.color + ', name: ' + self.name + ', season: ' + self.season
    #선호도를 기준으로 딕셔너리 정렬, 내림차순
    def sort(self):
        self.preferenceDict = sorted(self.preferenceDict.items(), key=lambda item: item[1], reverse=True)
    #그리드 알고리즘에서 매칭을 위해
    #pickList에 없다면, matchList에 넣고 매칭
    def matching(self, pickList, matchList):
        for i in self.preferenceDict:
            if i[0] not in pickList:
                pickList.append(i[0])
                matchList.append((self.name, i[0]))
                break
    def bestPick(self, groupList):
        maxNum = -1
        maxIndex = 0
        for i in groupList:
            for j in self.preferenceDict:
                if j[0] == i:
                    if maxNum < j[1]:
                        maxIndex = i
                        maxNum = j[1]
        return maxIndex

#옷 추천 클래스
class ClothesRecommend:
    def __init__(self, sex, season):
        #상의 리스트, 하의 리스트, 성별, 시즌
        self.topList = []
        self.bottomList = []
        self.sex = sex
        self.season = season
        #dataset에서 데이터 넣어주기
        self.getData()

    #문자열 rgb 전처리 -> 리스트
    def preprocessorTuple(self, rgb):
        temp = rgb
        temp = temp[1: len(temp) - 1].split(',')
        tempRGB = []
        for i in range(len(temp)):
            if temp[i][0] == ' ':
                temp[i] = temp[i][1: len(temp[i])]
            tempRGB.append(int(temp[i]))
        return tempRGB

    #옷 리스트에서 이름으로, 객체를 찾아서 반환
    def findClothes(self, list, name):
        for i in list:
            if i.name == name:
                return i

    
    #데이터 입력
    def getData(self):
        #csv 파일 가져오고 조건에 만족하는 데이터들만 가져오기
        csv = pd.read_csv('dataset.csv')
        condtion = (csv['sex'] == '공용') | (csv['sex'] == self.sex)
        filter_sex = csv[condtion]
        condtion = (filter_sex['season1'].str.contains(self.season))
        filter_season = filter_sex[condtion]
        condtion = (filter_season['type'] == 'top')
        filter_top = filter_season[condtion]
        condtion = (filter_season['type'] == 'pants')
        filter_pants = filter_season[condtion]


        colorList = ['black', 'white', 'gray', 'blue', 'ivory', 'lightGray', 'green', 'darkGray', 'skyBlue', 'brown', 'khaki', 'yellow',
                     'red', 'orange', 'purple', 'mint', 'pink', 'sand', 'darkGreen', 'oliveGreen', 'lavender', 'burgundy', 'lightGreen',
                     'lightYellow', 'lightPink', 'camel', 'deepRed', 'khakiBeige', 'demin']

        for color in colorList:
            condtion = (filter_top['color'] == color)
            filter_color = filter_top[condtion]
            index = random.randrange(0, len(list(filter_color.iloc)))
            i = filter_color.iloc[index]
            self.topList.append(Clothes(i[0], i[1], i[2], i[3], i[4], self.preprocessorTuple(i[6]), i[7]))
        for color in colorList:
            condtion = (filter_pants['color'] == color)
            filter_color = filter_pants[condtion]
            index = random.randrange(0, len(list(filter_color.iloc)))
            i = filter_color.iloc[index]
            self.bottomList.append(Clothes(i[0], i[1], i[2], i[3], i[4], self.preprocessorTuple(i[6]), i[7]))

        # 여기서 일부로 데이터에 제한을 두려고 합니다.
        # 원래는 데이터 베이스에서 n개의 상의 옷, 하의 옷을 입력을 받아야 함.
        # 나중에 수정 요망.
        # limit = 5
        # count = 0

        # for i in filter_top.iloc:
        #     if limit == count:
        #         break;
        #     self.topList.append(Clothes(i[0], i[1], i[2], i[3], i[4], self.preprocessorTuple(i[6]), i[7]))
        #     count += 1
        # count = 0
        # for i in filter_pants.iloc:
        #     if limit == count:
        #         break;
        #     self.bottomList.append(Clothes(i[0], i[1], i[2], i[3], i[4], self.preprocessorTuple(i[6]), i[7]))
        #     count += 1

    def greedy(self):
        #1. 각 상의 옷이 하의 옷에 대한 선호도를 입력
        #2. 선호도에 따른 옷 매칭

        # 1. 각 상의 옷이 하의 옷에 대한 선호도를 입력
        # 선호도 기능은 코사인 유사도
        for top in self.topList:
            for bottom in self.bottomList:
                top.preferenceDict[bottom.name] = cos_sim(top.rgb, bottom.rgb)
            print('{}: {}'.format(top.name, top.preferenceDict.values()))
            top.sort()
        print()
        # 2. 선호도에 따른 옷 매칭
        pickList = []
        matchList = []
        for top in self.topList:
            top.matching(pickList, matchList)
        print(matchList)
        
        #출력
        count = 1
        for i in matchList:
            top = self.findClothes(self.topList, i[0])
            bottom = self.findClothes(self.bottomList, i[1])
            print('{}번째 추천: {}, {}'.format(count, top.name, bottom.name))
            print('상의 옷 url:', top.url)
            print('하의 옷 url:', bottom.url)
            print()
            count += 1

    #이분 매칭 서브 메소드
    def subBipartiteMatching(self, n, graphList, selectList, visitList):
        if visitList[n]:
            return False
        visitList[n] = True

        for num in graphList[n]:
            if selectList[num] == -1 or self.subBipartiteMatching(selectList[num], graphList, selectList, visitList):
                selectList[num] = n
                return True
        return False
    #이분 매칭, DFS 사용
    def bipartiteMatching(self):
        # 1. 선호도를 기준으로 원하는 상의 옷이 원하는 하의 옷을 선택 (나는 선호도 기준을 0.8로 잡음)
        # 2. 이분매칭

        # 1. 선호도를 기준으로 원하는 상의 옷이 원하는 하의 옷을 선택 (나는 선호도 기준을 0.8로 잡음)
        for top in self.topList:
            for j in range(len(self.bottomList)):
                top.preferenceDict[j] = cos_sim(top.rgb, self.bottomList[j].rgb)
            print('{}: {}'.format(top.name, top.preferenceDict.values()))
            top.sort()
        print()

        #그래프 생성
        graph = []
        for i in self.topList:
            tempList = []
            for j in i.preferenceDict:
                if j[1] > 0.9999:
                    tempList.append(j[0])
            graph.append(tempList)

        print(graph)

        # 2. 이분매칭
        # 선택된 정점 번호
        selected = [-1] * (len(self.bottomList) + 1)
        for i in range(len(self.topList)):
            visited = [False] * (len(self.topList))
            self.subBipartiteMatching(i, graph, selected, visited)
        print(selected)
    
        #출력
        count = 1
        for i in range(len(self.topList)):
            if selected[i] == -1:
                continue

            top = self.topList[i]
            bottom = self.bottomList[selected[i]]
            print('{}번째 추천: {}, {}'.format(count, top.name, bottom.name))
            print('상의 옷 url:', top.url)
            print('하의 옷 url:', bottom.url)
            print()
            count += 1
    def galeShapley(self):
        # 1. 상의 옷에 대한 하의 옷의 선호도 순위를 매김
        for top in self.topList:
            for j in range(len(self.bottomList)):
                top.preferenceDict[j] = cos_sim(self.bottomList[j].rgb, complementary_color(top))
            print('{}: {}'.format(top.name, top.preferenceDict.values()))
            top.sort()
        print()

        topPreferance = []
        for i in self.topList:
            tempList = []
            for j in i.preferenceDict:
                tempList.append(j[0])
            topPreferance.append(tempList)

        # 2. 하의 옷에 대한 상의 옷의 선호도 순위를 매김
        for bottom in self.bottomList:
            for j in range(len(self.topList)):
                bottom.preferenceDict[j] = cos_sim(self.topList[j].rgb, complementary_color(bottom))
            print('{}: {}'.format(bottom.name, bottom.preferenceDict.values()))
            bottom.sort()

        bottomPreferance = []
        for i in self.bottomList:
            tempList = []
            for j in i.preferenceDict:
                tempList.append(j[0])
            bottomPreferance.append(tempList)

        print(topPreferance)
        print(bottomPreferance)

        # 매칭
        n = len(topPreferance)
        unmatchedTop = [1] * n
        matchedBottom = [-1] * n
        matchIndex = [0] * n

        while sum(unmatchedTop) != 0:
            top = unmatchedTop.index(1)
            unmatchedTop[top] = 0
            bottom = topPreferance[top][matchIndex[top]]
            pref = bottomPreferance[bottom]
            matchIndex[top] += 1

            # 선택한 하의가 아직 매치되지 않은 경우
            if matchedBottom[bottom] == -1:
                matchedBottom[bottom] = top
            # 선택한 하의가 매치되었지만 새로운 상의의 선호도가 더 높은 경우
            elif pref.index(top) < pref.index(matchedBottom[bottom]):
                unmatchedTop[matchedBottom[bottom]] = 1
                matchedBottom[bottom] = top
            # 매치에 실패한 경우
            else:
                unmatchedTop[top] = 1

        matchList = []
        for i in range(n):
            matchList.append((matchedBottom[i], i))

        print(matchList)

        # 출력
        count = 1
        for i in matchList:
            top = self.topList[i[0]]
            bottom = self.bottomList[i[1]]
            print('{}번째 추천: {}, {}'.format(count, top.name, bottom.name))
            print('상의 옷 url:', top.url)
            print('하의 옷 url:', bottom.url)
            print()
            count += 1




if __name__ == '__main__':
    print('===============그리드 사용===============')
    obj1 = ClothesRecommend('여자', '여름')
    obj1.greedy()
    print('\n===============이분 매칭 사용 사용===============')
    obj2 = ClothesRecommend('남자', '겨울')
    obj2.bipartiteMatching()
    print('\n===============게일 셰플리 사용 사용===============')
    obj3 = ClothesRecommend('남자', '여름')
    obj3.galeShapley()
