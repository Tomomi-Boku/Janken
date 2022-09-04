import os
import math
import cv2
import numpy as np
import mediapipe as mp
import time
import pickle
import pandas as pd  
import random

### 制作するにあたって参考にしたウェブサイト　https://qiita.com/Ihmon/items/17cb9e3bd3175608fcd5

#スコアカードに載せる勝敗カウント
drawCount = 0
winCount = 0
loseCount = 0


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        zList = []
        bbox = ()
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                xList.append(cx)
                yList.append(cy)
                zList.append(cz)
                # print(id, cx, cy)
                lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            zmin, zmax = min(zList), max(zList)
            bbox = xmin, ymin, zmin, xmax, ymax, zmax
            if draw:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[3], bbox[4]), 
                                (0, 255, 0), thickness = 2)

        return lmList, bbox


# 手首の座標を原点とし、20点との距離を求める。
def get_distance(wklst):
    lst_o = list([wklst[0][1], wklst[0][2], wklst[0][3]])      # 手首のx,y,z
    lst_dist = []
    for i in range(1,21):
        lst_t = list([wklst[i][1], wklst[i][2], wklst[i][3]])  # 手首以外の20箇所のx,y,z
        length = math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(lst_o, lst_t)))
        lst_dist.append(length)    # 手首の座標からの20個の距離
    lst_dist.append(0)
    return lst_dist
    
#じゃんけんのアルゴリズム
def janken(playerhand,PChand):  #プレイヤーの手・PCの手：両方　0,1,2
    global drawCount
    global winCount
    global loseCount

    if playerhand == 0: #プレイヤーがグー
        if PChand == 0: #go
            judge = "DRAW!"
        elif PChand == 1:   #choki
            judge = "WIN!"
        else :  #pa
            judge = "LOSE!"
    
    elif playerhand == 1:   #プレイヤーがチョキ
        if PChand == 1: 
            judge = "DRAW!"
        elif PChand == 2:   
            judge = "WIN!"
        else :  
            judge = "LOSE!"
    
    else:  #プレイヤーがパー
        if PChand == 2:
            judge = "DRAW!"
        elif PChand == 0:   
            judge = "WIN!"
        else :  
            judge = "LOSE!"
            
    #それぞれの場合に該当する画像の送出
    if judge=="DRAW!":
        drawCount += 1
        finish_img = cv2.imread('./item/draw.png', 3)
    elif judge=="WIN!":
        winCount += 1
        finish_img = cv2.imread('./item/win.png', 3)
    else:
        loseCount += 1
        finish_img = cv2.imread('./item/lose.png', 3)

    return finish_img


def main():
    #使用する画像の読み込み
    Go_img = cv2.imread('./item/go.png', 3)
    Choki_img = cv2.imread('./item/choki.png', 3)
    Pa_img = cv2.imread('./item/pa.png', 3)
    start_img = cv2.imread('./item/start.png', 3)
    wait_img = cv2.imread('./item/wait.png', 3)
    score_img = cv2.imread('./item/scorecard2.png', 3)

    #スタート画像の表示
    cv2.imshow('start!', start_img)
    cv2.waitKey(1000)
    cv2.destroyWindow('start!')
    
    #カメラの画像から手を検出
    myPath = os.path.join(os.getcwd(), 'pose_data_2')
    result_dict = {0: 'Go', 1: 'Choki', 2: 'Paa'}
    
    #作成したモデルの取得
    with open('myPipe.dat', 'rb') as f:
        ridge_cls = pickle.load(f)
    clmn_lst = [str(i) for i in range(1,21)]
    clmn_lst.append('cls')
    df = pd.DataFrame(columns=clmn_lst)
    
    #カメラ起動
    cap = cv2.VideoCapture(0)
    wCam, hCam = 640, 480
    cap.set(3, wCam)    # 3: CV_CAP_PROP_FRAME_WIDTH
    cap.set(4, hCam)    # 4: CV_CAP_PROP_FRAME_HEIGHT

    #検出機のインスタンス化
    detector = handDetector(maxHands=1, detectionCon=0.8, trackCon=0.8)
    
    
    for gameCount in range(3):
        gameCount += 1      #何回目の勝負か
        gameNum = 0 #後出し防止→毎回初期化
        
        choice_list = [0,1,2]
        pc = random.choice(choice_list)   #ランダムに出されるpcの手（0,1,2）
        
        #待機画面の表示
        cv2.rectangle(wait_img, (0, 0), (300, 100), (0, 255, 255), -1)
        cv2.putText(wait_img, "game count : "+str(gameCount), 
                            (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
        cv2.imshow('wait!', wait_img)
        cv2.waitKey(2000)
        cv2.destroyWindow('wait!')
        
        while True:
            success, img = cap.read()
            img = detector.findHands(img)
            lmList, bbox = detector.findPosition(img, draw=False)
            if len(lmList) != 0:
                dist_lst = get_distance(lmList)
                df.loc[0] = dist_lst
                rtn = int(ridge_cls.predict(df.iloc[0:1, 0:20]))   # 検出されたプレイヤーの手（0,1,2）
                
                if pc == 0:  #グーの場合
                    gameNum += 1
                    pc_img = Go_img
                elif pc == 1:    #チョキの場合
                    gameNum += 1
                    pc_img = Choki_img
                else:   #パーの場合
                    gameNum += 1
                    pc_img = Pa_img
                
                cv2.rectangle(pc_img, (0, 0), (480, 100), (0, 255, 255), -1)
                cv2.putText(pc_img, "Your hand: "+result_dict[rtn], 
                                (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                cv2.imshow('yourhand', pc_img)
                cv2.waitKey(5000)
                cv2.destroyWindow('yourhand')
                
            
            #後出しができないプログラム：プレイヤーが手を変更したら終了する
            if gameNum != 0:
                result_img = janken(rtn,pc)
                cv2.putText(result_img, "Your hand: "+result_dict[rtn]+" & PC hand: "+result_dict[pc], 
                                (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
                cv2.imshow('result!', result_img)
                cv2.waitKey(5000)
                cv2.destroyWindow('result!')
                break
        
    #スコアカードを表示させてプログラムを終了する    
    cv2.putText(score_img, str(winCount)+"  :  "+str(drawCount)+"  :  "+str(loseCount), 
                            (180, 260), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 3)
    cv2.imshow('your score!', score_img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
    #カメラの終了
    cap.release()


if __name__ == "__main__":
    main()
    