#라이브러리
import gradio as gr
import PIL.Image as Image
import torch
from ultralytics import YOLO,ASSETS
from datetime import datetime
import numpy as np
#import matplotlib.pyplot as plt
import random
import os
import shutil
import tqdm
import glob

#현재 날씨
import requests
import json

city = "Seoul" #도시
apiKey = "25ae92023210f98156f2d65b5cdb15b6"
lang = 'kr' #언어
units = 'metric' #화씨 온도를 섭씨 온도로 변경
api = f"https://api.openweathermap.org/data/2.5/weather?q={city}&APPID={apiKey}&lang={lang}&units={units}"

result = requests.get(api)
result = json.loads(result.text)

print(result['main']['temp'])
current_weather = int(result['main']['temp'])
print(current_weather)

# 모델 불러오기
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "best.pt"
model = YOLO(path)
model = model.to(device)


#현재 날씨로 적정 옷 분류
if current_weather <= 15:
  iswarm = False
elif current_weather > 15:
  iswarm = True
#현재 날짜
now = datetime.now()




#이미지 예측하기
def predict_image(img):
    """Predicts objects in an image using a YOLOv8 model with adjustable confidence and IOU thresholds."""
    #이미지 예측 핵심 코드
    results = model.predict(source=img, conf=0.4,show_labels=True,imgsz=640
    )
    #옷 인식하고 분류
    bottom_wear = False
    top_wear = False
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        for box in r.boxes:
          class_id = int(box.data[0][-1])
          if 'jacket' in model.names[class_id]:
              top_wear = True

          if 'pants' in model.names[class_id]:
              bottom_wear = True

          if 'shirt' in model.names[class_id]:
              if 'jacket' in model.names[class_id]:
                top_wear = True

              top_wear = False

          if 'shorts' in model.names[class_id]:
              bottom_wear = False
          if 'skirt' in model.names[class_id]:
              bottom_wear = False

        #옷 판단
        if iswarm == top_wear:
          if iswarm == True:
            res =  ("날씨가 덥기 때문에 상의 패션은 적절하지 않습니다. 다시 선택해주세요!")
          if iswarm == False:
            res = ("날씨가 춥기 때문에 상의 패션은 적절하지 않습니다. 다시 선택해주세요!")

        if iswarm != top_wear:
            res =  ("오늘에 날씨에 딱맞게 선택하셨습니다. 상의 옷은 입으셔도 됩니다!")
        if iswarm == bottom_wear:
          if iswarm == True:
            res1 = ("날씨가 덥기 때문에 하의 패션은 적절하지 않습니다. 다시 선택해주세요!")
          elif iswarm == False:
            res1 = ("날씨가 춥기 때문에 하의 패션은 적절하지 않습니다. 다시 선택해주세요!")

        if iswarm != bottom_wear:
            res1 = ("오늘에 날씨에 딱맞게 선택하셨습니다. 하의 옷은 입으셔도 됩니다!")

    #날짜와 날씨
    weather = "{}년 {}월 {}일 오늘의 날씨는 {}도이므로".format(now.year, now.month, now.day , current_weather)


    return im,weather, res, res1

#핵심 인터페이스
iface = gr.Interface(
    theme='freddyaboulton/dracula_revamped' ,
    fn=predict_image,
    inputs=[
        #이미지 입력
        gr.Image(type="pil", label="Upload Image"),
    ],
    outputs=[
       #이미지 출력
       gr.Image(type="pil", label="결과"),
       #텍스트 출력
       gr.Textbox(label = "날씨"),
       gr.Textbox(label="코멘트"),
       gr.Textbox(label="코멘트")
       ],
    title="Weather Clothes Cecommendation AI",
    description="NOID",
    examples=[
        [ASSETS / "bus.jpg", 0.25, 0.45],
        [ASSETS / "zidane.jpg", 0.25, 0.45],
    ],

)
#시작
if __name__ == "__main__":
    iface.launch()
