import tkinter
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cvlib
from cvlib.object_detection import draw_bbox
from tkinter import*
import tkinter as tk
from tkvideo import tkvideo
import PIL.Image, PIL.ImageTk 
import ffmpeg as ffmpeg
import io
import sys


# window = tkinter.Tk()
# window.title("OpenCV and Tkinter")
# window.geometry("1400x1100")

#vidPath=r"test.mp4"
capture=cv2.VideoCapture("test.mp4")

def display_frames(frames):
    fig=plt.figure(figsize=(10,6))
    mov=[]
    for i in range(len(frames)):
        img=plt.imshow(frames[i],animated=True)
        plt.axis('off')
        mov.append([img])

    anime = animation.ArtistAnimation(fig,mov,interval=20,repeat_delay=1000)

    plt.show()
    return anime

def display_diff_frames(frames,threshold):
  fig=plt.figure(figsize=(10,6))
  mov=[]
  for i in range(len(frames)):
    img=plt.imshow(frames[i]>int(threshold),cmap='gray',animated=True)
    plt.axis('off')
    mov.append([img])

  anime = animation.ArtistAnimation(fig,mov,interval=50,repeat_delay=1000)

  plt.show()
  return anime

frames=[]
i=0
while(1):
  ret,frame=capture.read()
  if ret:
    frame=cv2.resize(frame,dsize=(600,400))
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frames.append(frame)
  else:
    break
capture.release()

n=[]
diff_frames=[]
for i, frame in enumerate(frames):

  if i==0:
    left=np.float32(frame)
  else:
    diff_frame=(np.float32(frame)-left)**2
    diff_allchannel=np.sum(diff_frame,axis=2)
    k_r=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    diff_allchannel=cv2.morphologyEx(np.float32(diff_allchannel),cv2.MORPH_ERODE,k_r)
    if diff_allchannel.mean()>10:
      diff_frames.append(diff_allchannel)
      bbox,labels,conf=cvlib.detect_common_objects(frame)
      frame=draw_bbox(frame,bbox,labels,conf)
      n.append(frame)
    left=np.float32(frame)

writer = cv2.VideoWriter("output.avi",
cv2.VideoWriter_fourcc(*"MJPG"), 30,(640,480))

for frame in range(1000):
    writer.write(np.random.randint(0, 255, (480,640,3)).astype('uint8'))





capture.release()
writer.release()
cv2.destroyAllWindows()