import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import os
from pydub import *
import fastbook
from fastbook import *
model = load_learner("drum.pkl")

st.title("hi")
Dfile = st.file_uploader("Upload file here!")
x,y = librosa.load(Dfile)
ti=librosa.get_duration(x)
Beat = st.number_input("กรอก BPM")
Beat = Beat/60
sts=0
b=0
all=[]
countfname=1
for i in range(int(ti*Beat*2)):  
  sound = AudioSegment.from_mp3("/content/groove001.wav")
  StrtSec = sts
  EndSec = Beat*(i+1)/2
  StrtTime = StrtSec*1000
  EndTime = EndSec*1000
  extract = sound[StrtTime:EndTime]
  extract.export("/content/Half.wav", format="wav")

    

  x,y = librosa.load('/content/Half.wav')
  plt.figure(figsize=(12,4))
  a = librosa.feature.melspectrogram(x,sr=y,n_mels=550)
  b = librosa.power_to_db(a,ref=np.max)
  librosa.display.specshow(b,sr=y, x_axis='time', y_axis='mel')
  plt.colorbar(format='%+02.0f dB')
  plt.tight_layout()
  plt.savefig(f'{countfname}')
  wit = model.predict(f'{countfname}.png')
  all.append(wit[0])
  countfname+=1

  sts=EndSec
st.title(wit)