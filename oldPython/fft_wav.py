import scipy 
from scipy.fft import fft, ifft
import os 
import numpy as np

sample_rate = 44.1*10**3
test_path = "../A3CarScene_AudioData/audio-ch1-20221017-3.wav"

def my_fft(path):
    print(path)
    try:
        rate, data = scipy.io.wavfile.read(path)
        y = fft(data) 
        return y
    except:
        print("Rut Ro an exception occurred")
        print(f"the culprit: {path}")
    return "bad"

path = "../A3CarScene_AudioData/"
files = os.listdir(path)
os.mkdir("ffts")
for x in files:
    try:
        deextensioned = os.path.splitext(x) 
        whole_path = os.path.join(path, x)
        fft_var = my_fft(whole_path)
        print(fft_var.shape)
        np.save(os.path.join("ffts", (deextensioned[0] + ".npy")), fft_var)
    except:
        print("probably fft errored out")
