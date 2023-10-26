import scipy
import os 

sample_rate = 44.1*10**3

def length(path):
    try:
        rate, data = scipy.io.wavfile.read(path)
        minutes = (data.shape[0]/sample_rate)/60
        print(f"{path} {minutes}")
    except:
        print("Rut Ro an exception occurred")
        print(f"the culprit: {path}")


path = "../data/channels_1_2/"
files = os.listdir(path)

for x in files:
    length(os.path.join(path, x))
