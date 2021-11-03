# Author: Sultan S. Alqahtani
# Date: 06/16/2021 
import fasttext
from numpy import rint

def doPrediction(model, text):
    val = model.predict(text, 1)
    label = None

    if(val[0][0] == '__label__nonsec-report'):
        label = 'not security'
    else:
        label = 'security'
    
    print("Text = "+text)
    print("Label= "+ label)
    print("Confidence Score = "+ str(val[1][0]))
    

if __name__ == '__main__':
    model = fasttext.load_model("model_ambari.bin")

    doPrediction(model, "buffer overflow not security report")
    #print(model.words)
    #print(model.labels)

