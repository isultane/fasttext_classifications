# Author: Sultan S. Alqahtani
# Date: 06/16/2021 
import fasttext

model = fasttext.load_model("model_cve.bin")

#print(model.words)
#print(model.labels)


print(model.predict("buffer overflow security report", k=3))
print(model.predict("sql injection security report", k=3))
print(model.predict("denail of service dos security report", k=3))

