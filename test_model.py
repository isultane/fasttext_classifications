# Author: Sultan S. Alqahtani
# Date: 06/16/2021 
import fasttext

model = fasttext.load_model("./data/kfold_train_test_data/best_kfold5_model.bin")

#print(model.words)
#print(model.labels)


print(model.predict("buffer overflow security report", k=3))
print(model.predict("sql injection security report", k=3))
print(model.predict("denail of service dos security report", k=3))

