import fasttext

#model = fasttext.train_supervised(input="ambari.valid")
model = fasttext.load_model("./data/best_kfold_models/best_k5_Chromium_model.bin")

'''
print(model.predict("access control  security report"))
print(model.predict("sql injection security report"))

print(model.predict("xss security report"))
'''
print(model.test("./data/bug_reports/Chromium.valid"))


#print(model.test_label("./data/bug_reports/Chromium.valid"))
