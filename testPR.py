import fasttext

#model = fasttext.train_supervised(input="ambari.valid")
model = fasttext.load_model("model_ambari.bin")

'''
print(model.predict("access control  security report"))
print(model.predict("sql injection security report"))

print(model.predict("xss security report"))

print(model.test("./data/kfold_train_test_data/ft_k10_cve_cwe_summary_clean.train"))
'''

print(model.test_label("ambari.valid"))
