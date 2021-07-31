import fasttext

model = fasttext.train_supervised(input="cooking.train")
#model = fasttext.load_model("model_cooking.bin")


print(model.predict("which baking dish is best to bake banana bread ?"))

print(model.test("cooking.valid"))

print(model.test_label("cooking.valid"))