import fasttext
import pdb
import pandas as pd
model1 = fasttext.train_supervised(input="../../data/train_level1.csv",
                                  epoch=100,
                                  minn =    2,
                                  minCount = 1,
                                  bucket = 20000,
                                  lr = 1,
                                  wordNgrams=2,
                                  dim=80,
                                  )

model1.save_model("./model/fasttext_1.bin")

model2 = fasttext.train_supervised(input="../../data/train_level2.csv",
                                  epoch=100,
                                  minn =    2,
                                  minCount = 1,
                                  bucket = 20000,
                                  lr = 1,
                                  wordNgrams=2,
                                  dim=80,
                                  )

model2.save_model("./model/fasttext_2.bin")

model3 = fasttext.train_supervised(input="../data/train_level3.csv",
                                  epoch=100,
                                  minn =    2,
                                  minCount = 1,
                                  bucket = 20000,
                                  lr = 1,
                                  wordNgrams=2,
                                  dim=80,
                                  )

model3.save_model("./model/fasttext_3.bin")

print(model1.test("../data/dev_level1.csv"))
print(model2.test("../data/dev_level2.csv"))
print(model3.test("../data/dev_level3.csv"))




