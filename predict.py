import fasttext
import pandas as pd
import pdb
from tqdm import tqdm

model1 = fasttext.load_model("./model/fasttext_1.bin")
model2 = fasttext.load_model("./model/fasttext_2.bin")
model3 = fasttext.load_model("./model/fasttext_3.bin")
test =  pd.read_csv("../../data/test.txt", header = None, sep = '\t')
checking = pd.read_csv("./checking_sheet.csv")

answers = []
for sent in tqdm(test[0]):
    checking['score'] =0 
    ans = model1.predict(sent, k = 10)    
    for i in range(10):
        label = ans[0][i][9:]
        score =  ans[1][i]
        checking.loc[checking['level1'] == label, 'score'] += score
        ans = model1.predict(sent, k = 10)    
    
    ans = model2.predict(sent, k = 10)        
    for i in range(10):
        label = ans[0][i][9:]
        score =  ans[1][i]
        checking.loc[checking['level2'] == label, 'score'] += score
        
    ans = model3.predict(sent, k = 10)    
    for i in range(10):
        label = ans[0][i][9:]
        score =  ans[1][i]
        checking.loc[checking['level3'] == label, 'score'] += score
    final_ans = checking.sort_values(by = 'score', ascending = False).iloc[0]
    final_ans_label = ("-").join(final_ans[['level1', 'level2', 'level3']])
    answers.append(final_ans_label)
    
answer_sheet = pd.DataFrame({'predict': answers})
answer_sheet.to_csv("./predict_sheet.csv", index = False)