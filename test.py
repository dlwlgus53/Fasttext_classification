import fasttext
import pandas as pd
import pdb
from tqdm import tqdm

model1 = fasttext.load_model("./model/fasttext_1.bin")
model2 = fasttext.load_model("./model/fasttext_2.bin")
model3 = fasttext.load_model("./model/fasttext_3.bin")
test_org =  pd.read_csv("../../data/dev.txt", header = None, sep = '\t')
test_text1 = pd.read_csv("../../data/dev_level1.csv", header = None, sep = '\t')

checking = pd.read_csv("./checking_sheet.csv")
answers = []
for sent in tqdm(test_text1[1]):
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
    
answer_sheet = pd.DataFrame({'predict': answers ,'answer' : test_org[0]})
acc = sum(answer_sheet['predict'] == answer_sheet['answer'])/len(answer_sheet['predict'])
answer_sheet.to_csv("./answer_sheet.csv", index = False)
print("Acc : "  +str(acc*100) + "%")