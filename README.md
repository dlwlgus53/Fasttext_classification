# Fasttext_classification

[FastText](https://fasttext.cc/docs/en/supervised-tutorial.html) 를 이용한 발화의도 예측 모델


docker image 
폴더 위치(숨김폴더로 되어있음. 이유는 모르겠다..)
```
cd .ssh/text/fasttext2
```

학습
```
python train.py
```

dev.txt.를 이용한 test
```
python test.py
```

test.txt 를 이용한 test
```
python predict.py
```
