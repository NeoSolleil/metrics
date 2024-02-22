#coding:utf-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fasttext
from pprint import pprint
import sys
import re

import numpy as np

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


args = sys.argv
model_path = ".../fasttext_PATH"#fasttextのモデルへのパス
wv = fasttext.load_model(model_path)

aq=[]

with open('.../ref_PATH', 'r') as f:#参照訳へのパス
    aq = f.read()

az= aq.split("\n")#改行

sannsyou=[]

last = az.pop(-1)

for i in range(len(az)):
    fv=az[i].split(" ")
    if fv[-1] == "":
        fv.remove("")
    sannsyou.append(fv)
#  sannsyou に参照訳の１つの文が１つの単語ごとに入っている

gt=[]
with open('.../MT_PATH', 'r') as f:#MT訳へのパス
    gt = f.read()

gb= gt.split("\n")#改行

MT=[]
llast = gb.pop(-1)
for i in range(len(gb)):
    hy=gb[i].split(" ")
    hy.remove("")
    MT.append(hy)
#  MT にMT訳の１つの文が１つの単語ごとに入っている

naiyatu=[]
for i in range(len(MT)):#mt訳のいくつめの文か
#    print("何文目の文なの,,,,,,,,,,,,,,,",i)
#    print(len(MT))
#    print(MT[i])


#↓FASTETEXTを使用して単語を置き換える部分↓

    for j in range(len(MT[i])):

        if MT[i][j] not in aq:#参照役のどこにも存在しないMT役にしかない単語を検索

            if MT[i][j] not in naiyatu:#今までの新規に追加した単語で被りがないかを確認
                naiyatu.append(MT[i][j])#参照役のどこにも存在しないMT役にしかない単語（被りなし）
                x=wv.get_word_vector(MT[i][j])#１単語ごとにベクトルは変わっている
                qw=[]

                for k in range(len(sannsyou[i])):#fastetextを使用して類似度を計算するところ
                    y=wv.get_word_vector(sannsyou[i][k])
                    z=cos_sim(x, y)
                    qw.append(z)

                    if len(qw) == len(sannsyou[i]):#新たな単語を使用して文を生成するところ
                        we=[]
                        qqww=qw.index(max(qw))
                        we.append(sannsyou[i][qqww])
                        rresult = " ".join(s for s in we)
                        sannsyou[i][qqww]=MT[i][j]
                        result = ' '.join(s for s in sannsyou[i])
                        sannsyou[i][qqww]=rresult
                        print(result)
