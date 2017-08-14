# -*- coding: utf-8 -*-
import argparse,sys,os
import numpy as np
from net import DataAccessor,net
from keras.models import load_model

def memorize(f):
    cache = {}
    def helper(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    return helper

class net_test(net):
    def __init__(self,args,dAcc,columnPairs):
        self.reload  = args.reload
        self.nBatch  = args.nBatch
        self.nLength = args.nLength
        self.nDim = 100
        self.learnRate = 0
        self.dAcc = dAcc
        self.columnPairs = columnPairs
        self.buildModel()
        self.reloadModel(self.reload)
        return

    def reloadModel(self,fPath):
        self.model.load_weights(fPath)
        print "model loaded from %s"%fPath
        return

    def testOne(self,txt):
        vec = self.dAcc.str_to_w2v(txt,fixedLength=args.nLength)
        y1 = self.outX1.predict({"input_1":np.expand_dims(vec,axis=0)})
        return y1[0]

    def testAll(self,y1):
        pos = 0
        score = []
        words = []

        while pos<self.dAcc.nLines:
            vList = []
            num_processed = 0
            for i in range(self.nBatch):
                if pos+i>=self.dAcc.nLines:
                    vList.append(v) # このwhileループがスタートしているということは、vは少なくとも1回は呼ばれているはず。それをダミーとして入れ込む
                    continue
                w = self.dAcc.getColumnPairs_str(pos+i,self.columnPairs)
                w = w[1] # 一応、買われた側の値を使う
                v = self.dAcc.str_to_w2v(w,fixedLength=args.nLength)
                words.append(w)
                vList.append(v)
                num_processed += 1
            vList = np.array(vList)
            y2 = self.outX1.predict({"input_1":vList})
            sc = np.dot(y2,y1) 
            for i in range(num_processed):
                score.append(sc[i])
            pos += num_processed
        return score,words

    def test(self,txt,nMax=5):
        # txtに基づいて相性の良いindexを5件選定
        y1 = self.testOne(txt) # まず、入力されたものを変換
        score, words = self.testAll(y1)

        print
        print "match for :",txt
        cnt = 0
        goodOnes = []
        for idx in np.argsort(score)[::-1]:
            if words[idx] in goodOnes: continue
            print cnt+1,score[idx],words[idx]
            goodOnes.append(words[idx])
            cnt += 1
            if cnt>nMax: break


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nBatch" ,"-b",dest="nBatch",type=int,default=256)
    parser.add_argument("--nLength","-l",dest="nLength",type=int,default=10)
    parser.add_argument("--reload","-r",dest="reload",type=str,default=None)

    args = parser.parse_args()
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("w2v/wiki.w2v")
    t = net_test(args,d,columnPairs=(u"Acquiror business description(s)",u"Target business description(s)"))
    t.test("Chemicals distributor, Rubber and latex products distributor")
    t.test("Biopharmaceuticals developer, Biopharmaceuticals manufacturer, Consumer healthcare products manufacturer, Infant food manufacturer, Pharmaceutical products manufacturer")
    t.test("Food condiments and sauces manufacturer holding company, Soft and spreadable cheeses manufacturer holding company, Soft drinks manufacturer holding company")
    t.test("Broadband telecommunications services holding company, Cable television services holding company, High-speed data and voice services holding company, Wireless backhaul services holding company")
    t.test("Pharmaceutical solutions research and development services, Pharmaceuticals manufacturer")
