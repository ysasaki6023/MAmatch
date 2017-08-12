# -*- coding: utf-8 -*-
import os,sys,argparse,re,string,logging,random
#import tensorflow as tf
import numpy as np
import pandas as pd
from gensim import corpora, models, similarities
from gensim.models import word2vec
from keras.layers import Input, Dense, LSTM, merge, Lambda
#from keras.layers import Merge,Layer
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K

class DataAccessor:
    def __init__(self,verbose=False):
        self.nDim = 100
        self.puncReStr = string.punctuation
        self.puncReStr = "".join([x for x in self.puncReStr if x not in ["-","'"]])
        self.puncReStr = r"[%s]"%self.puncReStr
        self.verbose = verbose
        return

    def buildW2Vfile(self,filePath,outPath):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = word2vec.LineSentence(filePath)
        model = word2vec.Word2Vec(sentences, sg=1, size=self.nDim,min_count=1,window=10,hs=1)
        model.save(outPath)
        return

    def loadCSV(self,fPath):
        self.d = d = pd.read_csv(fPath)
        self.colList = d.columns
        self.nLines = d.shape[0]
        return

    def loadW2V(self,fPath):
        self.w2v_model = word2vec.Word2Vec.load(fPath)
        return

    def getColumnPairs_str(self,index,colPairs):
        d = self.d
        #return d[index][colPairs[0]],d[index][colPairs[1]]
        return d[colPairs[0]][index],d[colPairs[1]][index]

    def getColumnPairs_w2v(self,index,colPairs):
        strPairs = self.getColumnPairs_str(index,colPairs)
        strPairs = [self.cleanUpStr(x) for x in strPairs]
        return self.str_to_w2v(strPairs[0]),self.str_to_w2v(strPairs[1])

    def cleanUpStr(self,inStr):
        inStr = inStr.replace("\r\n","").replace("\n","")
        inStr = re.sub(self.puncReStr," ",inStr)
        inStr = re.sub("\s+"," ",inStr).strip().lower()
        return inStr

    def str_to_w2v(self,inStr):
        inStr = inStr.split()
        res = np.zeros( (len(inStr), self.nDim), dtype=np.float32)
        for i,w in enumerate(inStr):
            try:
                res[i,:] = self.w2v_model[w]
            except:
                if self.verbose: print "Not key found: %s"%w
                continue
        return res

class net:
    def __init__(self,args,dAcc,columnPairs,goodFrac=0.5):
        self.dAcc    = dAcc
        self.nDim    = dAcc.nDim
        self.nBatch  = args.nBatch
        self.nLength = args.nLength
        self.columnPairs = columnPairs
        self.goodFrac = goodFrac
        self.buildModel()
        return

    def buildModel(self):
        def create_base_network(input_shape):
            seq = Sequential()
            seq.add(LSTM(32,input_shape=input_shape))
            seq.add(Dense(32,activation="sigmoid",kernel_regularizer=l2(1e-3)))
            return seq
        def euclidean_distance(vects):
            x, y = vects
            return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
        def eucl_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)

        input_shape = (self.nLength,self.nDim)

        inX1 = Input(input_shape,name="input_1")
        inX2 = Input(input_shape,name="input_2")

        base_network = create_base_network(input_shape)

        outX1 = base_network(inX1)
        outX2 = base_network(inX2)

        distance = merge([outX1,outX2],mode = euclidean_distance, output_shape=eucl_dist_output_shape,name="distance")
        model = Model(inputs=[inX1, inX2], outputs=distance)

        optimizer = Adam(1e-4)
        model.compile(loss="mean_squared_error",optimizer=optimizer)

        model.summary()
        self.model = model
        return

    def getOneBatch(self,columnPairs,goodFrac=0.5):

        nGood = int(self.nBatch * goodFrac)
        nBad  = self.nBatch - nGood

        batchVec1  = np.zeros( (self.nBatch, self.nLength, self.nDim), dtype=np.float32)
        batchVec2  = np.zeros( (self.nBatch, self.nLength, self.nDim), dtype=np.float32)
        batchTruth = np.zeros( (self.nBatch, 1)        , dtype=np.float32)

        itemList1 = []
        itemList2 = []
        wordList1 = []
        wordList2 = []
        batchWord1 = []
        batchWord2 = []

        # まずは正解を入れ込む
        cnt = 0
        for i in np.random.randint(0,self.dAcc.nLines,nGood):
            v1,v2 = self.dAcc.getColumnPairs_w2v(i,columnPairs)
            w1,w2 = self.dAcc.getColumnPairs_str(i,columnPairs)
            itemList1.append(v1)
            itemList2.append(v2)
            wordList1.append(w1)
            wordList2.append(w2)
            batchWord1.append(w1)
            batchWord2.append(w2)
            l1 = min(v1.shape[0],self.nLength)
            l2 = min(v2.shape[0],self.nLength)
            batchVec1[cnt,:l1,:] = v1[:l1]
            batchVec2[cnt,:l2,:] = v2[:l2]
            batchTruth[cnt] = +1.0 # 正解
            cnt += 1

        # 次に不正解を入れ込む
        assert self.nBatch>1
        cnt = -1
        while True:
            idx1 = random.randint(0,len(itemList1)-1)
            idx2 = random.randint(0,len(itemList2)-1)
            if idx1==idx2: continue
            cnt += 1
            if (nGood+cnt)>=self.nBatch: break
            v1 = itemList1[idx1]
            v2 = itemList2[idx2]
            w1 = wordList1[idx1]
            w2 = wordList2[idx2]
            batchWord1.append(w1)
            batchWord2.append(w2)
            l1 = min(v1.shape[0],self.nLength)
            l2 = min(v2.shape[0],self.nLength)
            batchVec1[nGood + cnt,:l1] = v1[:l1]
            batchVec2[nGood + cnt,:l2] = v2[:l2]
            batchTruth[nGood + cnt] = -1.0 # 不正解

        ### For debug
        #for i in range(self.nBatch):
        #    with open("temp/%d_wd.txt"%i,"w") as f:
        #        f.write(unicode(batchWord1[i]).encode("shift-jis"))
        #        f.write("\n")
        #        f.write(unicode(batchWord2[i]).encode("shift-jis"))

        #    np.savetxt("temp/%d_v1.csv"%i, batchVec1[i], delimiter=",")
        #    np.savetxt("temp/%d_v2.csv"%i, batchVec2[i], delimiter=",")
        #sys.exit(-1)
        return batchTruth,batchVec1,batchVec2,batchWord1,batchWord2

    def yieldOne(self):
        nGood = int(self.nBatch * self.goodFrac)
        nBad  = self.nBatch - nGood

        while True:
            batchVec1  = np.zeros( (self.nBatch, self.nLength, self.nDim), dtype=np.float32)
            batchVec2  = np.zeros( (self.nBatch, self.nLength, self.nDim), dtype=np.float32)
            batchTruth = np.zeros( (self.nBatch, 1)        , dtype=np.float32)

            itemList1 = []
            itemList2 = []
            wordList1 = []
            wordList2 = []
            batchWord1 = []
            batchWord2 = []

            # まずは正解を入れ込む
            cnt = 0
            for i in np.random.randint(0,self.dAcc.nLines,nGood):
                v1,v2 = self.dAcc.getColumnPairs_w2v(i,self.columnPairs)
                w1,w2 = self.dAcc.getColumnPairs_str(i,self.columnPairs)
                itemList1.append(v1)
                itemList2.append(v2)
                wordList1.append(w1)
                wordList2.append(w2)
                batchWord1.append(w1)
                batchWord2.append(w2)
                l1 = min(v1.shape[0],self.nLength)
                l2 = min(v2.shape[0],self.nLength)
                batchVec1[cnt,:l1,:] = v1[:l1]
                batchVec2[cnt,:l2,:] = v2[:l2]
                batchTruth[cnt] = +1.0 # 正解
                cnt += 1

            # 次に不正解を入れ込む
            assert self.nBatch>1
            cnt = -1
            while True:
                idx1 = random.randint(0,len(itemList1)-1)
                idx2 = random.randint(0,len(itemList2)-1)
                if idx1==idx2: continue
                cnt += 1
                if (nGood+cnt)>=self.nBatch: break
                v1 = itemList1[idx1]
                v2 = itemList2[idx2]
                w1 = wordList1[idx1]
                w2 = wordList2[idx2]
                batchWord1.append(w1)
                batchWord2.append(w2)
                l1 = min(v1.shape[0],self.nLength)
                l2 = min(v2.shape[0],self.nLength)
                batchVec1[nGood + cnt,:l1] = v1[:l1]
                batchVec2[nGood + cnt,:l2] = v2[:l2]
                batchTruth[nGood + cnt] = -1.0 # 不正解

            yield ({"input_1":batchVec1,"input_2":batchVec2},{"distance":batchTruth})

    def train(self):
        self.model.fit_generator(self.yieldOne(),steps_per_epoch=int(self.dAcc.nLines/self.nBatch),use_multiprocessing=True, max_queue_size=10, workers=1)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode","-m",dest="mode",type=str,choices=["train"],default="train")
    parser.add_argument("--nBatch" ,"-b",dest="nBatch",type=int,default=64)
    parser.add_argument("--nLength","-l",dest="nLength",type=int,default=20)
    parser.add_argument("--learnRate","-r",dest="learnRate",type=float,default=1e-3)
    parser.add_argument("--saveFolder","-s",dest="saveFolder",type=str,default="save")

    args = parser.parse_args()
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("wiki.w2v")
    #d.buildW2Vfile("all.csv",outPath="wiki.w2v")
    n = net(args,d,columnPairs=[u"Acquiror business description(s)",u"Target business description(s)"],goodFrac=0.5)
    n.train()