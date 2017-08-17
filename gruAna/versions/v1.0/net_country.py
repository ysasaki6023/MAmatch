# -*- coding: utf-8 -*-
import os,sys,argparse,re,string,logging,random
#import tensorflow as tf
import numpy as np
import pandas as pd
from gensim import corpora, models, similarities
from gensim.models import word2vec
from keras.layers import Input, Dense, LSTM, merge, Lambda, GRU, Dot, Reshape, Concatenate, Flatten, Dropout, Bidirectional
from keras.constraints import min_max_norm
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
import keras.backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

#from keras.utils.visualize_util import plot

def memorize(f):
    cache = {}
    def helper(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    return helper

class DataAccessor(object):
    def __init__(self,verbose=False):
        self.randomSeed = 99 # これによって、Train/Testが変更されるので注意必要
        self.validFrac = 0.1 # Validationで使う割合
        self.nDim = 100
        self.puncReStr = string.punctuation
        self.puncReStr = "".join([x for x in self.puncReStr if x not in ["-","'"]])
        self.puncReStr = r"[%s]"%self.puncReStr
        self.verbose = verbose
        return

    def buildW2Vfile(self,filePath,outPath,min_count=1000,window=5):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = word2vec.LineSentence(filePath)
        model = word2vec.Word2Vec(sentences, size=self.nDim,min_count=min_count,window=window)
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

    @memorize
    def getColumnPairs_str(self,index,colPairs):
        d = self.d
        return d[colPairs[0]][index],d[colPairs[1]][index]

    @memorize
    def getColumnPairs_w2v(self,index,colPairs):
        strPairs = self.getColumnPairs_str(index,colPairs)
        strPairs = [self.cleanUpStr(x) for x in strPairs]
        return self.str_to_w2v(strPairs[0]),self.str_to_w2v(strPairs[1])

    def getColumnPairs_w2v_fixedLength(self,index,colPairs,fixedLength):
        strPairs = self.getColumnPairs_str(index,colPairs)
        strPairs = [self.cleanUpStr(x) for x in strPairs]
        return self.str_to_w2v(strPairs[0],fixedLength=fixedLength),self.str_to_w2v(strPairs[1],fixedLength=fixedLength)

    def cleanUpStr(self,inStr):
        inStr = inStr.replace("\r\n","").replace("\n","")
        inStr = re.sub(self.puncReStr," ",inStr)
        inStr = re.sub("\s+"," ",inStr).strip().lower()
        return inStr

    def str_to_w2v(self,inStr,fixedLength=None):
        inStr = inStr.split()
        if fixedLength:  res = np.zeros( (fixedLength, self.nDim), dtype=np.float32)
        else:            res = np.zeros( (len(inStr) , self.nDim), dtype=np.float32)
        for i,w in enumerate(inStr):
            try:
                res[i,:] = self.w2v_model[w]
            except:
                if self.verbose: print "Not key found: %s"%w
                continue
        return res

    def getTargetLines(self,mode="all"):
        assert mode in ["all","train","valid"]
        random.seed(self.randomSeed) # 重要
        if   mode=="all"  : targetLines = range(self.nLines)
        else:
            validLines = random.sample(range(self.nLines), int(self.nLines * self.validFrac))
            trainLines   = list(set(range(self.nLines)) - set(validLines))
            if   mode=="train": targetLines = trainLines
            elif mode=="valid": targetLines = validLines
        return targetLines

class net(object):
    def __init__(self,args,dAcc,columnPairs,goodFrac=0.5):
        self.dAcc    = dAcc
        self.nDim    = dAcc.nDim
        self.nBatch  = args.nBatch
        self.nLength = args.nLength
        self.columnPairs = columnPairs
        self.goodFrac = goodFrac
        self.learnRate = args.learnRate
        self.buildModel()
        return

    def yieldOne(self,mode="all"):
        nGood = int(self.nBatch * self.goodFrac)
        nBad  = self.nBatch - nGood

        targetLines = self.dAcc.getTargetLines(mode)
        if mode=="train": self.trainLines = targetLines
        if mode=="valid": self.validLines = targetLines

        while True:
            print "a"
            batchVec1  = np.zeros( (self.nBatch, self.nLength, self.nDim), dtype=np.float32)
            batchVec2  = np.zeros( (self.nBatch, self.nLength, self.nDim), dtype=np.float32)
            batchTruth = np.zeros( (self.nBatch, 1)        , dtype=np.int32)

            itemList1 = []
            itemList2 = []
            wordLists = []

            # まずは正解を入れ込む
            cnt = 0
            for i in random.sample(targetLines,nGood):
                print i,"1"
                v1,v2 = self.dAcc.getColumnPairs_w2v(i,self.columnPairs)
                print i,"2"
                w1,w2 = self.dAcc.getColumnPairs_str(i,self.columnPairs)
                print i,"3"
                itemList1.append(v1)
                itemList2.append(v2)
                wordLists.append((w1,w2))
                print i,"4"
                l1 = min(v1.shape[0],self.nLength)
                l2 = min(v2.shape[0],self.nLength)
                batchVec1[cnt,:l1,:] = v1[:l1]
                batchVec2[cnt,:l2,:] = v2[:l2]
                batchTruth[cnt] = 1 # 正解
                cnt += 1
                print i,"5"

            print "k"
            # 次に不正解を入れ込む
            assert self.nBatch>1
            cnt = -1
            for _ in range(nBad*2): # nBadの2倍を上限にループ。すべての行が同じになっている場合に発生する無限ループへ対処
                idx1 = random.randint(0,len(itemList1)-1)
                idx2 = random.randint(0,len(itemList2)-1)
                if idx1==idx2: continue
                w1 = wordLists[idx1][0]
                w2 = wordLists[idx2][1]
                if (w1,w2) in wordLists:continue # この条件を満たすと、正解データになってしまうので削除。データの中には繰り返しが多く含まれているので、こうなるパターンもある。
                cnt += 1
                if (nGood+cnt)>=self.nBatch: break
                v1 = itemList1[idx1]
                v2 = itemList2[idx2]
                l1 = min(v1.shape[0],self.nLength)
                l2 = min(v2.shape[0],self.nLength)
                batchVec1[nGood + cnt,:l1] = v1[:l1]
                batchVec2[nGood + cnt,:l2] = v2[:l2]
                batchTruth[nGood + cnt] = 0 # 不正解
            else:
                print "too much loop. cnt=",cnt
                # この場合、学習はゼロベクトルだと不正解とみなすような学習になる。それはそれで良い気がするので、特に追加加工などは行わない

            print "b"
            yield ({"input_1":batchVec1,"input_2":batchVec2},{"distance":batchTruth})

    def buildModel(self):
        def create_base_network(input_shape):
            seq = Sequential()
            seq.add(Bidirectional(GRU(256,activation='tanh', recurrent_activation='hard_sigmoid'),input_shape=input_shape))
            seq.add(Dropout(rate=0.5))
            seq.add(Dense(64,activation="tanh",kernel_initializer="he_normal",use_bias=True,bias_initializer="uniform"))
            seq.add(Lambda(lambda  x: K.l2_normalize(x,axis=-1)))
            return seq

        def siam_distance(vests):
            x, y = vests
            x = K.l2_normalize(x, axis=-1)
            y = K.l2_normalize(y, axis=-1)
            h = K.sum(x*y,axis=-1,keepdims=True) # cosine
            return 0.25 * ((1.-h)**2)

        def siam_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)

        input_shape = (self.nLength,self.nDim)

        inX1 = Input(input_shape,name="input_1")
        inX2 = Input(input_shape,name="input_2")

        base_network = create_base_network(input_shape)

        outX1 = base_network(inX1)
        outX2 = base_network(inX2)

        distance = Lambda(siam_distance, output_shape=siam_dist_output_shape, name="distance")([outX1,outX2])

        model = Model(inputs=[inX1, inX2], outputs=distance)

        optimizer = Adam(self.learnRate)
        model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["binary_accuracy"])

        model.summary()
        self.model = model
        #plot(model, to_file="model.png", show_shapes=True)

        self.outX1 = Model(inputs=inX1,outputs=outX1)
        self.outX2 = Model(inputs=inX2,outputs=outX2)
        return

    def train(self,saveFolder="save"):
        cp_cb = ModelCheckpoint(filepath = saveFolder+"/weights.{epoch:02d}.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto')
        tb_cb = TensorBoard(log_dir=saveFolder, histogram_freq=1)
        while True:
            self.model.fit_generator(generator=self.yieldOne("train"),
                                    validation_data=self.yieldOne("valid"),
                                    validation_steps=10,
                                    epochs=100000000,
                                    callbacks=[cp_cb,tb_cb],
                                    steps_per_epoch=int(self.dAcc.nLines*(1.-self.dAcc.validFrac)/self.nBatch),
                                    use_multiprocessing=True, 
                                    max_queue_size=10, 
                                    workers=1)
            for idx in range(20):
                x = self.dAcc.getColumnPairs_w2v_fixedLength(idx,self.columnPairs,self.nLength)
                w = self.dAcc.getColumnPairs_str            (idx,self.columnPairs)
                y1 = self.outX1.predict({"input_1":np.expand_dims(x[0],axis=0)})
                y2 = self.outX2.predict({"input_2":np.expand_dims(x[1],axis=0)})
                res = self.model.predict({"input_1":np.expand_dims(x[0],axis=0),"input_2":np.expand_dims(x[1],axis=0)})
                print idx
                print w[0]
                #print x[0]
                print w[1]
                #print x[1]
                print y1
                print y2
                print res
                print

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode","-m",dest="mode",type=str,choices=["train"],default="train")
    parser.add_argument("--nBatch" ,"-b",dest="nBatch",type=int,default=256)
    parser.add_argument("--nLength","-l",dest="nLength",type=int,default=20)
    parser.add_argument("--learnRate","-r",dest="learnRate",type=float,default=1e-5)
    parser.add_argument("--saveFolder","-s",dest="saveFolder",type=str,default="save")

    args = parser.parse_args()
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("w2v/wiki.w2v")
    n = net(args,d,columnPairs=(u"Acquiror business description(s)",u"Target business description(s)"),goodFrac=0.5) # columnPairsはtupleにしないとキャッシュのところで落ちるので注意
    n.train(saveFolder="train")
