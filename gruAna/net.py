# -*- coding: utf-8 -*-
import os,sys,argparse,re,string,logging,random,csv
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
        self.nW2VDim = 100
        self.puncReStr = string.punctuation
        self.puncReStr = "".join([x for x in self.puncReStr if x not in ["-","'"]])
        self.puncReStr = r"[%s]"%self.puncReStr
        self.verbose = verbose
        return

    def buildW2Vfile(self,filePath,outPath,min_count=1000,window=5):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = word2vec.LineSentence(filePath)
        model = word2vec.Word2Vec(sentences, size=self.nW2VDim,min_count=min_count,window=window)
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
        ret = self.str_to_w2v(strPairs[0]),self.str_to_w2v(strPairs[1])
        return ret

    def getColumnPairs_w2v_fixedLength(self,index,colPairs,fixedLength):
        strPairs = self.getColumnPairs_str(index,colPairs)
        strPairs = [self.cleanUpStr(x) for x in strPairs]
        #strPairs = strPairs[-fixedLength:]
        return self.str_to_w2v(strPairs[0],fixedLength=fixedLength),self.str_to_w2v(strPairs[1],fixedLength=fixedLength)

    def cleanUpStr(self,inStr):
        inStr = inStr.replace("\r\n","").replace("\n","")
        inStr = re.sub(self.puncReStr," ",inStr)
        inStr = re.sub("\s+"," ",inStr).strip().lower()
        return inStr

    def str_to_w2v(self,inStr,fixedLength=None):
        inStr = inStr.split()
        if fixedLength:  res = np.zeros( (fixedLength, self.nW2VDim), dtype=np.float32)
        else:            res = np.zeros( (len(inStr) , self.nW2VDim), dtype=np.float32)
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
    def __init__(self,dAcc,columnPairs,goodFrac=0.5,nBatch=256,nGRU=512,nLength=20,learnRate=1e-4,saveFolder="save",asymDistance=False):
        self.dAcc    = dAcc
        self.nW2VDim = dAcc.nW2VDim
        self.nBatch  = nBatch
        self.nGRU    = nGRU
        self.nLength = nLength
        self.columnPairs = columnPairs
        self.goodFrac = goodFrac
        self.learnRate = learnRate
        self.saveFolder = saveFolder
        self.asymDistance = asymDistance
        self.buildModel()
        return

    def yieldOne(self,mode="all"):
        nGood = int(self.nBatch * self.goodFrac)
        nBad  = self.nBatch - nGood

        targetLines = self.dAcc.getTargetLines(mode)
        if mode=="train": self.trainLines = targetLines
        if mode=="valid": self.validLines = targetLines

        while True:
            batchVec1  = np.zeros( (self.nBatch, self.nLength, self.nW2VDim), dtype=np.float32)
            batchVec2  = np.zeros( (self.nBatch, self.nLength, self.nW2VDim), dtype=np.float32)
            batchTruth = np.zeros( (self.nBatch, 1)        , dtype=np.int32)

            itemList1 = []
            itemList2 = []
            wordLists = []

            # まずは正解を入れ込む
            cnt = 0
            for i in random.sample(targetLines,nGood):
                try:
                    v1,v2 = self.dAcc.getColumnPairs_w2v(i,self.columnPairs)
                    w1,w2 = self.dAcc.getColumnPairs_str(i,self.columnPairs)
                    itemList1.append(v1)
                    itemList2.append(v2)
                    wordLists.append((w1,w2))
                    l1 = min(v1.shape[0],self.nLength)
                    l2 = min(v2.shape[0],self.nLength)
                    batchVec1[cnt,:l1,:] = v1[:l1]
                    batchVec2[cnt,:l2,:] = v2[:l2]
                    batchTruth[cnt] = 1 # 正解
                    cnt += 1
                except:
                    #print "error occured"
                    continue

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
            #print "batchTruth=",batchTruth
            #print "batchVec1=",batchVec1
            #print "batchVec2=",batchVec2

            yield ({"input_1":batchVec1,"input_2":batchVec2},{"distance":batchTruth})

    def reloadModel(self,fPath):
        self.model.load_weights(fPath)
        print "model loaded from %s"%fPath
        return

    def buildModel(self):
        def create_base_network(input_shape):
            seq = Sequential()
            seq.add(Bidirectional(GRU(self.nGRU,activation='tanh', recurrent_activation='hard_sigmoid'),input_shape=input_shape))
            seq.add(Dropout(rate=0.5))
            #seq.add(Dense(64,activation=None,kernel_initializer="he_normal",use_bias=True,bias_initializer="uniform",kernel_regularizer=l2(1e-6),bias_regularizer=l2(1e-6)))
            seq.add(Dense(64,activation="tanh",kernel_initializer="he_normal",use_bias=True,bias_initializer="uniform"))
            seq.add(Lambda(lambda  x: K.l2_normalize(x,axis=-1)))
            return seq

        def siam_distance(vests):
            x, y = vests
            x = K.l2_normalize(x, axis=-1)
            y = K.l2_normalize(y, axis=-1)
            h = K.sum(x*y,axis=-1,keepdims=True) # cosine
            return 0.25 * ((1.-h)**2)

        def eucl_distance(vects):
            x, y = vects
            return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

        def eucl_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)

        def siam_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)


        def contrastive_loss(y_true, y_pred):
            '''Contrastive loss from Hadsell-et-al.'06
            http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
            '''
            margin = 1
            return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

        def compute_accuracy(y_true, y_pred):
            '''Compute classification accuracy with a fixed threshold on distances.
            '''
            thres = 0.5
            return K.mean(K.equal(K.greater(y_true,0.5),K.less(y_pred,thres)))

        input_shape = (self.nLength,self.nW2VDim)

        inX1 = Input(input_shape,name="input_1")
        inX2 = Input(input_shape,name="input_2")

        base_network = create_base_network(input_shape)

        outX1 = base_network(inX1)
        outX2 = base_network(inX2)

        if self.asymDistance:
            outX2 = Dense(64,activation=None)(outX2) # ここがv3.0での唯一の変更点

        distance = Lambda(eucl_distance, output_shape=eucl_dist_output_shape, name="distance")([outX1,outX2])

        model = Model(inputs=[inX1, inX2], outputs=distance)

        optimizer = Adam(self.learnRate)
        #optimizer = RMSprop(self.learnRate)
        #model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["binary_accuracy"])
        model.compile(loss=contrastive_loss,optimizer=optimizer,metrics=[compute_accuracy])

        model.summary()
        self.model = model

        self.outX1 = Model(inputs=inX1,outputs=outX1)
        self.outX2 = Model(inputs=inX2,outputs=outX2)
        return

    def train(self):
        cp_cb = ModelCheckpoint(filepath = self.saveFolder+"/weights.{epoch:02d}.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto')
        tb_cb = TensorBoard(log_dir=self.saveFolder, histogram_freq=1)
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

    def testOne(self,txt):
        vec = self.dAcc.str_to_w2v(txt,fixedLength=self.nLength)
        y1 = self.outX1.predict({"input_1":np.expand_dims(vec,axis=0)}) # こちらはoutX1を使用
        return y1[0]

    def testAll(self,y1):
        pos = 0
        score = []
        words = []
        #index = []

        while pos<self.dAcc.nLines:
            vList = []
            num_processed = 0
            for i in range(self.nBatch):
                if pos+i>=self.dAcc.nLines:
                    vList.append(v) # このwhileループがスタートしているということは、vは少なくとも1回は呼ばれているはず。それをダミーとして入れ込む
                    continue
                w = self.dAcc.getColumnPairs_str(pos+i,self.columnPairs)
                w = w[1] # 一応、買われた側の値を使う
                v = self.dAcc.str_to_w2v(w,fixedLength=self.nLength)
                words.append(w)
                vList.append(v)
                #index.append(i+pos)
                num_processed += 1
            vList = np.array(vList)
            y2 = self.outX2.predict({"input_2":vList}) # y2側は、outX2で計算すること
            #sc = np.dot(y2,y1) 
            sc = np.linalg.norm(y2-y1,axis=1)
            for i in range(num_processed):
                score.append(sc[i])
            pos += num_processed
        return score,words

    def testByIndex(self,idx,nMax=10):
        txt  = self.dAcc.d[self.columnPairs[0]][idx]
        name = self.dAcc.d["Acquiror name"][idx]
        ret,_ = self.test(txt,nMax=nMax,verbose=False)
        header = (idx,name,txt)
        print
        print
        print "match for :\t",name
        print txt
        print "-----"
        for val in ret:
            cnt, score, targetName,words = val
            print cnt,"\t",score,"\t",targetName,":\t\t",words
        print
        print
        return ret,header

    def test(self,txt,nMax=10,verbose=True):
        # txtに基づいて相性の良いindexを5件選定
        y1 = self.testOne(txt) # まず、入力されたものを変換
        score, words = self.testAll(y1)

        if verbose:
            print
            print "match for :",txt
        header = txt
        cnt = 0
        goodOnes = []
        ret = []
        for idx in np.argsort(score)[::+1]:
            if words[idx] in goodOnes: continue
            if verbose:
                print cnt+1,"\t",score[idx],"\t",self.dAcc.d["Target name"][idx],":\t\t",words[idx]
            else:
                ret.append( (cnt+1,score[idx],self.dAcc.d["Target name"][idx],words[idx]) )
            goodOnes.append(words[idx])
            cnt += 1
            if cnt>=nMax: break
        return ret,header

def train_v3_0():
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("w2v/wiki_mincount100.w2v")
    n = net(d,columnPairs=(u"Acquiror business description(s)",u"Target business description(s)"),
            goodFrac=0.5,
            nBatch=256,
            nGRU=512,
            nLength=5,
            asymDistance = True,
            saveFolder="models/v3.0_BusinessDescription_minCount100_nLength5_nGRU512") # columnPairsはtupleにしないとキャッシュのところで落ちるので注意
    n.train()

def test_v3_0(ver):
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("w2v/wiki_mincount100.w2v")
    n = net(d,columnPairs=(u"Acquiror business description(s)",u"Target business description(s)"),
            goodFrac=0.5,
            nBatch=256,
            nGRU=512,
            nLength=5,
            asymDistance = True)
    n.reloadModel("models/v3.0_BusinessDescription_minCount100_nLength5_nGRU512/weights.%d.hdf5"%ver)

    for i in (1,10,50,100,300,500,700,1000,3000,5000,7000,9000,10000,30000,50000,70000):
        n.testByIndex(i)


    #print
    #n.test("Chemicals distributor, Rubber and latex products distributor")
    #n.test("Biopharmaceuticals developer, Biopharmaceuticals manufacturer, Consumer healthcare products manufacturer, Infant food manufacturer, Pharmaceutical products manufacturer")
    #n.test("Food condiments and sauces manufacturer holding company, Soft and spreadable cheeses manufacturer holding company, Soft drinks manufacturer holding company")
    #n.test("Broadband telecommunications services holding company, Cable television services holding company, High-speed data and voice services holding company, Wireless backhaul services holding company")
    #n.test("Pharmaceutical solutions research and development services, Pharmaceuticals manufacturer")

def train_v3_1():
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("w2v/wiki_mincount100.w2v")
    n = net(d,columnPairs=(u"Acquiror business description(s)",u"Target business description(s)"),
            goodFrac=0.5,
            nBatch=256,
            nGRU=512,
            nLength=20,
            asymDistance = True,
            saveFolder="models/v3.1_BusinessDescription_minCount100_nLength20_nGRU512") # columnPairsはtupleにしないとキャッシュのところで落ちるので注意
    n.train()

def test_v3_1(ver):
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("w2v/wiki_mincount100.w2v")
    n = net(d,columnPairs=(u"Acquiror business description(s)",u"Target business description(s)"),
            goodFrac=0.5,
            nBatch=256,
            nGRU=512,
            nLength=20,
            asymDistance = True)
    n.reloadModel("models/v3.1_BusinessDescription_minCount100_nLength20_nGRU512/weights.%d.hdf5"%ver)

    res = []
    for i in (1,10,50,100,300,500,700,1000,3000,5000,7000,9000,10000,30000,50000,70000):
        res.append(n.testByIndex(i))

    for i,rr in enumerate(res):
        ret,header = rr
        idx,AcqName,words1 = header
        with open("log/%d.csv"%idx,"w") as f:
            c = csv.writer(f)
            c.writerow(["","",AcqName,words1])
            for cnt, score, targetName, words2 in ret:
                c.writerow([cnt,score,targetName,words2])

    return

    n.test("Rubber and latex products distributor")
    n.test("Consumer healthcare products manufacturer")
    n.test("Soft drinks manufacturer holding company")
    n.test("High-speed data and voice services holding company, Wireless internet connection services holding company")
    n.test("Pharmaceutical solutions research and development services")

def train_v3_2(): # 理由不明だが、何故か学習が進まない. Learn rateやOptimizerをいじってもダメだった
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("w2v/wiki_mincount100.w2v")
    n = net(d,columnPairs=(u"Acquiror overview",u"Target overview"),
            goodFrac=0.5,
            nBatch=256,
            nGRU=512, # 2048にしてもダメ
            nLength=50, # 20や200にしてもダメ
            asymDistance = True, # Falseにしてもダメ
            saveFolder="models/v3.2_Overview_minCount100_nLength50_nGRU512") # columnPairsはtupleにしないとキャッシュのところで落ちるので注意
    n.train()

def test_v3_2(ver):
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("w2v/wiki_mincount100.w2v")
    n = net(d,columnPairs=(u"Acquiror overview",u"Target overview"),
            goodFrac=0.5,
            nBatch=256,
            nGRU=512,
            nLength=50,
            asymDistance = True)
    n.reloadModel("models/v3.2_Overview_minCount100_nLength50_nGRU512/weights.%d.hdf5"%ver)

    res = []
    for i in (1,10,50,100,300,500,700,1000,3000,5000,7000,9000,10000,30000,50000,70000):
        res.append(n.testByIndex(i))

    for i,rr in enumerate(res):
        ret,header = rr
        idx,AcqName,words1 = header
        with open("log/%d.csv"%idx,"w") as f:
            c = csv.writer(f)
            c.writerow(["","",AcqName,words1])
            for cnt, score, targetName, words2 in ret:
                c.writerow([cnt,score,targetName,words2])

    return

def train_v3_3(): 
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("w2v/wiki_mincount100.w2v")
    n = net(d,columnPairs=(u"Acquiror name",u"Target name"),
            goodFrac=0.5,
            nBatch=256,
            nGRU=512, 
            nLength=10,
            asymDistance = True,
            saveFolder="models/v3.3_Name_minCount100_nLength10_nGRU512") # columnPairsはtupleにしないとキャッシュのところで落ちるので注意
    n.train()

def test_v3_3(ver):
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("w2v/wiki_mincount100.w2v")
    n = net(d,columnPairs=(u"Acquiror name",u"Target name"),
            goodFrac=0.5,
            nBatch=256,
            nGRU=512,
            nLength=10,
            asymDistance = True)
    n.reloadModel("models/v3.3_Name_minCount100_nLength10_nGRU512/weights.%d.hdf5"%ver)

    res = []
    for i in (1,10,50,100,300,500,700,1000,3000,5000,7000,9000,10000,30000,50000,70000):
        res.append(n.testByIndex(i))

    return

def train_v3_4(): 
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("w2v/wiki_mincount100.w2v")
    n = net(d,columnPairs=(u"",u"Target name"),
            goodFrac=0.5,
            nBatch=256,
            nGRU=512, 
            nLength=10,
            asymDistance = True,
            saveFolder="models/v3.3_Name_minCount100_nLength10_nGRU512") # columnPairsはtupleにしないとキャッシュのところで落ちるので注意
    n.train()

def test_v3_4(ver):
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("w2v/wiki_mincount100.w2v")
    n = net(d,columnPairs=(u"Acquiror name",u"Target name"),
            goodFrac=0.5,
            nBatch=256,
            nGRU=512,
            nLength=10,
            asymDistance = True)
    n.reloadModel("models/v3.3_Name_minCount100_nLength10_nGRU512/weights.%d.hdf5"%ver)

    res = []
    for i in (1,10,50,100,300,500,700,1000,3000,5000,7000,9000,10000,30000,50000,70000):
        res.append(n.testByIndex(i))

    return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode" ,"-m",dest="mode",type=str,default="dummy")
    parser.add_argument("--ver"  ,"-v",dest="ver" ,type=int,default=0)
    args = parser.parse_args()

    if   args.mode=="train_v3.0": train_v3_0()
    elif args.mode=="test_v3.0" : test_v3_0(args.ver)
    elif args.mode=="train_v3.1": train_v3_1()
    elif args.mode=="test_v3.1" : test_v3_1(args.ver)
    elif args.mode=="train_v3.2": train_v3_2()
    elif args.mode=="test_v3.2" : test_v3_2(args.ver)
    elif args.mode=="train_v3.3": train_v3_3()
    elif args.mode=="test_v3.3" : test_v3_3(args.ver)
