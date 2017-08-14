# -*- coding: utf-8 -*-
import argparse,sys,os
from net import net,DataAccessor
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

class net_train(net):
    def __init__(self,args,dAcc,columnPairs,goodFrac=0.5):
        super(net_train,self).__init__(args,dAcc,columnPairs,goodFrac)
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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode","-m",dest="mode",type=str,choices=["train"],default="train")
    parser.add_argument("--nBatch" ,"-b",dest="nBatch",type=int,default=256)
    parser.add_argument("--nLength","-l",dest="nLength",type=int,default=10)
    parser.add_argument("--learnRate","-r",dest="learnRate",type=float,default=1e-4)
    parser.add_argument("--saveFolder","-s",dest="saveFolder",type=str,default="save")

    args = parser.parse_args()
    d = DataAccessor()
    d.loadCSV("all.csv")
    d.loadW2V("w2v/wiki.w2v")
    n = net_train(args,d,columnPairs=(u"Acquiror business description(s)",u"Target business description(s)"),goodFrac=0.5) # columnPairsはtupleにしないとキャッシュのところで落ちるので注意
    n.train(saveFolder="train2")
