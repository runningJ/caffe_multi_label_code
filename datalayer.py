#coding:utf-8
#########################################################################
# File Name: datalayer.py
# Author:Lei Jiang
# mail:leijiang@163.com
# Created Time: 2016年03月23日 星期三 14时21分20秒
# Copyright Nanjing Qing So information technology
#########################################################################

import sys
from params import Params
par=Params()
sys.path.insert(0,par.caffe_root)
import caffe
import os.path as osp
import exceptions
import numpy as np
from multiprocessing import Process,Queue
from minibatch import Batchloader

class BlobFetcher(Process):
    '''using multi-process prepare blobs'''
    def __init__(self,imagepath,annotion,queue,phase):
        super(BlobFetcher,self).__init__()
        self.batch=Batchloader(imagepath,annotion)
        self.queue=queue
        self.phase=phase
    def run(self):
        print "---------------BlobFetcher start------------------------"
        while True:
            blobs=self.batch.get_minibatch(self.phase)
            self.queue.put(blobs)




class multilabel(caffe.Layer):
    '''
    This is a simple datalayer for train multilabel model on clothes datasets
    '''
    def setup(self,bottom,top):
        print "my datalayer is inilializing...."
        label_numbers=len(par.top_names)
        layer_params=eval(self.param_str)
        self.phase=layer_params['phase']
        top[0].reshape(par.batchsize_dic[self.phase],par.channel,par.height,par.width)
        for i in range(1,label_numbers):
            top[i].reshape(par.batchsize_dic[self.phase],1)
        self.batchloader=Batchloader(par.data_label_dic[self.phase][0],par.data_label_dic[self.phase][1])
        print "---------------------inilialized %s----------------"%(self.phase,)

    def set_queue(self):
        self.blob_queue=Queue(par.queue_num)
        self.prefetch_process=BlobFetcher(par.data_label_dic[self.phase][0],par.data_label_dic[self.phase][1],self.blob_queue,self.phase)
        self.prefetch_process.start()
        def clean():
            print "Terminating BlobFetcher"
            self.prefetch_process.terminate()
            self.prefetch_process.join()
        import atexit
        atexit.register(clean)


    def reshape(self,bottom,top):
        '''do nothing'''
        pass
    
    def get_batch(self):
        '''interface control if use multi-process'''
        if par.Using_Process_dic[self.phase]:
            return self.blob_queue.get()
        else:
            return self.batchloader.get_minibatch(self.phase)


    def forward(self,bottom,top):
        blob=self.get_batch()#blob=[[image,[color,long,sleeve]],[image,[color,long,sleeve]],....[]]
        num_labels=len(blob[0][1])
        for i in xrange(par.batchsize_dic[self.phase]):
            top[0].data[i,...]=blob[i][0] #data
            for j in range(num_labels):
                top[j+1].data[i,...]=int(blob[i][1][j]) #label


    def backward(self,bottom,top):
        '''do nothing'''
        pass
    

