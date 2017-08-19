#coding:utf-8
#########################################################################
# File Name: train.py
# Author:Lei Jiang
# mail: jianglei@1000look.com
# Created Time: 2016年03月28日 星期一 09时21分43秒
# Copyright Nanjing Qing So information technology
#########################################################################

from params import Params
pa=Params()
caffe_root=pa.caffe_root
import sys
sys.path.insert(0,caffe_root)
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import os.path as osp
import numpy as np


class SolverWrapper(object):
    def __init__(self):
        if pa.GPU==True:
            caffe.set_device(pa.device)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.solver=caffe.SGDSolver(pa.solver)
        if pa.pretrain!="":
            self.solver.net.copy_from(pa.pretrain)
        self.solver_param=caffe_pb2.SolverParameter()
        with open(pa.solver,'rt') as f:
            pb2.text_format.Merge(f.read(),self.solver_param)

        #self.output_dir=pa.output_dir
        self.solver.net.layers[0].set_queue()
    
    def snap_shot(self):
        net=self.solver.net
        filename=(self.solver_param.snapshot_prefix+'_iter_{:d}'.format(self.solver.iter)+'.caffemodel')
        #filename=osp.join(self.output_dir,filename)
        net.save(str(filename))
        print "Wrote snapshot to {:s}".format(filename)

    def train_mode(self):
        '''Network train looping...'''
        print "------------training start---------------"
        last_snapshot_iter=-1
        while self.solver.iter<self.solver_param.max_iter:
            self.solver.step(1) #update once
            if self.solver.iter % self.solver_param.snapshot ==0:
                last_snapshot_iter=self.solver.iter
                self.snap_shot()
            if self.solver.iter % self.solver_param.test_interval ==0:
                print "------------Iteration testing----------------"
                self.solver.test_nets[0].share_with(self.solver.net)
                self.calculate_accuracy()
        if last_snapshot_iter!=self.solver.iter:
            self.snap_shot()

    def calculate_accuracy(self):
        print "------------------calculate_accuracy---------------"
        accuracy_nums=len(pa.accuracy_layers_dic)
        input_labels=pa.top_names[1:]
        correct=np.zeros(accuracy_nums)
        for i in range(self.solver_param.test_iter[0]):
            self.solver.test_nets[0].forward()
            for j,ac_l,inp_l in zip(range(accuracy_nums),pa.accuracy_layers_dic,input_labels):
                prob=self.solver.test_nets[0].blobs[ac_l].data.argmax(1)
                gt=np.squeeze(self.solver.test_nets[0].blobs[inp_l].data)
                correct[j]+=np.sum(prob==gt)
                
        for i in range(accuracy_nums):
            out=correct[i]/(pa.batchsize_dic['val']*self.solver_param.test_iter[0])
            print "Test net output%d#accuracy=%f"%(i,out)



        




        

