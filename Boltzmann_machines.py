# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 18:00:28 2019

@author: YASH SAINI
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import pandas as pd

"""Data Preprocessing"""

movies=pd.read_csv("movies.dat",sep="::",header=None, engine="python", encoding="latin-1")
users=pd.read_csv("users.dat",sep="::",header=None, engine="python", encoding="latin-1")
ratings=pd.read_csv("ratings.dat",sep="::",header=None, engine="python", encoding="latin-1")
 
#Training and test set
training_set=pd.read_csv("u1.base",delimiter="\t")
training_set=np.array(training_set, dtype='int')
test_set=pd.read_csv("u1.test",delimiter="\t")
test_set=np.array(test_set, dtype='int')

#number of users and movies
nb_users=int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies=int(max(max(training_set[:,1]),max(test_set[:,1])))

#movies in columns, each user in row and rating to the corresponding movie in the table
def convert(data):
    new_dat=[]
    for i in range(1,nb_users+1):
        movies_id=data[:,1][data[:,0]==i]
        ratings_id=data[:,2][data[:,0]==i]
        ratings_list=np.zeros(nb_movies)
        ratings_list[movies_id-1]=ratings_id
        new_dat.append(list(ratings_list))
    return new_dat
training_set=convert(training_set)
test_set=convert(test_set)

# to torch tensors
training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)

""" Boltzman machine"""

#convert ratings to liked(1) or not liked(0)
training_set[training_set==0]=-1
training_set[training_set==1]=0
training_set[training_set==2]=0
training_set[training_set>=3]=1

test_set[test_set==0]=-1
test_set[test_set==1]=0
test_set[test_set==2]=0
test_set[test_set>=3]=1


class RBM():
    def __init__(self,nh,nv):
        self.W=torch.randn(nh,nv)
        self.a=torch.randn(1,nh) # bias for hidden node
        self.b=torch.randn(1,nv)# bias for visible nodes
    def sample_h(self,x):
        # x represents visible node
        wx=torch.mm(x,self.W.t())
        activation= wx+ self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v , torch.bernoulli(p_h_given_v)
    def sample_v(self,y):
        # y represent hidden nodes
        wy=torch.mm(y,self.W)
        activation= wy+ self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h , torch.bernoulli(p_v_given_h)
    # contrastive divergence
    def train(self,v0,vk,ph0,phk):
        self.W += torch.mm(v0.t(),ph0)-torch.mm(vk.t(),phk)
        self.b+=torch.sum((v0-vk),0)
        self.a+=torch.sum((ph0-phk),0)

nv=len(training_set[0])
nh=100 # no. of features to detect
batch_size=100
rbm=RBM(nh,nv)    

#Training the model

nb_epoch=10
for epoch in range(1,nb_epoch+1):
    train_loss=0
    s=0.
    for id_user in range(0,nb_users-batch_size,batch_size):
        vk=training_set[id_user:id_user+batch_size]#visible node at kth step
        v0=training_set[id_user:id_user+batch_size]
        ph0,_=rbm.sample_h(v0)
        for k in range(10):
            _,hk=rbm.sample_h(vk)
            _,vk=rbm.sample_v(hk)
            vk[v0<0]=v0[v0<0]
        phk,_=rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss+= torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s+=1.
    print ("epoch: "+str(epoch)+" loss: "+str(train_loss/s))            

#Testing

test_loss=0
s=0.
for id_user in range(0,nb_users):
    v=training_set[id_user:id_user+1]#visible node at kth step
    vt=test_set[id_user:id_user+1]
    if len(vt[vt>=0])>0:
        _,h=rbm.sample_h(v)
        _,v=rbm.sample_v(h)
        
        test_loss+= torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
        s+=1.
print ( "test loss: "+str(test_loss/s))



