from __future__ import division
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display
import pickle
import numpy as np
from GetDataPath import *
from sys import argv
import math
from enum import Enum
import sys
from utils import *

mylayer=argv[1]
mycluster=argv[2]


mycluster=int(mycluster)

class featDim_set(Enum):
    pool1=64
    pool2=128
    pool3=256
    pool4=512
    pool5=512
    conv2_1=128
    conv2_2=128
    conv3_1=256
    conv3_2=256
    conv3_3=256

cluster_num = featDim_set[mylayer].value

class Arf_set(Enum):
    pool1=6
    pool2=16
    pool3=44
    pool4=100
    pool5=212
    conv2_1=10
    conv2_2=14
    conv3_1=24
    conv3_2=32
    conv3_3=40
patch_size=Arf_set[mylayer].value

def disttresh(input_index,cluster_center):
    thresh1=0.5
    temp_feat=feat_set[:,input_index]
    error = np.sum((temp_feat.T - cluster_center)**2, 1)
    sort_idx = np.argsort(error)
    return input_index[sort_idx[:int(thresh1*len(sort_idx))]]

def sparsity(x):
    return ( 1 - (np.mean(x,axis=0))**2/np.mean(x**2,axis=0) ) / (1 - 1/len(x))

def cosine_distance(A,B):
    num =np.sum(A*B) 
    denom = np.linalg.norm(A) * np.linalg.norm(B)  
    cos = num / denom 
    sim = 0.5 + 0.5 * cos
    return sim

def nearest_cosine_distance(mycenters,x):
    nearest_sim=0
    for center_feature in mycenters:
        center_feature=np.array(center_feature)
        sim=cosine_distance(center_feature,x)
        if sim>nearest_sim:
            nearest_sim=sim
    return nearest_sim

# Chen Liu Project
cluster_num=64

# save_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL'+mylayer+'/vgg16_'+cat+'_K'+str(cluster_num)+'.pickle'
# file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature'+mylayer+'/L'+mylayer+'Feature'+cat

# file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/ChenLiu/'+mylayer+'_'
# cluster_file='/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/ChenLiu/'+mylayer+'.pickle'

file_path='/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature3/L3Feature0'
cluster_file = '/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL3/dictionary_0.pickle'

fname = file_path+str(0)+'.npz'
ff = np.load(fname)

feat_dim = ff['res'].shape[0]
img_cnt = ff['res'].shape[1]
oldimg_index=0

# number of files to read in
# number of files to read in
file_num = 10
maximg_cnt=img_cnt*3

originimage=[]
feat_set = np.zeros((feat_dim, maximg_cnt*file_num))
feat_set[:,0:img_cnt] = ff['res']

originimage+=list(ff['originpath'])
loc_dim = ff['loc_set'].shape[1]
loc_set = np.zeros((maximg_cnt*file_num, loc_dim))
loc_set[0:img_cnt,:] = ff['loc_set']

oldimg_index+=img_cnt

for ii in range(1,file_num):
    print(ii)
    fname = file_path+str(ii)+'.npz'
    ff = np.load(fname)
    originimage+=list(ff['originpath'])
    img_cnt=ff['res'].shape[1]
    print(img_cnt)
    feat_set[:,oldimg_index:(oldimg_index + img_cnt)] = ff['res']
    loc_set[oldimg_index:(oldimg_index + img_cnt),:] = ff['loc_set']
    #img_set[oldimg_index:(oldimg_index + img_cnt)] = ff['img_set']
    oldimg_index+=img_cnt

feat_set=feat_set[:,:oldimg_index]
#img_set=img_set[:oldimg_index]
loc_set=loc_set[:oldimg_index,:]
originimage=np.array(originimage)


# L2 normalization as preprocessing
feat_norm = np.sqrt(np.sum(feat_set**2, 0))
feat_set = feat_set/feat_norm

with open(cluster_file, 'rb') as fh:
    assignment, centers,example,__= pickle.load(fh)
patch_num=len(assignment)
#############################################################
index = np.where(assignment==mycluster)[0]
index=disttresh(index,centers[mycluster])

# index=index[:20]


temp_feat = feat_set[:, index]
# print('maxneuron')
# print(np.argmax(temp_feat,axis=0))

temp_feat=np.sort(temp_feat,axis=0)
# print(temp_feat[len(temp_feat)-1,:])
activation=np.average(temp_feat,axis=1)
activation=np.flipud(activation)
# rindexsort=np.argsort(-activation)

# print('test0')
# print(len(index))
# print(sparsity(activation))
# print(activation[:4])

big_img = np.zeros((patch_size, 5+(patch_size+5)*20, 3))
for i in range(20):
    myindex=index[i]
    print(myindex)
    original_img=cv2.imread(originimage[myindex], cv2.IMREAD_UNCHANGED)
    oimage,_,__=process_image(original_img,'_',0)
    oimage+=np.array([104., 117., 124.])
    hindex=int(loc_set[myindex,3])
    windex=int(loc_set[myindex,4])
    big_img[0:patch_size,i*(patch_size+5):i*(patch_size+5)+patch_size,:]=oimage[hindex:hindex+patch_size,windex:windex+patch_size,:]
fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/tmpexample/prototype0_'+str(mycluster)+'.png'
cv2.imwrite(fname, big_img)
######################

# ss=44

# big_img = np.zeros((49, 5+(ss+5)*20, 3))

# for i in range(20):
#     cnum=5+i*(44+5)
#     big_img[0:44,cnum:cnum+ss, :] = example[88][:,i].reshape(ss,ss,3)

# fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/ChenLiu/test.png'
# cv2.imwrite(fname, big_img)
################ Feb 14th comparison between reponse of visual concepts and averaged population response to any patches
# Portrait_Images='/data2/haow3/data/flickr/sample/portrait_sample'
# average_any_patch_activation=[]
# for s in os.listdir(Portrait_Images):
#     index=np.where(originimage==os.path.join(Portrait_Images,s))[0]
#     temp_feat = feat_set[:, index]
#     temp_feat=np.sort(temp_feat,axis=0)

#     any_patch_activation=np.average(temp_feat,axis=1)

#     # rindexsort=np.argsort(-any_patch_activation)
#     # print(rindexsort[:4])
#     # print(any_patch_activation[rindexsort[:4]])

#     any_patch_activation=list(any_patch_activation)
#     any_patch_activation.sort(reverse=True)
#     average_any_patch_activation.append(any_patch_activation)

# np_activation=list(np.average(average_any_patch_activation,axis=0))
###################################
# patch_index=[]
# for k in range(0,64):
#     target = centers[k]
#     index=np.where(assignment==k)[0]
#     tempFeat = feat_set[:,index]
#     error = np.sum((tempFeat.T - target)**2, 1)
#     sort_idx = np.argsort(-error)
#     patchindex=index[sort_idx[:20]]
#     patch_index+=list(patchindex)

# temp_feat = feat_set[:, np.array(patch_index)]

# print(len(patch_index))
# temp_feat=np.sort(temp_feat,axis=0)
# np_activation=np.average(temp_feat,axis=1)

# rindexsort=np.argsort(-np_activation)
# print(rindexsort[:4])
# print(np_activation[rindexsort[:4]])

# np_activation=list(np_activation)
# np_activation.sort(reverse=True)

####################################
# index = np.where(assignment==-1)[0]
# nearest_cos=[]
# for i in range(patch_num):
#     nearest_cos.append(nearest_cosine_distance(centers,np.array(feat_set[:,i])))
    
# nearest_cos=np.array(nearest_cos)
# index=np.argsort(nearest_cos)[:200]
# print(nearest_cos[:10])
# print(nearest_cos[index])
# print('mymyindex')
# print(index)

# for k in range(len(centers)):
#     target = centers[k]
#     index=np.where(assignment==k)[0]
#     tempFeat = feat_set[:,index]
#     error = np.sum((tempFeat.T - target)**2, 1)
#     sort_idx = np.argsort(-error)
#     patchindex=index[sort_idx[:5]]
#     patch_index+=list(patchindex)
# index=np.array(patch_index)
index=np.array([3314,60122,19786,58102,60124,19787,60123,60125,60121,32786,91477,60120,27874,44946,22832,39583,72222,30807 ,73489 ,30772])
print(len(centers))
print(len(index))


# print('done')
temp_feat = feat_set[:, index]
# print('maxneuron')
# print(np.argmax(temp_feat,axis=0))
temp_feat=np.sort(temp_feat,axis=0)

# print(temp_feat[len(temp_feat)-1,:])
np_activation=np.average(temp_feat,axis=1)
np_activation=np.flipud(np_activation)
# rindexsort=np.argsort(-np_activation)

###### examples for non prototype images
big_img = np.zeros((patch_size, 5+(patch_size+5)*20, 3))
print('test')
print(len(index))
for i in range(20):
    myindex=index[i]
    print(myindex)
    original_img=cv2.imread(originimage[myindex], cv2.IMREAD_UNCHANGED)
    oimage,_,__=process_image(original_img,'_',0)
    oimage+=np.array([104., 117., 124.])
    hindex=int(loc_set[myindex,3])
    windex=int(loc_set[myindex,4])
    big_img[0:patch_size,i*(patch_size+5):i*(patch_size+5)+patch_size,:]=oimage[hindex:hindex+patch_size,windex:windex+patch_size,:]

fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/tmpexample/new0_'+str(mycluster)+'.png'
cv2.imwrite(fname, big_img)
#

print(sparsity(np_activation))
# print(np_activation[:4])
################################## calculate small sparsity patches
index=np.argsort(sparsity(feat_set))[:20]
print(sparsity(feat_set[:,index]))

big_img = np.zeros((patch_size, 5+(patch_size+5)*20, 3))
print('test1')
for i in range(20):
    myindex=index[i]
    print(myindex)
    original_img=cv2.imread(originimage[myindex], cv2.IMREAD_UNCHANGED)
    oimage,_,__=process_image(original_img,'_',0)
    oimage+=np.array([104., 117., 124.])
    hindex=int(loc_set[myindex,3])
    windex=int(loc_set[myindex,4])
    big_img[0:patch_size,i*(patch_size+5):i*(patch_size+5)+patch_size,:]=oimage[hindex:hindex+patch_size,windex:windex+patch_size,:]
fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/example/tmpexample/big_sparsity_0_'+str(mycluster)+'.png'
cv2.imwrite(fname, big_img)

################################### relationship between sparsity and importance

# img_num=300
# fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/layer3/cat0.npz'
# ff=np.load(fname)
# img_vc=ff['vc_score']
# vc_num=len(img_vc[0])
# #print(img_vc[0])
# img_vc_avg=[]
# for i in range(vc_num):
#     img_vc_avg.append(float(np.sum(img_vc[np.where(img_vc[:,i]!=-1),i]))/img_num)
# img_vc_avg=np.asarray(img_vc_avg)
# rindexsort=np.argsort(-img_vc_avg)

# all_sparsity=[]
# for k in range(vc_num):
#     mycluster=int(rindexsort[k])
#     index = np.where(assignment==mycluster)[0]
#     index=disttresh(index,centers[mycluster])
#     temp_feat = feat_set[:, index]
#     activation=np.average(temp_feat,axis=1)
#     activation=np.sort(activation,axis=0)
#     all_sparsity.append(sparsity(activation))

# # show curve
# x=[i for i in range(0,vc_num)]
# plt.figure()
# plt.title('relationship between sparsity and usefulness')
# plt.xlabel("usefulness (x increases when importance decreases)")
# plt.ylabel("sparsity")
# myplot,=plt.plot(x,all_sparsity,'r-')

# savedir='/data2/xuyangf/OcclusionProject/NaiveVersion/ChenLiuResult/'+'sparsity_usefulness.png'
# plt.savefig(savedir) 
###########################################
x=[i for i in range(0,feat_dim)]
# show curve
plt.figure()
plt.title('Averaged Population Tuning Curves of Visual Concepts')
plt.xlabel("Rank-Ordered Neuronal Index")
plt.ylabel("Averaged Normalized Activation")
plot1,=plt.plot(x,activation,'r-')
plot2,=plt.plot(x,np_activation,'b-')

plt.legend([plot1,plot2],['Prototype Bird Beak Concept','non prototype Image Patches'],loc='best')

savedir='/data2/xuyangf/OcclusionProject/NaiveVersion/ChenLiuResult/'+'topneuron5.png'
plt.savefig(savedir) 