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
myneuron=argv[2]

myneuron=int(myneuron)

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

# Chen Liu Project
cluster_num=64

# save_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/cluster/clusterL'+mylayer+'/vgg16_'+cat+'_K'+str(cluster_num)+'.pickle'
# file_path = '/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature'+mylayer+'/L'+mylayer+'Feature'+cat
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


all_response=[]
all_h=[]
all_w=[]
all_originalimg=[]


file_path ='/data2/xuyangf/OcclusionProject/NaiveVersion/feature/ChenLiu/'+mylayer+'_rall'
# file_path ='/data2/xuyangf/OcclusionProject/NaiveVersion/feature/ChenLiu/'+mylayer+'_'
fname = file_path+str(0)+'.npz'
ff = np.load(fname)

feat_dim = ff['res'].shape[0]
img_cnt = ff['res'].shape[1]
oldimg_index=0

# number of files to read in
# number of files to read in
file_num = 50
# file_num=10

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

# L2 normalization as preprocessing
feat_norm = np.sqrt(np.sum(feat_set**2, 0))
feat_set = feat_set/feat_norm

hi=loc_set[:,3]
wi=loc_set[:,4]
all_h+=list(hi)
all_w+=list(wi)
all_originalimg+=originimage
all_response+=list(feat_set[myneuron])
natural_image_number=oldimg_index
########################################################################
# file_path ='/data2/xuyangf/OcclusionProject/NaiveVersion/feature/ChenLiu/'+mylayer+'_rall'
file_path ='/data2/xuyangf/OcclusionProject/NaiveVersion/feature/ChenLiu/'+mylayer+'_'
fname = file_path+str(0)+'.npz'
ff = np.load(fname)

feat_dim = ff['res'].shape[0]
img_cnt = ff['res'].shape[1]
oldimg_index=0

# number of files to read in
# number of files to read in
# file_num = 50
file_num=10

maximg_cnt=img_cnt*3

originimage=[]
feat_set = np.zeros((feat_dim, maximg_cnt*file_num))
feat_set[:,0:img_cnt] = ff['res']

originimage+=list(ff['originpath'])
loc_dim = ff['loc_set'].shape[1]
loc_set = np.zeros((maximg_cnt*file_num, loc_dim))
loc_set[0:img_cnt,:] = ff['loc_set']

oldimg_index+=img_cnt

print(originimage[0])
sys.exit(0)

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

# L2 normalization as preprocessing
feat_norm = np.sqrt(np.sum(feat_set**2, 0))
feat_set = feat_set/feat_norm

hi=loc_set[:,3]
wi=loc_set[:,4]
all_h+=list(hi)
all_w+=list(wi)
all_originalimg+=originimage
all_response+=list(feat_set[myneuron])
##########################################################################

all_response=np.array(all_response)
rindexsort=np.argsort(-all_response)
print(rindexsort[:20])

shreshold=all_response[rindexsort[0]]/2
over_half_number=len(np.where(all_response>shreshold)[0])
print('over_half_number')
print(over_half_number)

big_img = np.zeros((patch_size, 5+(patch_size+5)*20, 3))
for i in range(0,20):
    myindex=int(rindexsort[i])
    print(myindex)
    print(all_originalimg[myindex])
    original_img=cv2.imread(all_originalimg[myindex], cv2.IMREAD_UNCHANGED)
    if myindex<natural_image_number:
        oimage,_,__=process_image(original_img,'_',0)
    else:
        oimage,_,__=process_image2(original_img)
    
    ####################################
    oimage+=np.array([104., 117., 124.])
    hindex=int(all_h[myindex])
    windex=int(all_w[myindex])
    big_img[0:patch_size,i*(patch_size+5):i*(patch_size+5)+patch_size,:]=oimage[hindex:hindex+patch_size,windex:windex+patch_size,:]

fname = '/data2/xuyangf/OcclusionProject/NaiveVersion/ChenLiuResult/'+str(myneuron)+'.png'
cv2.imwrite(fname, big_img)

all_response=list(all_response)
all_response.sort(reverse=True)
x=[i for i in range(len(all_response))]

plt.figure()
plt.title('top patch responses')
plt.xlabel("patch")
plt.ylabel("activation")
plt.plot(x,all_response,'r-')
savedir='/data2/xuyangf/OcclusionProject/NaiveVersion/ChenLiuResult/'+str(myneuron)+'toppatch.png'
plt.savefig(savedir) 