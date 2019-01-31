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

def disttresh(input_index,cluster_center):
    thresh1=0.5
    temp_feat=feat_set[:,input_index]
    error = np.sum((temp_feat.T - cluster_center)**2, 1)
    sort_idx = np.argsort(error)
    return input_index[sort_idx[:int(thresh1*len(sort_idx))]]

def sparsity(x):
	return ( 1 - (np.mean(x,axis=0))**2/np.mean(x**2,axis=0) ) / (1 - 1/len(x))

final_sparsity=[]

for cat in range(0,100):
	if cat==39:
		continue

	file_path='/data2/xuyangf/OcclusionProject/NaiveVersion/feature/feature3/L3Feature'+str(cat)
	cluster_file = '/data2/xuyangf/OcclusionProject/NaiveVersion/prunning/prunL3/dictionary_'+str(cat)+'.pickle'

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

	img_num=300
	fname ='/data2/xuyangf/OcclusionProject/NaiveVersion/vc_score/layer3/cat'+str(cat)+'.npz'
	ff=np.load(fname)
	img_vc=ff['vc_score']
	vc_num=len(img_vc[0])

	if vc_num<100:
		continue
	#print(img_vc[0])
	img_vc_avg=[]
	for i in range(vc_num):
	    img_vc_avg.append(float(np.sum(img_vc[np.where(img_vc[:,i]!=-1),i]))/img_num)
	img_vc_avg=np.asarray(img_vc_avg)
	rindexsort=np.argsort(-img_vc_avg)

	all_sparsity=[]
	for k in range(vc_num):
	    mycluster=int(rindexsort[k])
	    index = np.where(assignment==mycluster)[0]
	    index=disttresh(index,centers[mycluster])
	    temp_feat = feat_set[:, index]
	    activation=np.average(temp_feat,axis=1)
	    activation=np.sort(activation,axis=0)
	    all_sparsity.append(sparsity(activation))
	final_sparsity.append(all_sparsity[:100])

final_sparsity=np.array(final_sparsity)
print(np.shape(final_sparsity))
final_sparsity=list(np.mean(np.array(final_sparsity),axis=0))

x=[i for i in range(0,100)]
plt.figure()
plt.title('relationship between sparsity and usefulness of top 100 visual concepts')
plt.xlabel("usefulness (x increases when importance decreases)")
plt.ylabel("sparsity")

print(x)
print(final_sparsity)
myplot,=plt.plot(x,final_sparsity,'r-')

savedir='/data2/xuyangf/OcclusionProject/NaiveVersion/ChenLiuResult/'+'all_class_sparsity_usefulness.png'
plt.savefig(savedir) 