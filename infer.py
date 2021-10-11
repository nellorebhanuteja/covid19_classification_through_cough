#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:02:04 2021
@author: Team DiCOVA, IISC, Bangalore
"""
import argparse, configparser
import pickle, random
import numpy as np
from models import *
from sklearn.preprocessing import StandardScaler
import os 
def main(config, modelfil,file_list,featsfil,outfil):
    outdir, _ = os.path.split(outfil)
    # load model
    model = pickle.load(open(modelfil,'rb'))
    pca_reload = pickle.load(open(outdir + '/' + config['default']['pca_model'] + '.pkl','rb'))
    file_list = open(file_list).readlines()
    file_list = [line.strip().split() for line in file_list]
    # 
    feats_list = open(featsfil).readlines()
    feats_list = [line.strip().split() for line in feats_list]
    feats={}
    for fileId,file_path in feats_list:
        feats[fileId] = file_path
    scores={}
    for fileId,_ in file_list:
        F = pickle.load(open(feats[fileId],'rb'))
        F=F.T
        F = StandardScaler().fit_transform(F)
        F = pca_reload.transform(F)
        score = model.validate([F])
        score = np.mean(score[0],axis=0)[1]
        scores[fileId]=score
    with open(outfil,'w') as f:
        for item in scores:
            f.write(item+" "+str(scores[item])+"\n")
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelfil','-m',required=True)
    parser.add_argument('--featsfil','-f',required=True)
    parser.add_argument('--file_list','-i',required=True)
    parser.add_argument('--outfil','-o',required=True)
    parser.add_argument('--config','-c',required=True)
    
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    np.random.seed(int(config['default']['seed']))
    random.seed(int(config['default']['seed']))

    main(config, args.modelfil, args.file_list, args.featsfil, args.outfil)