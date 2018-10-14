import cv2
import os
import math
from sklearn.cluster import MiniBatchKMeans
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import fix
from CLASSES.CLASSES import Pic
import matlab.engine
import scipy.io as sio
import operator as op
import time

numclusters = 20000
numiters = 50

d=64
d_b=32

# class Pic():
#     def __init__(self, des, x, y):
#         self.des = des
#         self.x = x
#         self.y = y
#
#     def set_hamming_code(self, code):
#         self.hamming_code = code

def SURF_exa(dir, file, dest):
    hessianThreshold = 3000
    img = cv2.imread(dir + os.sep + file, cv2.IMREAD_COLOR)
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
    kp, des = surf.detectAndCompute(img, None)
    s = list()
    # if len(des) > 2000:
    #     while len(des) > 2000:
    #         hessianThreshold += 500
    #         surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
    #         kp, des = surf.detectAndCompute(img, None)
    # else:
    #     while len(des) < 2000 and hessianThreshold > 0:
    #         hessianThreshold -= 100
    #         surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
    #         kp, des = surf.detectAndCompute(img, None)
    for i in range(len(des)):
        s.append(Pic(des[i], kp[i].pt[1], kp[i].pt[0]))
    file = open(dest + os.sep + os.path.splitext(file)[0] + '.pkl', 'wb')
    pickle.dump(s, file)
    file.close()



def getSURFS(dir, dest):
    imgs = os.listdir(dir)
    # for img in imgs:
    #     SURF_exa(dir, img, dest)
    with ThreadPoolExecutor(15) as executor:
        for img in imgs:
            SURF_exa(dir, img, dest)
            executor.submit(SURF_exa, dir, img, dest)


if __name__ == '__main__':
    dir = 'resource\\gongsi1000'
    dest = 'resource\\gongsi1000SURF'
    segment_path = 'resource\\segment'
    sample_path = 'data\\sample_with_kp.txt'
    sample_pkl_path = 'data\\sample_with_kp.pkl'
    trainmodel_path = 'data\\trainmodel.mat'
    SURF_path = 'resource\\gongsi1000SURF'
    SURF_txt_path = 'resource\\gongsi1000SURFtxt'
    sparse_txt_path = 'resource\\gongsi1000_sparse_txt'
    sparse_data_path = 'data\\sparse.txt'
    kmeans_data_path = 'data\\minibatchkmeans.pkl'
    cluster_data_path = 'data\\cluster.pkl'
    proj_data_path = 'data\\proj.pkl'
    configFile = 'config\\config.txt'
    # get SURF and perform sparse coding

    # getSURFS(dir, dest)

    # extract SURF data to one file to train sparse data

    # fix.fix(SURF_path, SURF_txt_path, sample_path, 'SURF')
    # Hamming part

    #read P
    # if os.path.isfile('data\\projection.npy'):
    #     P = np.load('data\\projection.npy')
    # else:
    #     M = np.random.randn(d, d)
    #     Q, R = np.linalg.qr(M)
    #     P = Q[:d_b, :]
    #     np.save('data\\projection.npy', P)
    # SURF = np.loadtxt(sample_path)
    # proj = np.matmul(P, np.transpose(SURF))
    # proj = np.transpose(proj)
    # file = open(proj_data_path, 'wb')
    # pickle.dump(proj, file)
    # file.close()

    # Perform kmeans on proj
    # kmeans = MiniBatchKMeans(n_clusters=1000).fit(SURF)
    # file = open(kmeans_data_path, 'wb')
    # pickle.dump(kmeans, file)
    # file.close()

    # compute clusters' median vector
    # file = open(proj_data_path, 'rb')
    # proj = pickle.load(file)
    # file.close()
    # file = open(kmeans_data_path, 'rb')
    # kmeans = pickle.load(file)
    # file.close()
    # cluster = dict()
    # for path in os.listdir(dest):
    #     file = open(dest + os.sep + path, 'rb')
    #     Pics = pickle.load(file)
    #     file.close()
    #     des = list()
    #     for line in Pics:
    #         des.append(line.des)
    #     labels = kmeans.predict(des)
    #     for i in range(len(Pics)):
    #         if labels[i] not in cluster.keys():
    #             cluster[labels[i]] = list()
    #         cluster[labels[i]].append(des[i])

    # compute hamming code and save it to Pics

    # for key, value in cluster.items():
    #     cluster[key] = np.mean(value, axis=0)
    # file = open(cluster_data_path, 'wb')
    # pickle.dump(cluster, file)
    # file.close()

    # file = open(cluster_data_path, 'rb')
    # cluster = pickle.load(file)
    # file.close()
    # file = open(kmeans_data_path, 'rb')
    # kmeans = pickle.load(file)
    # file.close()
    # for path in os.listdir(dest):
    #     file = open(dest + os.sep + path, 'rb')
    #     Pics = pickle.load(file)
    #     file.close()
    #     des = list()
    #     for line in Pics:
    #         des.append(line.des.tolist())
    #     labels = kmeans.predict(des)
    #     for i in range(len(Pics)):
    #         tmp = (Pics[i].des > cluster[Pics[i].cluster]).astype(int)
    #         Pics[i].set_hamming_code(tmp)
    #     file = open(dest + os.sep + path, 'wb')
    #     pickle.dump(Pics, file)
    #     file.close()

    # on-line of hamming

    query_dir = 'resource\\query'
    query_dest = 'resource\\query_SURF'
    query_SURF_txt_path = 'resource\\query_SURF_txt'
    query_sample_path = 'data\\query_sample_with_kp.txt'
    query_configFile = 'config\\query_config.txt'
    query_segment_path = 'resource\\query_segment'
    query_proj_data_path = 'data\\query_proj.pkl'
    query_sample_pkl_path = 'data\\query_sample_with_kp.pkl'

    start = time.time()
    getSURFS(query_dir, query_dest)

    # extract SURF data to one file to train sparse data

    fix.fix(query_dest, query_SURF_txt_path, query_sample_path, 'SURF')
    # eng = matlab.engine.start_matlab()
    # eng.segmentFeature_train(query_configFile, query_SURF_txt_path, query_segment_path, trainmodel_path, nargout=0)
    # eng.quit()
    # SURF_to_sparse(query_segment_path, query_dest)
    # files = os.listdir(query_SURF_txt_path)
    # l = list()
    # for path in files:
    #     file = open(query_SURF_txt_path + os.sep + path, 'r')
    #     for line in file.readlines():
    #         l.append(list(map(eval, line.split())))
    #     file.close()
    # file = open(query_sample_pkl_path, 'wb')
    # pickle.dump(l, file)
    # file.close()

    P = np.load('data\\projection.npy')
    query_SURF = np.loadtxt(query_sample_path)
    proj = np.matmul(P, np.transpose(query_SURF))
    proj = np.transpose(proj)
    file = open(query_proj_data_path, 'wb')
    pickle.dump(proj, file)
    file.close()
    #
    file = open(kmeans_data_path, 'rb')
    kmeans = pickle.load(file)
    file.close()
    labels = kmeans.predict(query_SURF)
    file = open(cluster_data_path, 'rb')
    cluster = pickle.load(file)
    file.close()
    path = os.listdir(query_dest)[0]
    file = open(query_dest+os.sep+path, 'rb')
    Pics = pickle.load(file)
    file.close()
    for i in range(len(labels)):
        tmp = (query_SURF[i] > cluster[labels[i]]).astype(int)
        Pics[i].set_cluster(labels[i])
        Pics[i].set_hamming_code(tmp)
    file = open(query_dest+os.sep+path, 'wb')
    pickle.dump(Pics, file)
    file.close()

    # query

    query_Pics_dir = query_dest
    base_Pics_dir = dest
    bf = cv2.BFMatcher(cv2.NORM_L2)
    query_paths = os.listdir(query_Pics_dir)
    files = os.listdir(base_Pics_dir)
    deses = list()
    for path in files:
        file = open(dest + os.sep + path, 'rb')
        Pics = pickle.load(file)
        file.close()
        deses.append(Pics)

    # scores = dict()
    # mistakes = list()
    # for query_Pics_path in query_paths:
    #     start = time.time()
    #     query_Pics_path = dest + os.sep + query_Pics_path
    #     file = open(query_Pics_path, 'rb')
    #     query_Pics = pickle.load(file)
    #     found = 0
    #     file.close()
    #     counter = 0
    #     for des in deses:
    #         matches = bf.knnMatch(des, query_Pics, k=2)
    #         index = list()
    #         for m, n in matches:
    #             if m.distance < 0.5*n.distance:
    #                 index.append(m.queryIdx)
    #         if len(index) > 2300:
    #             print(len(index))
    #             found += 1
    #             print(files[counter])
    #         counter += 1
    #     if found > 4:
    #         print(query_Pics_path)
    #         mistakes.append(query_Pics_path)
    #     end = time.time()
    #     scores[query_Pics_path] = end-start
    #     print(query_Pics_path + ' done')
    # file = open('data\\mistakes.pkl', 'wb')
    # pickle.dump(mistakes, file)
    # file.close()
    # file = open('result.pkl', 'wb')
    # pickle.dump(scores, file)
    # file.close()
    # times = list()
    # for key, value in scores.items():
    #     times.append(value)
    # print(np.mean(times))

    query_Pics_path = query_dest + os.sep + os.listdir(query_dest)[0]
    file = open(query_Pics_path, 'rb')
    query_Pics = pickle.load(file)
    file.close()
    query_des = list()
    for line in query_Pics:
        query_des.append(line.des)
    query_des = np.array(query_des)
    files = os.listdir(dest)
    scores = dict()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    for item in files:
        file = open(dest + os.sep + item, 'rb')
        Pics = pickle.load(file)
        counter = 0
        des = list()
        for line in Pics:
            des.append(line.des)
        des = np.array(des)
        matches = bf.knnMatch(query_des, des, k=2)
        index = dict()
        for m, n in matches:
            if m.distance < 0.5*n.distance:
                index[m.queryIdx] = m.trainIdx
        for key, value in index.items():
            if query_Pics[key].cluster == Pics[value].cluster:
                if sum((query_Pics[key].hamming_code == Pics[value].hamming_code).astype(int)) > 20:
                    counter += 1
        scores[item] = counter/len(query_Pics)
        # if len(index) != 0 and counter / len(index) > 0.8:
        #     print(item + ' matches, score: ' + str(counter / len(index)))
        # scores[item] = len(index)/len(query_Pics)
    scores = sorted(scores.items(), key=lambda item:item[1], reverse=True)
    print(scores)
    file = open('result.pkl', 'wb')
    pickle.dump(scores, file)
    file.close()
    end = time.time()
    print(end-start)