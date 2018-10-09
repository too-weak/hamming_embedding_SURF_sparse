import cv2
import os
from sklearn.cluster import KMeans
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import fix
# import akm
# import clusterAssign
import matlab.engine
import scipy.io as sio

numclusters = 20000
numiters = 50

d=64
d_b=64

class Pic(object):
    def __init__(self, des, x, y):
        self.des = des
        self.x = x
        self.y = y

    def set_hamming_code(self, code):
        self.hamming_code = code

def SURF_exa(dir, file, dest):
    hessianThreshold = 3000
    img = cv2.imread(dir + os.sep + file, cv2.IMREAD_COLOR)
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
    kp, des = surf.detectAndCompute(img, None)
    s = list()
    if len(des) > 2000:
        while len(des) > 2000:
            hessianThreshold += 500
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
            kp, des = surf.detectAndCompute(img, None)
    else:
        while len(des) < 2000 and hessianThreshold > 0:
            hessianThreshold -= 100
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
            kp, des = surf.detectAndCompute(img, None)
    for i in range(len(des)):
        s.append(Pic(des[i], kp[i].pt[1], kp[i].pt[0]))
    file = open(dest + os.sep + os.path.splitext(file)[0] + '.pkl', 'wb')
    pickle.dump(s, file)
    file.close()
    des = des.tolist()
    np.savetxt(dest + os.sep + os.path.splitext(file)[0] + '.txt', des, fmt='%.4f')


def getSURFS(dir, dest):
    imgs = os.listdir(dir)
    # for img in imgs:
    #     SURF_exa(dir, img, dest)
    with ThreadPoolExecutor(5) as executor:
        for img in imgs:
            executor.submit(SURF_exa, dir, img, dest)

def get_data(dir):
    dest_path = 'SURFs.pkl'
    if os.path.exists(dest_path):
        dest_file = open(dest_path, 'rb')
        res = pickle.load(dest_file)
        dest_file.close()
    else:
        surfs = list()
        files = os.listdir(dir)
        for item in files:
            file = open(dir+os.sep+item, 'rb')
            data = pickle.load(file)
            surfs.append(data)
            file.close()
        res = surfs.pop()
        while len(surfs) > 0:
            res = np.append(res, surfs.pop(), axis=0)
        dest_file = open('SURFs.pkl', 'wb')
        pickle.dump(res, dest_file)
        dest_file.close()
    return res

if __name__ == '__main__':
    dir = 'resource\\gongsi1000'
    dest = 'resource\\gongsi1000SURF'
    segment_path = 'resource\\segment'
    getSURFS(dir, dest)
    fix.fix()
    eng = matlab.engine.start_matlab()
    eng.ksvd_train('data\\sample_with_kp.txt', 'data\\trainmodel.mat')
    eng.segmentFeature_train()

    segments = os.listdir(segment_path)
    for segment in segments:
        name = os.path.splitext(segment)[0]
        file = open(dest + os.sep + name + '.pkl', 'rb')
        s = pickle.load(file)
        file.close()
        sparse = sio.loadmat(segment_path + os.sep + segment)['V']
        if len(sparse) == len(s):
            for i in range(len(sparse)):
                s[i].des = sparse[i]
        file = open(dest + os.sep + name + '.pkl', 'wb')
        pickle.dump(s, file)
        file.close()
    # SURF = np.array(get_data(dest))
    # kmeans = KMeans(n_clusters=500).fit(SURF)

    # #read P
    # if os.path.isfile('data\\projection.npy'):
    #     P = np.load('data\\projection.npy')
    # else:
    #     # compute projection matrix
    #     M = np.random.randn(d, d)
    #     Q, R = np.linalg.qr(M)
    #     P = Q[:d_b, :]
    #     np.save('data\\projection.npy', P)

    # proj = np.matmul(P, np.transpose(SURF))
    # proj = np.transpose(proj)
    # labels = kmeans.predict(proj)
    # bag = dict()
    # for i in range(len(labels)):
    #     if labels[i] not in bag.keys():
    #         bag[labels[i]] = list()
    #     bag[labels[i]].append(proj[i])
    # b = dict()
    # for key, value in bag.items():
    #     tmp = np.ndarray(value)
    #     tmp_sum = np.sum(tmp, axis=0)
    #     b[key] = tmp_sum/len(tmp)

