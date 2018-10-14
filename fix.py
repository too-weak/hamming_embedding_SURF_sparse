import os
import numpy as np
import pickle
from CLASSES.CLASSES import Pic
from concurrent.futures import ThreadPoolExecutor

def convert(dir, name, dest_dir, fmt):
    file = open(dir + os.sep + name, 'rb')
    dest_file = dest_dir + os.sep + os.path.splitext(name)[0] + '.txt'
    Pics = pickle.load(file)
    file.close()
    deses = list()
    if type(Pics[0].des) == int:
        for pic in Pics:
            deses.append(pic.des)
    else:
        for pic in Pics:
            deses.append(pic.des.tolist())
    np.savetxt(dest_file, deses, fmt=fmt)

def fix(source_path, target_path, out_path, op):
    files = os.listdir(source_path)
    if op == 'SURF':
        fmt = '%0.4f'
    else:
        fmt = '%d'
    with ThreadPoolExecutor(10) as executor:
       for file in files:
           # executor.submit(convert, source_path, file, target_path)
            convert(source_path, file, target_path, fmt)
    files = os.listdir(target_path)
    out_file = open(out_path, 'w')
    for path in files:
        file = open(target_path + os.sep + path, 'r')
        out_file.writelines(file.readlines())
        file.close()
    out_file.close()


# dir = 'resource\\gongsi1000'
# dest = 'resource\\gongsi1000SURF'
# segment_path = 'resource\\segment'
# sample_path = 'data\\sample_with_kp.txt'
# trainmodel_path = 'data\\trainmodel.mat'
# SURF_path = 'resource\\gongsi1000SURF'
# SURF_txt_path = 'resource\\gongsi1000SURFtxt'
# sparse_txt_path = 'resource\\gongsi1000_sparse_txt'
# sparse_data_path = 'data\\sparse.txt'
# fix(SURF_path, sparse_txt_path, sparse_data_path)