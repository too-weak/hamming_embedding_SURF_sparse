import os
import numpy as np
import pickle
from hamming import *
from concurrent.futures import ThreadPoolExecutor

def convert(dir, name, dest_dir):
    file = open(dir + os.sep + name, 'rb')
    dest_file = dest_dir + os.sep + os.path.splitext(name)[0] + '.txt'
    Pics = pickle.load(file)
    deses = list()
    for pic in Pics:
        deses.append(pic.des.tolist())
    np.savetxt(dest_file, deses, fmt='%.4f')
    file.close()

def fix():
    source_path = 'resource\\gongsi1000SURF'
    target_path = 'resource\\gongsi1000SURFtxt'
    files = os.listdir(source_path)
    with ThreadPoolExecutor(10) as executor:
       for file in files:
           executor.submit(convert, source_path, file, target_path)
    files = os.listdir(target_path)
    out_path = 'data\\sample_with_kp.txt'
    out_file = open(out_path, 'w')
    for path in files:
        file = open(target_path + os.sep + path, 'r')
        out_file.writelines(file.readlines())
        file.close()
    out_file.close()