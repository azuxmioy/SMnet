import csv
import os
import glob
import numpy as np

filepath = './soft_label/'
output_file = 'assemble.csv'

filelist =  glob.glob(filepath + "*.csv")

print (filelist)

all_label = np.zeros((2174,20),dtype=np.float)

for path in filelist:

    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row.
        soft_label = list(reader)

    float_label=[]

    for pair in soft_label:
        st = pair[1].replace('[', '').replace(']', '').split()
        float_label.append([float(i) for i in st])

    np_float_label = np.asarray(float_label)

    print (np_float_label.shape)

    all_label = all_label + np_float_label

print (all_label)

hard_label = np.argmax(all_label, axis=1)
print (hard_label)


with open(output_file, 'w') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["Id", "y"])
    for idx in range(2174):
        writer.writerow([idx+1, hard_label[idx]+1])