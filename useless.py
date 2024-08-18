import  pickle

path = '/disk2/jqc/COSeg/datasets/ScanNet/blocks_bs1_s1/class2scans.pkl'
with open(path, "rb") as f:
    class2scans = pickle.load(f)
for k,v in class2scans.items():
    print('##################################################################')
    print(k)
    print(v)
# print(class2scans)