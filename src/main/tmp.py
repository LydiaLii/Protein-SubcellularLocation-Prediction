import hashlib
import os
import time


def GetFileMd5(filename):
    if not os.path.isfile(filename):
        return
    myhash = hashlib.md5()
    f = open(filename, 'rb')
    while True:
        b = f.read(8096)
        if not b:
            break
        myhash.update(b)
    f.close()
    return myhash.hexdigest()


filepath = '../../data/LX_features/20180611_223038_[284 columns]_adjusted.csv'

# 输出文件的md5值以及记录运行时间
starttime = time.time()
print(GetFileMd5(filepath))
endtime = time.time()
print('运行时间：%.2fs' % (endtime - starttime))
