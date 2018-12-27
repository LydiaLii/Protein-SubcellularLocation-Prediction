import os
import datetime


class SaveWithHistory:
    def __init__(self, root, col_name=['']*1114):
        self.root = root
        self.now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.col_len = str(len(col_name))
        print('[ Logging in file %s.csv ]' % self.now_str)
        with open(root+self.now_str+'_['+self.col_len+' columns].csv', 'w') as f:
            f.write('Index,Entry,Label,'+','.join(col_name))
            f.write('\n')

    def add_line(self, i, entry, index, features):
        features = [str(x) for x in features]
        with open(self.root+self.now_str+'_['+self.col_len+' columns].csv', 'a') as f:
            f.write(str(i)+','+entry+','+str(index)+','+','.join(features)+'\n')


if __name__ == '__main__':

    sh = SaveWithHistory('../../data/features/')
