import os
import datetime


class SaveWithHistory:
    def __init__(self, root, col_name=['']*1114):
        self.root = root
        self.now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print('[ Logging in folder %s ]' % self.now_str)
        if not os.path.exists(root+self.now_str+'/'):
            os.makedirs(root+self.now_str+'/')
        with open('./feature_config.py') as f:
            configs = f.read()
        with open(root+self.now_str+'/config.log', 'w') as f:
            f.write(configs)
        with open(root+self.now_str+'/features.csv', 'w') as f:
            f.write('Entry,Index,'+','.join(col_name))
            f.write('\n')

    def add_line(self, entry, index, features):
        features = [str(x) for x in features]
        with open(self.root+self.now_str+'/features.csv', 'a') as f:
            f.write(entry+','+str(index)+','+','.join(features)+'\n')


if __name__ == '__main__':

    sh = SaveWithHistory('../../data/features/')
