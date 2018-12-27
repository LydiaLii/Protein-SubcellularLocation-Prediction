import matplotlib.pyplot as plt
import pandas as pd


def plot(length_dict):
    i = 1
    fig = plt.figure(figsize=(24, 10))
    for key, value in length_dict.items():
        ax = fig.add_subplot(5, 12, i)
        ax.hist(value, bins=200)
        plt.title(key)
        plt.xlim(0, 1)
        plt.ylim(0, 600)
        i += 1
    plt.tight_layout()
    plt.show()


df = pd.DataFrame(pd.read_csv('../../data/LX_features/20180611_223038_[284 columns]_adjusted.csv', index_col=0))

coi = df.columns.values.tolist()[3:]
draw_list = coi[:60]

data = {}
for column in draw_list:
    data[column] = df[column].tolist()
plot(data)
