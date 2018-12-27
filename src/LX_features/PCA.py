from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

source_file = '../../data/LX_features/20180611_223038_[284 columns].csv'
# source_file = '../../data/LX_features/test.csv'

df = pd.DataFrame(pd.read_csv(source_file, index_col=0))
static = df[['Index', 'Entry', 'Label']]
old = pd.DataFrame(df.drop(['Index', 'Entry', 'Label'], axis=1))
# old, static = df, df

X = np.array(old)
pca = PCA(n_components='mle')
new_X = pd.DataFrame(pca.fit_transform(X))
new_X = static.join(new_X)

print(pca.explained_variance_ratio_)
print(pca.components_)
print(pca.n_components_)
new_X.to_csv(source_file[:-4]+'_PCA.csv')
