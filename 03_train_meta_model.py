import numpy as np
import os
import pandas as pd
import random

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from config import ENCODINGS, ROOT

INPUT_FOLDER = './data'
OUTPUT_FOLDER = os.path.join(ROOT, 'model')
SEEDS = [42]
CV = 10
model_ = 'rf'
np.random.seed(100) 
random.seed(100)
# For a single encoding type, ela.shape[0] is 14040.
cv_splits = np.random.choice(CV, size=976)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

meta_data = pd.read_csv(os.path.join(INPUT_FOLDER, 'meta_data.csv'))
meta_data.rename(columns={'bname': 'bench'}, inplace=True)
meta_data['rel_cat'] = meta_data.apply(lambda x: x['n_cat'] / x['dim'], axis = 1)

# why are meta_labels only of length 991?
meta_labels = pd.read_csv(os.path.join(INPUT_FOLDER, 'meta_model_labels.csv'))
all_rel_ert = pd.read_csv(os.path.join(INPUT_FOLDER, 'rel_ert.csv'))
print(len(meta_labels))
for seed in SEEDS:
    results = []
    ela = pd.read_csv(os.path.join(INPUT_FOLDER, 'ela_data_combined.csv'))
    print(len(ela))
    ela = pd.merge(ela, meta_data[['bench', 'rel_cat']], how = 'left', on = 'bench')
    ela = ela.loc[:, ela.columns != 'type']
    #ela.drop('dim_y', axis=1, inplace=True)

    # Drop columns which contain NA or infinity values. Does not occur here
    ela.replace([np.inf, -np.inf], np.nan, inplace=True)
    ela.drop(columns=ela.columns[ela.isna().any().values], inplace = True)
    print(len(ela))
    # Remove runtime costs
    ela = ela.drop(columns = [x for x in ela.columns if 'costs_runtime' in x])
    ela_colnames = [x for x in ela.columns if x not in ['bench', 'instance', 'rep', 'model_mae']]


    # Merge ELA features with meta data
    ela = pd.merge(ela, meta_labels, how = 'inner', on = ['bench', 'dim', 'instance', 'rep']).reset_index(drop=True)
    X = ela.loc[:, ela_colnames]
    y = ela.label
    print(ela.shape)
    print(X)
    model = RandomForestClassifier(random_state=seed)

    print(len(X))
    for k in np.arange(CV):
        idx = cv_splits == k
        X_train = X.loc[~idx, :]
        X_test = X.loc[idx, :]
        y_train = y.loc[~idx]
        y_test = y.loc[idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        tmp = ela.loc[idx, ['bench', 'dim', 'instance', 'rep']]
        tmp['solver'] = y_pred
        print(len(tmp))
        tmp = pd.merge(tmp, all_rel_ert, how='left', on=['bench', 'instance', 'solver'])
        tmp['model'] = model_
        results.append(tmp)

    result = pd.concat(results)

    result.to_csv(os.path.join(OUTPUT_FOLDER, f'my_meta_model_prediction.csv'), index=False)

