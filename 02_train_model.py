import numpy as np
import os
import pandas as pd
import random

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from config import ENCODINGS, ROOT

# INPUT_FOLDER == ROOT
INPUT_FOLDER = './data'
OUTPUT_FOLDER = os.path.join(ROOT, 'model')
SEEDS = [42]
CV = 10
model_ = 'rf'
np.random.seed(100) #setting randoms seeds just to make sure ? 
random.seed(100) # but why are the different from 42
# For a single encoding type, ela.shape[0] is 14040.
cv_splits = np.random.choice(CV, size=14040)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

meta_data = pd.read_csv(os.path.join(INPUT_FOLDER, 'meta_data.csv'))
meta_data.rename(columns={'bname': 'bench'}, inplace=True)
meta_data['rel_cat'] = meta_data.apply(lambda x: x['n_cat'] / x['dim'], axis = 1)
class_labels = pd.read_csv(os.path.join(INPUT_FOLDER, 'label_data.csv'))
class_labels = class_labels.loc[:, ['bench', 'instance', 'solver']]
all_rel_ert = pd.read_csv(os.path.join(INPUT_FOLDER, 'rel_ert.csv'))

for seed in SEEDS:
    results = []
    # Only set to "SH" right now? Kann ich auf "SH" und "TE" setzen
    for enc in ENCODINGS:
        ela = pd.read_csv(os.path.join(INPUT_FOLDER, 'ela_data_all.csv'))
        ela = pd.merge(ela, meta_data[['bench', 'rel_cat']], how = 'left', on = 'bench')
        ela = ela[ela.type == enc]
        ela = ela.loc[:, ela.columns != 'type']

        # Drop columns which contain NA or infinity values. Does not occur here
        ela.replace([np.inf, -np.inf], np.nan, inplace=True)
        ela.drop(columns=ela.columns[ela.isna().any().values], inplace = True)

        # Remove runtime costs
        ela = ela.drop(columns = [x for x in ela.columns if 'costs_runtime' in x])
        ela_colnames = [x for x in ela.columns if x not in ['bench', 'instance', 'rep', 'model_mae']]

        # Scale data if we are going to use SVM
        if model_ == 'svm':
            ela[ela_colnames] = ela[ela_colnames].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        # Merge ELA features with meta data
        ela = pd.merge(ela, class_labels, how = 'left', on = ['bench', 'instance'])
        X = ela.loc[:, ela_colnames]
        y = ela.solver


            
        model = RandomForestClassifier(random_state=seed)

        
        for k in np.arange(CV):
            print(f'Training {model_} with {enc} in split {k}')
            idx = cv_splits == k
            X_train = X.loc[~idx, :]
            X_test = X.loc[idx, :]
            y_train = y.loc[~idx]
            y_test = y.loc[idx]

            #model.fit(X_train[list(sffs.k_feature_names_)], y_train)
            model.fit(X_train, y_train)
            #y_pred = model.predict(X_test[list(sffs.k_feature_names_)])
            y_pred = model.predict(X_test)


            tmp = ela.loc[idx, ['bench', 'dim', 'instance', 'rep']]
            tmp['solver'] = y_pred

            tmp = pd.merge(tmp, all_rel_ert, how='left', on=['bench', 'instance', 'solver'])
            tmp['encoding'] = enc
            tmp['model'] = model_
            results.append(tmp)

    result = pd.concat(results)

    result.to_csv(os.path.join(OUTPUT_FOLDER, f'final_{model_}_{seed}_aas_prediction.csv'), index=False)

