import numpy as np
import os
import pandas as pd
import random


from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from config import ENCODINGS, ROOT

INPUT_FOLDER = './data'
OUTPUT_FOLDER = os.path.join(ROOT, 'model')
SEEDS = [42]
CV = 10

np.random.seed(100) 
random.seed(100)
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

for seed in SEEDS: # only one seed which is also not used  ??
    results = []
    for enc in ENCODINGS: # only one encoding which is not used in this file ??
        ela = pd.read_csv(os.path.join(INPUT_FOLDER, 'ela_data_all.csv'))
        ela = pd.merge(ela, meta_data[['bench', 'rel_cat']], how = 'left', on = 'bench')
        ela_sh = ela.copy()
        ela_sh = ela_sh[ela_sh.type == 'SH'].reset_index(drop=True)
        ela_te = ela.copy()
        ela_te = ela_te[ela_te.type == 'TE'].reset_index(drop=True)


        # Drop columns which contain NA or infinity values. Does not occur here
        ela_sh.replace([np.inf, -np.inf], np.nan, inplace=True)
        ela_sh.drop(columns=ela_sh.columns[ela_sh.isna().any().values], inplace = True)
        ela_te.replace([np.inf, -np.inf], np.nan, inplace=True)
        ela_te.drop(columns=ela_sh.columns[ela_te.isna().any().values], inplace = True)

        # Remove runtime costs
        ela_sh = ela_sh.drop(columns = [x for x in ela_sh.columns if 'costs_runtime' in x])
        ela_te = ela_te.drop(columns = [x for x in ela_te.columns if 'costs_runtime' in x])
        ela_colnames = [x for x in ela_sh.columns if x not in ['bench', 'instance', 'rep', 'model_mae', 'type']]

        # Scale data if we are going to use SVM
        # if? only in other file? here the data is always scaled
        # FIXME: Was the search space normalized when ELA was calculated?
        ela_sh[ela_colnames] = ela_sh[ela_colnames].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        ela_te[ela_colnames] = ela_te[ela_colnames].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


        # Merge ELA features with meta data
        ela_sh = pd.merge(ela_sh, class_labels, how = 'left', on = ['bench', 'instance'])
        ela_te = pd.merge(ela_te, class_labels, how = 'left', on = ['bench', 'instance'])

        X_sh = ela_sh.loc[:, ela_colnames]
        y_sh = ela_sh.solver
        X_te = ela_te.loc[:, ela_colnames]
        y_te = ela_te.solver
 
        #model_te = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(512, 512, 4), random_state=100, max_iter=500)
        #model_sh = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(512, 512, 4), random_state=100, max_iter=500)
        model_sh = RandomForestClassifier(random_state=42)
        model_te = RandomForestClassifier(random_state=42)


        for k in np.arange(CV):
            print(f'Training with {enc} in split {k}') # this is not a true print since the ENCODINGS loop is not made use of
            idx = cv_splits == k
            X_train_sh = X_sh.loc[~idx, :]
            X_test_sh = X_sh.loc[idx, :]
            y_train_sh = y_sh.loc[~idx]
            y_test_sh = y_sh.loc[idx]

            X_train_te = X_te.loc[~idx, :]
            X_test_te = X_te.loc[idx, :]
            y_train_te = y_te.loc[~idx]
            y_test_te = y_te.loc[idx]

            model_te.fit(X_train_te, y_train_te)
            model_sh.fit(X_train_sh, y_train_sh)
            y_pred_te = model_te.predict(X_test_te)
            y_prob_te = model_te.predict_proba(X_test_te)
            y_pred_sh = model_sh.predict(X_test_sh)
            y_prob_sh = model_sh.predict_proba(X_test_sh)

            tmp = ela_sh.loc[idx, ['bench', 'dim', 'instance', 'rep']]
            tmp['solver_te'] = y_pred_te
            tmp['prob_te'] = np.max(y_prob_te, axis = 1)
            tmp['solver_sh'] = y_pred_sh
            tmp['prob_sh'] = np.max(y_prob_sh, axis = 1)

            tmp['solver'] = tmp.apply(lambda x: x['solver_te'] if x['prob_te'] >= x['prob_sh'] else x['solver_sh'], axis = 1)
            tmp = pd.merge(tmp, all_rel_ert, how='left', on=['bench', 'instance', 'solver'])
            tmp['encoding'] = enc
            tmp['model'] = 'ML' # Not MLP used but random forest in this case
            
            results.append(tmp)

    result = pd.concat(results)

    result.to_csv(os.path.join(OUTPUT_FOLDER, f'final_RFHB_{seed}_aas_prediction.csv'), index=False)

