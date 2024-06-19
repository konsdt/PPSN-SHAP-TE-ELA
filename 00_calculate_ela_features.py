import itertools
import numpy as np
import os
import pandas as pd
import sys
import re

from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter
from config import BENCHMARKS, REPS, ROOT, SAMPLE_COEFFICIENT, ENCODINGS, OVERSAMPLE, OS_FACTOR, TRUE_FUNCTION
from multiprocessing import Pool
from pflacco.classical_ela_features import calculate_dispersion, calculate_ela_distribution, calculate_information_content, calculate_ela_meta, calculate_nbc
from yahpo_gym import benchmark_set
from utils import _extract_decision_space, _update_sample, _repair_sample
from objective_wrapper import ObjectiveWrapper

import shap
from sklearn.preprocessing import TargetEncoder, LabelEncoder

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT = os.path.join(ROOT, 'ela')
if not os.path.exists(ROOT):
   os.makedirs(ROOT)

def run_experiments(job):
    
    ###########################################################
    # is not used and could be deleted
    if OVERSAMPLE:
        encoding, bname, ins, os_factor = job
    else:
        encoding, bname, ins = job
    ###########################################################
    
    ela_results = []
    bench = benchmark_set.BenchmarkSet(scenario=bname, instance=ins, check=True, multithread=False)
    pnames, types, lower_bound, upper_bound, choices = _extract_decision_space(bench)

    for rep in range(REPS):
        if bname == 'nb301':
            dim = 34
        else:
            dim = len(bench.config.hp_names) - 1 - len(bench.config.fidelity_params)

        config = bench.get_opt_space(drop_fidelity_params = True, seed = rep).sample_configuration(SAMPLE_COEFFICIENT * dim)
        sample = []
        for s in config:
            s = _update_sample(s.get_dictionary(), bench)
            sample.append(s)
        result_dict = bench.objective_function(sample, multithread=False)
        
        # Preprocess initial sample
        X = pd.DataFrame(sample)
        X = X.loc[:, pnames]
        
        ###########################################################
        # is not used and could be deleted
        if OVERSAMPLE:
            na_cells = X.isna().sum().max()
            if na_cells > 0:
                na_idx = X.isna().any(axis=1)
                X_os = X[na_idx].copy()
                os_list = [X_os for x in range(os_factor)]
                os_list.append(X)
                X = pd.concat(os_list).reset_index(drop=True)
        ###########################################################
        for i in range(len(pnames)):
            pn = pnames[i]
            # Replace NA values with valid values from the respective search domain
            if X[pn].isna().any():
                if types[i] == 'cont':
                    X.loc[X[pn].isna(), pn] = np.random.uniform(lower_bound[i], upper_bound[i], size = X[pn].isna().sum())
                elif types[i] == 'int':
                    X.loc[X[pn].isna(), pn] = np.random.randint(lower_bound[i], upper_bound[i], size = X[pn].isna().sum())
                elif types[i] == 'cat':
                    X.loc[X[pn].isna(), pn] = np.random.choice(choices[i], size = X[pn].isna().sum())
                else:
                    raise Exception('Unknown param type encountered while replacing NAs')
                
            # Normalize non-categorical variables
            if types[i] != 'cat':
                X.loc[:, pn] = (X[pn] - lower_bound[i]) / (upper_bound[i] - lower_bound[i])

        # Get objective value
        y = pd.DataFrame(result_dict)
        if bname.startswith('iaml'):
            y = y['mmce'].values
        elif bname.startswith('rbv2'):
            y = 1 - y['acc'].values
        else:
            y = 1 - (y['val_accuracy'].values/100)
    
        ###########################################################
        # is not used and could be deleted
        if OVERSAMPLE:
            if na_cells > 0:
                y_os = y[na_idx]
                os_list = [y_os for x in range(os_factor)]
                os_list.append(y)
                y = np.concatenate(os_list) 
        ###########################################################

        # Transform Categorical Variables
        if encoding != 'SP':
            if encoding == 'OH':
                X = pd.get_dummies(X)
            elif encoding == 'TE':
                if not (X.dtypes == np.float64).all():
                    enc_auto = TargetEncoder(smooth="auto", target_type='continuous')
                    cat_cols = X.columns[np.array(types) == 'cat'].to_list()
                    # Transformation included on 5-fold CV
                    X_trans = pd.DataFrame(enc_auto.fit_transform(X.loc[:, cat_cols], y), columns = cat_cols)
                    # Whole dataset without CV
                    #tmp = enc_auto.fit(X.loc[:, cat_cols], y).transform(X.loc[:, cat_cols])
                    #X_trans = pd.DataFrame(tmp, columsn = cat_cols)
                    #X_trans = X_trans.apply(lambda x: (x - x.min())/(x.max() - x.min()), axis = 0)
                    X[cat_cols] = X_trans
            elif encoding == 'SH':

                cat_cols = X.columns[np.array(types) == 'cat'].to_list()
                labels = []
                # this encoders thing is only used in "if TRUE_FUNCTION": and 
                # this is a FALSE boolean
                encoders = {}
                for col in cat_cols:
                    le = LabelEncoder()
                    le.fit(np.unique(X[col].to_numpy()))
                    label = le.transform(X[col].to_numpy())
                    labels.append(label)
                    encoders[col] = le
                labels = np.array(labels)
                X_trans = pd.DataFrame(labels.T, columns=cat_cols)
                X[cat_cols] = X_trans

                if TRUE_FUNCTION:
                    model = ObjectiveWrapper(bname, ins, encoders = encoders, cat_cols = cat_cols)
                    model_mae = 0
                else:
                
                    model = RandomForestRegressor(random_state=50)
                    model.fit(X, y)
                    model_mae = mean_absolute_error(y, model.predict(X))
                
                #ex = shap.TreeExplainer(model)
                #shap_values = ex.shap_values(X)

                #ex = shap.explainers.Exact(model.predict, X)
                #shap_values = ex(X)
                #ex = shap.KernelExplainer(model.predict, X)
                ex = shap.PermutationExplainer(model.predict, X)
                shap_values = ex.shap_values(X)
                shap_values = pd.DataFrame(shap_values, columns=X.columns)
                #shap_values = shap_values.apply(lambda x: (x - x.min())/(x.max() - x.min()), axis = 0)
                X[cat_cols] = shap_values[cat_cols]

            else:
                raise ValueError(f'Unknown encoding [{encoding}] selected.')

            # Calculate ELA features
            disp = calculate_dispersion(X, y)
            ela_distr = calculate_ela_distribution(X, y)
            ic = calculate_information_content(X, y, seed = 50)
            ela_meta = calculate_ela_meta(X, y)
            nbc = calculate_nbc(X, y)

            ela_features = pd.DataFrame({**ela_distr, **ela_meta, **nbc, **disp, **ic}, index = [0])
            ela_features['bench'] = bname
            ela_features['dim'] = dim
            ela_features['instance'] = ins
            ela_features['rep'] = rep
            if encoding == 'SH':
                ela_features['model_mae'] = model_mae
                print(model_mae)
            ela_results.append(ela_features)

        else:
            rel_cols = np.array(types) == 'cat'
            ela_features = []
            cart_prod = []
            for ch in choices:
                if ch is not None:
                    cart_prod.append(list(ch))
            #cart_prod = [x for x in itertools.product(*cart_prod)]
            for subset in itertools.product(*cart_prod):
                X_subset = X[(X[X.columns[rel_cols]] == subset).all(axis = 1)]
                if X_subset.shape[1] < 5:
                    continue
                y_subset = y[X_subset.index]
                X_subset.reset_index(drop = True, inplace = True)

                # Remove categorical features
                X_subset = X_subset[X.columns[~rel_cols]]

                # Calculate ELA features
                disp = calculate_dispersion(X_subset, y_subset)
                ela_distr = calculate_ela_distribution(X_subset, y_subset)
                ic = calculate_information_content(X_subset, y_subset, seed = 50)
                ela_meta = calculate_ela_meta(X_subset, y_subset)
                nbc = calculate_nbc(X_subset, y_subset)

                ela_features_subset = pd.DataFrame({**ela_distr, **ela_meta, **nbc, **disp, **ic}, index = [0])
                ela_features.append(ela_features_subset)

            ela_features = pd.concat(ela_features).reset_index(drop = True)
            ela_features = ela_features.mean(axis = 0).to_frame().transpose()
            ela_features['bench'] = bname
            ela_features['dim'] = dim
            ela_features['instance'] = ins
            ela_features['rep'] = rep
            ela_results.append(ela_features)

    results = pd.concat(ela_results).reset_index(drop=True)
    if OVERSAMPLE:
        results.to_csv(os.path.join(ROOT,  f'os{os_factor}_{encoding}_{bname}_{ins}_ela_features.csv'), index = False)
    else:
        if TRUE_FUNCTION:
            true_func = 'tf'
        else:
            true_func = 'ml'
        #results.to_csv(os.path.join(ROOT,  f'inspect_{encoding}_{bname}_{ins}_ela_features.csv'), index = False)

# Function wrapper needed for multiprocessing
if __name__ ==  '__main__':
    if len(sys.argv) == 1:
        ela_files = os.listdir(ROOT)
        ela_jobs = []
        
        for file_ in ela_files:
            name = file_.replace('_ela_features.csv', '').split('_')
            if len(name) == 4:
                job = (name[0], name[1] + '_' + name[2], name[3])
            elif len(name) == 5:
                job = (name[1], name[2] + '_' + name[3], name[4], int(''.join(re.findall("\d", name[0]))))
            else:
                job = (name[0], name[1], name[2])
            ela_jobs.append(job)
        
        jobs_to_run = []
        for encoding in ENCODINGS:
            for bname in BENCHMARKS:
                if OVERSAMPLE:
                    for factor in OS_FACTOR:
                        bench = benchmark_set.BenchmarkSet(bname)
                        job = [(encoding, bname, x, factor) for x in bench.instances]
                        for j in job:
                            if j not in ela_jobs:
                                jobs_to_run.append(j)
                else:
                    bench = benchmark_set.BenchmarkSet(bname)
                    job = [(encoding, bname, x) for x in bench.instances]
                    for j in job:
                        if j not in ela_jobs:
                            jobs_to_run.append(j)


        # Debug code:
        for job in jobs_to_run[0:2]:
            run_experiments(job)

        with Pool(15) as p:
            p.map(run_experiments, jobs_to_run)
        
    else:
        raise SyntaxError("Insufficient number of arguments passed")
