import pandas as pd
import numpy as np
import biom
import random
#import qiime2 as q2

from scipy.sparse import issparse
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, average_precision_score
 
'''
# load full table & metadata
full_md = pd.read_csv('agp_metadata.tsv', sep='\t', index_col=0, low_memory=False)
full_bt = biom.load_table('agp_samples.biom')

# filter metadata 
md_agp = full_md.loc[full_md.host_age >= 18] # remove individuals under 18
md_agp = md_agp.loc[md_agp.sample_type == 'Stool'] # only keep stool samples
full_bt.filter(ids_to_keep=md_agp.index)

# add sequencing depth to metadata 
md_agp['seq_depth'] = full_bt.to_dataframe().sum()

md_agp_sorted = md_agp.sort_values(by='seq_depth', ascending=False)
md_agp_dedup = md_agp_sorted.loc[~md_agp_sorted['host_subject_id'].duplicated()]
md_agp_5000_min = md_agp_dedup.loc[md_agp_dedup['seq_depth'] >= 5000]
full_bt.filter(ids_to_keep=md_agp_5000_min.index)
'''

# load pre-filtered table & metadata 
md_agp_5000_min = pd.read_csv('agp_metadata_filtered.tsv', sep='\t', index_col=0)
full_bt = biom.load_table('agp_table_filtered.biom')

print('filter done') 

def do_ridge(df, md, column_name, random_state= 2, alpha=0.01, l1_ratio=None):
    X = df
    y = md.loc[X.index].get(column_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    sel_ = SelectFromModel(Ridge(alpha=alpha, random_state=random_state))
    sel_.fit(X, y)
    return X[X.columns[sel_.get_support()]]

def prev_filter(df, min_prev, min_rel_abun = 0):
    row_sum = df.sum(axis = 1)
    df_new = df.div(row_sum, axis = 0)
    df_new = df_new.where(df_new <= min_rel_abun, 1)
    col_sum = df_new.sum(axis = 0)
    min_col_sum = min_prev * df.shape[0]
    cols_to_keep = col_sum[col_sum >= min_col_sum].index
    return df[cols_to_keep]

def rf(df, md, n_splits, test_variable, true_value, random_state = 6, use_fi = False):
    if use_fi== True:
        X = df.copy()
        y = md.loc[X.index].get(test_variable)
        
        forest = RandomForestClassifier(random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
        feature_names = [X.columns[i] for i in range(X.shape[1])]
        forest = RandomForestClassifier(random_state=0)
        forest.fit(X_train, y_train)
        result = permutation_importance(
                forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
        forest_importances = pd.Series(result.importances_mean, index=feature_names)
        forest_importances = forest_importances.sort_values(ascending=False)
        filist = forest_importances.index.tolist()
        filist
        subsettedX = X[filist]
        subsettedX
    else:
        subsettedX = df
    i_to_id = {i:subsettedX.index[i] for i in range(len(subsettedX.index))}
    y = md.loc[subsettedX.index].get(test_variable)
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    auclist = []
    aprlist = []
    for i, (train_index, test_index) in enumerate(kf.split(subsettedX)):
        train_ids = [i_to_id[j] for j in train_index]
        test_ids = [i_to_id[k] for k in test_index]
        X_train = subsettedX.loc[train_ids]
        y_train = y.loc[train_ids]
    
        X_test = subsettedX.loc[test_ids]
        y_test = y.loc[test_ids]
        rf = RandomForestClassifier(random_state = random_state)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_test)[:, 1]
        fpr, tpr, _, = roc_curve(y_test, y_pred, pos_label=true_value)
        auc = round(roc_auc_score(y_test, y_pred), 4)
        auclist.append(auc)
        apr = round(average_precision_score(y_test, y_pred, pos_label=true_value), 4)
        aprlist.append(apr)
    return pd.DataFrame(data={'auc': auclist, 'apr': aprlist, 'fold': range(n_splits)})

data_dict = {
    'HC vs MDD': full_bt.to_dataframe().T,  
}
print('loading done') 
birdman_reference_df = pd.read_csv('/projects/u19/Tulsa/final_tables/birdman_reference_data.tsv', sep='\t', index_col=0)

result_dfs = []
random_integers = random.sample(range(1, 101), 10)
for i in range(10): 
    print(i)
    for k in data_dict: 
        if 'HC' in k: 
            test_variable = 'mental_illness_type_depression'
            true_value = True
        else: 
            test_variable = 'medication_group'
            true_value = 'Antidepressant/Antianxiety'


        
        # run helper on full thing 
        unfilt_result = rf(data_dict[k], md_agp_5000_min, 5, test_variable, true_value, random_state=random_integers[i]) 
        unfilt_result['feature_selection'] = 'unfiltered'
        print('unfiltered done')

        # call prevalence filter 
        prev_filtered = prev_filter(data_dict[k], 0.05, min_rel_abun = 0)
        prev_filtered_result = rf(prev_filtered, md_agp_5000_min, 5, test_variable, true_value, 
                                  random_state=random_integers[i]) 
        prev_filtered_result['feature_selection'] = 'prevalence_filter'
        print('prev done') 

        # call ridge 
        #md_copy = md_agp_5000_min.copy()
        #md_copy[test_variable] = np.where(md_copy[test_variable] == true_value, 1, 0)
        ridge_filtered = do_ridge(data_dict[k], md_agp_5000_min, test_variable)
        ridge_filtered_result = rf(ridge_filtered, md_agp_5000_min, 5, test_variable, true_value, 
                                   random_state=random_integers[i])
        ridge_filtered_result['feature_selection'] = 'ridge_filter'
        print('ridge done') 
        

        '''
        # FI 
        fi_filtered_result = rf(data_dict[k], md_agp_5000_min, 5, test_variable, true_value, random_state=random_integers[i], 
                                use_fi=True) 
        fi_filtered_result['feature_selection'] = 'feature_importance'
        '''
        # birdman 
        key_name = k.split(' ')[-1]
        birdman_filter_cols = birdman_reference_df.loc[birdman_reference_df['subset'] == key_name].taxa.values 
        birdman_filtered = data_dict[k][data_dict[k].columns.intersection(birdman_filter_cols)]
        birdman_filtered_results = rf(birdman_filtered, md_agp_5000_min, 5, test_variable, true_value, 
                                      random_state=random_integers[i])
        birdman_filtered_results['feature_selection'] = 'birdman'
        print('birdman done')

        concat_df = pd.concat([unfilt_result, prev_filtered_result, ridge_filtered_result, #fi_filtered_result, 
                               birdman_filtered_results])
        
        #concat_df = fi_filtered_result.copy()
        concat_df['subset'] = k
        concat_df['iteration'] = i
        result_dfs.append(concat_df)
final_results_auc = pd.concat(result_dfs)
final_results_auc.to_csv('agp_transfer_rf.tsv', sep='\t')