
"""
The aim of this script is to compare TRACEvar performance to common genetic tools (Fig. 2C-D).
Tissue specific performance data and AUC plots created here, overall comparision is in the Methods_Comparison_Analysis.ipynb script.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import pickle
import shap
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
import pickle
from scipy import interp

root_directory = 'Your relevant directory'

"------------------------------------------  Overlap cols ----------------------------------"
path = os.path.join(root_directory, 'Relevant_Columns_Names_Edited_2.csv')
Relevant_Cols_df = pd.read_csv(path)
overlap_cols = Relevant_Cols_df['Feature'].tolist()
rename_dict = dict(zip(overlap_cols, Relevant_Cols_df['Feature Name'].tolist()))

"------------------------------------------  Load Data ----------------------------------"

path = os.path.join(root_directory, 'test_set_2023_diff_no_common_genes.csv')
Variants_data = pd.read_csv(path, engine='python')

path = os.path.join(root_directory, 'new_parameters_file.csv')
Best_param = pd.read_csv(path, engine='python')

path_capice = os.path.join(root_directory, 'CAPICE_hg37_VariantID.csv')
CAPICE_score = pd.read_csv(path_capice)

"----------------------------------- Create Scores DF -------------------------------------"
scores_y_columns = Variants_data.columns[Variants_data.columns.str.contains(pat='disease_causing')].tolist()
scores_y_columns = scores_y_columns + ['VariationID', 'PHRED', 'SIFTval', 'PolyPhenVal']
Scores_DF = pd.merge(Variants_data[scores_y_columns], CAPICE_score, on='VariationID', how='left') # VariationID
print('Scores_DF', Scores_DF)
Scores_DF['CAPICE'] = Scores_DF['CAPICE_Score'].apply(lambda x: x * 100)
Scores_DF['SIFT'] = Scores_DF['SIFTval'].apply(lambda x: x * 100)
Scores_DF['PolyPhen'] = Scores_DF['PolyPhenVal'].apply(lambda x: x * 100)

"----------------------------------- Split data into non baiased folds -------------------------------------"


def split_data_on_genes(Variants_data, y):
    Genes_fraction = Variants_data['GeneID_y'][Variants_data[y] == True].value_counts(normalize=True)
    print(Genes_fraction.head(5))
    tissue = y.split('_')[0]
    Genes_as_factor = pd.DataFrame()
    Genes_as_factor['GeneID_y'] = Variants_data['GeneID_y'][Variants_data[y] == True].astype('category')
    Genes_as_factor["Gene_Categorical"] = Genes_as_factor["GeneID_y"].cat.codes
    Genes_fraction.columns = ['Fraction']
    return Genes_fraction, Genes_as_factor


def find_fold(Genes_fraction):
    Genes_fraction = Genes_fraction.sort_values()
    Genes_fraction = Genes_fraction.reset_index()
    Genes_fraction.columns = ['GeneID', 'Fraction']
    inds = Genes_fraction.index
    fold_list = []
    genes_list = []
    count_list = []
    fold = 0
    threshold = 0.1
    counter = 0
    for ind in inds:
        fraction = Genes_fraction.iloc[ind]['Fraction']
        gene = Genes_fraction.iloc[ind]['GeneID']
        counter += fraction
        if counter > threshold:
            fold += 1
            counter = 0
        if fraction >= threshold or fold > (1 / threshold - 1):
            genes_list.append(gene)
            fold_list.append(None)
            break
        else:
            genes_list.append(gene)
            fold_list.append(fold)
    Fold_df = pd.DataFrame(list(zip(genes_list, fold_list)), columns=['GeneID', 'Fold'])
    Fold_merge = Genes_fraction.merge(Fold_df, on='GeneID', how='left')

    return Fold_merge


def random_split(Folds_df):
    threshold = 0.1
    Folds_df = Folds_df.sort_values('GeneID')
    Shafeled_df = Folds_df.sample(frac=1, random_state=1234).reset_index(drop=True)
    inds = Shafeled_df.index
    new_fold = [None] * len(Folds_df)
    nan_inds = Shafeled_df[Shafeled_df['Fold'].isna()].index
    used_inds = [x for x in nan_inds]
    fold = 0
    for ind in inds:
        cuonter = 0
        fraction = Shafeled_df.iloc[ind]['Fraction']
        gene = Shafeled_df.iloc[ind]['GeneID']
        old_fold = Shafeled_df.iloc[ind]['Fold']
        cuonter += fraction
        if ind not in used_inds:
            new_fold[ind] = fold
            used_inds.append(ind)
            relevant_inds = [x for x in inds if x not in used_inds]
            for rel_ind in relevant_inds:
                fraction2 = Shafeled_df.iloc[rel_ind]['Fraction']
                if fraction2 + cuonter <= threshold:
                    cuonter += fraction2
                    new_fold[rel_ind] = fold
                    used_inds.append(rel_ind)
            fold += 1
    Shafeled_df['New_Fold'] = np.array(new_fold)
    print(Shafeled_df)
    folds = Shafeled_df['New_Fold'].unique()
    last_fold_num = Shafeled_df['New_Fold'].max()
    last_fold = Shafeled_df[Shafeled_df['New_Fold'] == last_fold_num]
    last_fold_sum = last_fold['Fraction'].sum()

    if last_fold_sum < 0.07:
        Shafeled_df['New_Fold'][Shafeled_df['New_Fold'] == last_fold_num] = last_fold_num - 1

    return Shafeled_df


y_columns = Variants_data.columns[Variants_data.columns.str.contains(pat='disease_causing')].tolist()
print(y_columns)
cols = overlap_cols
"-------------------------- Data PreProcessing ------------------------"


def preprocessing_new(Relevant_data, y_columns, y):
    "------------- Remove nonrelevant coluns -------------------------"
    non_relevant_columns = ['VariationID', 'OMIMs', 'Manifested_Tissues', '#Chr', 'Pos', 'ConsDetail', 'motifEName',
                            'FeatureID', 'GeneName', 'CCDS', 'Intron', 'Exon', 'SIFTcat', 'PolyPhenCat', 'bStatistic',
                            'targetScan', 'dbscSNV-rf_score', 'oAA', 'Ref', 'nAA', 'Alt',
                            'Segway']  # it will be good to replace oAA and nAA with blssuom64 matrix. What bStatistic doing?
    non_relevant_columns = non_relevant_columns + y_columns
    print(non_relevant_columns)
    relevant_columns = [x for x in cols if x not in non_relevant_columns and x in list(Variants_data)]
    relevant_columns.append(y)
    print(relevant_columns)
    Relevant_data = Variants_data[relevant_columns]
    print(Relevant_data)

    "---------------------- One Hot Columns -------------------------"

    one_hot_columns = ['Type', 'AnnoType', 'Consequence', 'Domain', 'Dst2SplType']
    one_hot = pd.get_dummies(Relevant_data[one_hot_columns])
    Relevant_data = Relevant_data.drop(one_hot_columns, axis=1)
    Relevant_data = Relevant_data.join(one_hot)

    "---------------------- Missing Values Imputation ---------------"

    special_imputation_cols = {'SIFTval': 1, 'GC': 0.42, 'CpG': 0.02, 'priPhCons': 0.115, 'mamPhCons': 0.079,
                               'verPhCons': 0.094, 'priPhyloP': -0.033, 'mamPhyloP': -0.038, 'verPhyloP': 0.017,
                               'GerpN': 1.91, 'GerpS': -0.2}

    for cl in special_imputation_cols:
        Relevant_data[cl] = Relevant_data[cl].fillna(special_imputation_cols[cl])

    Relevant_data.fillna(0, inplace=True)

    return Relevant_data


"---------------------------------------- AUC drawing functions --------------------------------"


def plot_mean_auc(tprs, mean_fpr, aucs, dataset, color, ax):
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color=color,
            label=dataset + r' (auROC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=3, alpha=.8)


def cadd_as_score_measurements(cadd_score_column, y):
    CADD_Scores_df = pd.concat([cadd_score_column, y], axis=1)  # , ignore_index=True)
    CADD_Scores_df.columns = ['Pathogenicity_Score', 'tissue_specific_disease']
    CADD_Scores_df.dropna(subset=['Pathogenicity_Score'], inplace=True)
    print('CADD_Scores_df', CADD_Scores_df)
    max = 102
    ROC = np.zeros((max, 2))
    Precision_Recall = np.zeros((max, 2))
    average_precision = 0
    TPR_previuos = 0
    for i in range(max):
        # t = thresholds[i]
        t = i
        print('threshold: ', t)
        TP_t = len(CADD_Scores_df[(CADD_Scores_df['Pathogenicity_Score'] >= t) & (
                    CADD_Scores_df['tissue_specific_disease'] == True)])
        TN_t = len(CADD_Scores_df[(CADD_Scores_df['Pathogenicity_Score'] < t) & (
                    CADD_Scores_df['tissue_specific_disease'] == False)])
        FP_t = len(CADD_Scores_df[(CADD_Scores_df['Pathogenicity_Score'] >= t) & (
                    CADD_Scores_df['tissue_specific_disease'] == False)])
        FN_t = len(CADD_Scores_df[(CADD_Scores_df['Pathogenicity_Score'] < t) & (
                    CADD_Scores_df['tissue_specific_disease'] == True)])
        FPR_t = FP_t / float(FP_t + TN_t)
        ROC[i, 0] = FPR_t
        TPR_t = TP_t / float(TP_t + FN_t)  # = Recall
        ROC[i, 1] = TPR_t
        print('TP_t', TP_t)
        print('TN_t', TN_t)
        print('FP_t', FP_t)
        print('FN_t', FN_t)
        print('FPR_t', FPR_t)
        print('TPR_t', TPR_t)
        Precision_Recall[i, 0] = TPR_t
        try:
            precision = TP_t / float(TP_t + FP_t)
        except:
            precision = 0
        print('precision', precision)
        Precision_Recall[i, 1] = precision
        average_precision += (TPR_t - TPR_previuos) * precision
        TPR_previuos = TPR_t
    average_precision = average_precision * -1
    AUC = 0.
    for i in range(max - 1):
        AUC += (ROC[i + 1, 0] - ROC[i, 0]) * (ROC[i + 1, 1] + ROC[i, 1])
    AUC *= -0.5
    roc_auc = metrics.auc(ROC[:, 0], ROC[:, 1])
    average_precision = metrics.auc(Precision_Recall[:, 0], Precision_Recall[:, 1])

    return Precision_Recall, average_precision, ROC, AUC


def sift_as_score_measurements(cadd_score_column, y):
    print('@', 'SIFT')
    CADD_Scores_df = pd.concat([cadd_score_column, y], axis=1)  # , ignore_index=True)
    CADD_Scores_df.columns = ['Pathogenicity_Score', 'tissue_specific_disease']
    print('CADD_Scores_df', CADD_Scores_df)
    # thresholds = np.linspace(101, 1, 101)
    # print(thresholds)
    max = 102
    ROC = np.zeros((max, 2))
    Precision_Recall = np.zeros((max, 2))
    average_precision = 0
    TPR_previuos = 0
    for i in range(-1, max - 1):
        t = i
        print('threshold: ', t)
        TP_t = len(CADD_Scores_df[(CADD_Scores_df['Pathogenicity_Score'] <= t) & (
                    CADD_Scores_df['tissue_specific_disease'] == True)])
        TN_t = len(CADD_Scores_df[(CADD_Scores_df['Pathogenicity_Score'] > t) & (
                    CADD_Scores_df['tissue_specific_disease'] == False)])
        FP_t = len(CADD_Scores_df[(CADD_Scores_df['Pathogenicity_Score'] <= t) & (
                    CADD_Scores_df['tissue_specific_disease'] == False)])
        FN_t = len(CADD_Scores_df[(CADD_Scores_df['Pathogenicity_Score'] > t) & (
                    CADD_Scores_df['tissue_specific_disease'] == True)])
        FPR_t = FP_t / float(FP_t + TN_t)
        ROC[i + 1, 0] = FPR_t
        TPR_t = TP_t / float(TP_t + FN_t)  # = Recall
        ROC[i + 1, 1] = TPR_t
        print('TP_t', TP_t)
        print('TN_t', TN_t)
        print('FP_t', FP_t)
        print('FN_t', FN_t)
        print('FPR_t', FPR_t)
        print('TPR_t', TPR_t)
        Precision_Recall[i + 1, 0] = TPR_t
        try:
            precision = TP_t / float(TP_t + FP_t)
        except:
            precision = 0
        print('precision', precision)
        Precision_Recall[i + 1, 1] = precision
        average_precision += (TPR_t - TPR_previuos) * precision
        TPR_previuos = TPR_t
    average_precision = average_precision
    AUC = 0.
    for i in range(max - 1):
        AUC += (ROC[i + 1, 0] - ROC[i, 0]) * (ROC[i + 1, 1] + ROC[i, 1])
    AUC *= 0.5
    roc_auc = metrics.auc(ROC[:, 1], ROC[:, 0])
    print('ROC')
    print(ROC)
    print('AUC')
    print(AUC)

    roc_auc = metrics.auc(ROC[:, 0], ROC[:, 1])
    average_precision = metrics.auc(Precision_Recall[:, 0], Precision_Recall[:, 1])

    return Precision_Recall, average_precision, ROC, AUC


import ast
from time import gmtime, strftime

"------------------------------ Train Model ---------------------------------------"

results_dict = {'Tissue': [], 'Data_set': [], 'ROC_AUC': [], 'PR_AUC': []}
best_parameters_dict = {}

y_columns = ['Heart - Left Ventricle_disease_causing', 'brain_disease_causing', 'Lung_disease_causing',
             'Muscle - Skeletal_disease_causing', 'Skin - Sun Exposed_disease_causing',
             'Adipose - Subcutaneous_disease_causing', 'Artery - Aorta_disease_causing',
             'Artery - Coronary_disease_causing', 'brain-0_disease_causing', 'Liver_disease_causing',
             'Nerve - Tibial_disease_causing', 'Colon - Sigmoid_disease_causing', 'kidney_disease_causing',
             'Heart - Atrial Appendage_disease_causing', 'Breast - Mammary Tissue_disease_causing',
             'Uterus_disease_causing', 'Adipose - Visceral_disease_causing',
             'Esophagus - Gastroesophageal Junction_disease_causing', 'Esophagus - Mucosa_disease_causing',
             'brain-1_disease_causing', 'Skin - Not Sun Exposed_disease_causing', 'Artery - Tibial_disease_causing',
             'Pituitary_disease_causing', 'Ovary_disease_causing', 'brain-3_disease_causing', 'Thyroid_disease_causing',
             'Testis_disease_causing', 'Whole Blood_disease_causing', 'brain-2_disease_causing']

y_columns_models = ['Heart - Left Ventricle_disease_causing', 'brain_disease_causing', 'Lung_disease_causing',
             'Muscle - Skeletal_disease_causing', 'brain-0_disease_causing', 'Liver_disease_causing',
             'Nerve - Tibial_disease_causing', 'kidney_disease_causing',
             'brain-1_disease_causing', 'Skin - Not Sun Exposed_disease_causing', 'brain-3_disease_causing',
             'Testis_disease_causing', 'Whole Blood_disease_causing', 'brain-2_disease_causing']

def compare_score_tissue(y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    tissue = y.replace("_disease_causing", "")
    try:
        pathogenic_proportion = Variants_data[y].value_counts(normalize=True)[True]
    except KeyError:
        pathogenic_proportion = 0
    Relevant_data = preprocessing_new(Variants_data, y_columns, y)
    Relevant_data.rename(columns=rename_dict, inplace=True)
    Model_input = Relevant_data.drop(columns=y)
    y_test = Relevant_data[y]
    features_model_path = root_directory + tissue + "_Features_dict.pkl"
    with open(features_model_path, 'rb') as handle:
        model_features_dict = pickle.load(handle)

    relevant_model_path = root_directory + tissue + '_RF_Model.pkl'
    print(relevant_model_path)
    with open(relevant_model_path, 'rb') as handle:
        model = pickle.load(handle)

    feature_order_path = root_directory + tissue + '_Features_Order.csv'
    Feature_Order = pd.read_csv(feature_order_path)
    feature_order_list = Feature_Order['0'].to_list()

    "--------------------- Deal with Missed Features ----------------------------------"

    model_features = [*model_features_dict]
    #print(model_features)
    missed_features_in_patient = [x for x in model_features if x not in list(Model_input)]
    missed_features_in_model = [x for x in list(Model_input) if x not in model_features]
    for missed_f in missed_features_in_patient:
        Model_input[missed_f] = model_features_dict[missed_f]

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    y_real = []
    y_proba = []
    x_test_list = []
    y_test_list = []
    y_pred = model.predict(Model_input[feature_order_list])  # predict the test data
    y_pred = pd.DataFrame(y_pred)
    predictions_proba = model.predict_proba(Model_input[feature_order_list])
    columns_merge = Variants_data[['#Chr', 'Pos', 'Ref', 'Alt', 'GeneID_y']]
    new_data = pd.DataFrame({'Predicted_probability': predictions_proba[:, 1], 'Label': y_test})

    pred_true = predictions_proba[:, 1]
    prec, recall, _ = precision_recall_curve(y_test, pred_true)  # pred[:, 1], pos_label=model.classes_[1]
    fpr, tpr, _ = roc_curve(y_test, pred_true)
    pr_auc1 = average_precision_score(y_test, pred_true)
    pr_auc1 = metrics.auc(recall, prec)
    roc_auc = metrics.auc(fpr, tpr)

    data_set = 'Full TRACE'
    results_dict['Tissue'].append(tissue)
    results_dict['Data_set'].append(data_set)
    results_dict['ROC_AUC'].append(roc_auc)
    results_dict['PR_AUC'].append(pr_auc1)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(roc_auc)
    y_real.append(y_test)
    y_proba.append(pred_true)
    x_test_list.append(Model_input[feature_order_list])
    y_test_list.append(y_test)
    Precision_Recall, average_precision, ROC, AUC = cadd_as_score_measurements(Scores_DF['PHRED'],
                                                                               Scores_DF[y])
    ax2.plot(Precision_Recall[:, 0], Precision_Recall[:, 1], color='#EE9322',
             label='%s (auPRC = %0.2f)' % ('CADD', average_precision), lw=3)
    ax1.plot(ROC[:, 0], ROC[:, 1], color='#EE9322', label='%s (auROC = %0.2f)' % ('CADD', AUC), lw=3)
    results_dict['Tissue'].append(tissue)
    results_dict['Data_set'].append('CADD')

    results_dict['ROC_AUC'].append(AUC)
    results_dict['PR_AUC'].append(average_precision)

    Precision_Recall, average_precision, ROC, AUC = cadd_as_score_measurements(Scores_DF['CAPICE'],
                                                                               Scores_DF[y])
    ax2.plot(Precision_Recall[:, 0], Precision_Recall[:, 1], color='#D83F31',
             label='%s (auPRC = %0.2f)' % ('CAPICE', average_precision), lw=3)
    ax1.plot(ROC[:, 0], ROC[:, 1], color='#D83F31', label='%s (auROC = %0.2f)' % ('CAPICE', AUC), lw=3)
    results_dict['Tissue'].append(tissue)
    results_dict['Data_set'].append('CAPICE')
    results_dict['ROC_AUC'].append(AUC)
    results_dict['PR_AUC'].append(average_precision)

    full_interp_list = tprs
    full_auc_list = aucs
    plot_mean_auc(full_interp_list, mean_fpr, full_auc_list, 'TRACEvar', '#219C90', ax1)
    ax1.plot([0, 1], [0, 1], 'r--', color='black', lw=3)
    ax1.set_xlabel('1-Specificity(False Positive Rate)')
    ax1.set_ylabel('Sensitivity(True Positive Rate)')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend([])
    fig.suptitle(tissue)
    y_real_trace = np.concatenate(y_real)
    y_proba_trace = np.concatenate(y_proba)
    precision_trace, recall_trace, _ = precision_recall_curve(y_real_trace, y_proba_trace)
    plt.plot(recall_trace, precision_trace, color='#219C90', label=r'TRACEvar (mean auPRC = %0.2f)' % (
        average_precision_score(y_real_trace, y_proba_trace)), lw=3, alpha=.8)
    ax2.axhline(y=pathogenic_proportion, color='black', linestyle='--',
                label=r'Pathogenic variants frequency = %0.3f' % (pathogenic_proportion), lw=3)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall curve')

    path = root_directory + tissue + '_Compare_Scores_Slim_Model_Docker.png'
    plt.savefig(path, dpi=300)
    plt.close()
    Results = pd.DataFrame.from_dict(results_dict)
    path = root_directory + tissue + '_Compare_Scores_Slim_Model_Docker.csv'
    Results.to_csv(path, index=None)
#
    return tissue, strftime("%Y-%m-%d %H:%M:%S", gmtime())

"------------------------------ Multi-processing ---------------------------------------"

import multiprocessing as mp


def driver_func_shap():
    PROCESSES = 17
    df_list = []

    with mp.Pool(PROCESSES) as pool:
        results = [pool.apply_async(compare_score_tissue, (y,)) for y in y_columns_models]
        for r in results:
            results_tuple = r.get(timeout=None)
            print('@', results_tuple[0], ' finished', results_tuple[1])


if __name__ == '__main__':
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    driver_func_shap()
