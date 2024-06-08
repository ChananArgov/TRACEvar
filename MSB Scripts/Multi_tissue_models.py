"""
The aim of this script is to create and estimate performance of multi-tissue model for variant prediction
"""


import time
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
import multiprocessing as mp
from time import gmtime, strftime

root_directory = 'Your relevant directory'

"---------------------------- Load Data -----------------------"

startTime = time.time()
path = os.path.join(root_directory, 'Transfer_Learning_Data_hg37.csv')
All_tissues_data= pd.read_csv(path)

path = os.path.join(root_directory, 'Transfer_Learninig_Relevant_Columns_Names_Edited.csv')
Relevant_Cols_df = pd.read_csv(path)
overlap_cols = Relevant_Cols_df['Feature'].tolist()

rename_dict = dict(zip(overlap_cols, Relevant_Cols_df['Feature Name'].tolist()))
# print(rename_dict)
All_tissues_data.rename(columns=rename_dict, inplace=True)

cols_names = Relevant_Cols_df['Feature Name'].tolist()
print(cols_names)

"---------------------------- Preprossecing Data --------------------"
duplicated_tissues = ['Skin - Sun Exposed', 'Heart - Atrial Appendage', 'brain-1', 'brain-0', 'brain-3', 'brain-2', 'Artery - Coronary', 'Artery - Tibial']
non_relevant_tissues = ['Adipose - Subcutaneous', 'Colon - Sigmoid', 'Breast - Mammary Tissue', 'Uterus', 'Adipose - Visceral', 'Esophagus - Gastroesophageal Junction', 'Esophagus - Mucosa', 'Thyroid', 'Artery - Aorta', 'Pituitary', 'Ovary', 'kidney']
duplicated_tissues = duplicated_tissues + non_relevant_tissues
Transfer_Data_set = All_tissues_data[~All_tissues_data['Tissue'].isin(duplicated_tissues)]

print(Transfer_Data_set)
tissues_list = ['brain', 'Heart - Left Ventricle', 'Skin - Not Sun Exposed','Testis', 'Whole Blood',
'Liver', 'Nerve - Tibial','Lung', 'Muscle - Skeletal']
print('tissues_list', len(tissues_list), tissues_list)

"---------------------------- Functions ----------------------------------"

def cadd_as_score_measurements(cadd_score_column, y):
    CADD_Scores_df = pd.concat([cadd_score_column, y], axis=1)
    CADD_Scores_df.columns = ['PHRED', 'tissue_specific_disease']
    print('CADD_Scores_df', CADD_Scores_df)
    thresholds = np.linspace(100, 1, 100)
    # print(thresholds)
    ROC = np.zeros((100, 2))
    Precision_Recall = np.zeros((100, 2))
    average_precision = 0
    TPR_previuos = 0

    for i in range(100):
        t = thresholds[i]
        # print('threshold: ', t)
        # Classifier / label agree and disagreements for current threshold.
        TP_t = len(CADD_Scores_df[(CADD_Scores_df['PHRED'] > t) & (CADD_Scores_df['tissue_specific_disease'] == True)])
        TN_t = len(CADD_Scores_df[(CADD_Scores_df['PHRED'] <= t) & (CADD_Scores_df['tissue_specific_disease'] == False)])
        FP_t = len(CADD_Scores_df[(CADD_Scores_df['PHRED'] > t) & (CADD_Scores_df['tissue_specific_disease'] == False)])
        FN_t = len(CADD_Scores_df[(CADD_Scores_df['PHRED'] <= t) & (CADD_Scores_df['tissue_specific_disease'] == True)])
        # Compute false positive rate for current threshold.
        FPR_t = FP_t / float(FP_t + TN_t)
        ROC[i, 0] = FPR_t

        # Compute true  positive rate for current threshold.
        TPR_t = TP_t / float(TP_t + FN_t)  # = Recall
        ROC[i, 1] = TPR_t
        # print(TP_t, TN_t, FP_t, FN_t, FPR_t, TPR_t)

        Precision_Recall[i, 0] = TPR_t
        try:
            precision = TP_t / float(TP_t + FP_t)
        except:
            precision = 0
        Precision_Recall[i, 1] = precision

        average_precision += (TPR_t - TPR_previuos) * precision
        TPR_previuos = TPR_t
    average_precision = average_precision
    AUC = 0.
    for i in range(99):
        AUC += (ROC[i + 1, 0] - ROC[i, 0]) * (ROC[i + 1, 1] + ROC[i, 1])
    AUC *= 0.5
    
    roc_auc = metrics.auc(ROC[:,0], ROC[:,1])
    average_precision = metrics.auc(Precision_Recall[:,0], Precision_Recall[:,1])
    
    return Precision_Recall, average_precision, ROC, AUC

"---------------------------- Transfer Learning ---------------------------"

relevant_cols = [x for x in cols_names if (x not in  ['Unnamed: 0', 'Tissue', 'Pathogenic_Mutation', 'CADD | GeneID y', 'Segway'])]
print(relevant_cols)
colors = ['royalblue', 'indianred', 'mediumseagreen', 'mediumorchid', '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

from time import gmtime, strftime
# tissues_list = ['Testis', 'Whole Blood', 'brain']

def transfear_learning_tissue(tissue):
    results_dict = {'Tissue':[], 'Method':[], 'PR_AUC':[], 'ROC_AUC':[]}
    
    print('------------------', tissue, '-----------------------')

    pathogenic_proportion = Transfer_Data_set[Transfer_Data_set['Tissue'] == tissue]['Pathogenic_Mutation'].value_counts(normalize=True)[True]
    print('Pathogenic_proportion: ', pathogenic_proportion)

    Tissue_data = Transfer_Data_set[Transfer_Data_set['Tissue'] == tissue]
    Other_tissues_data = Transfer_Data_set[Transfer_Data_set['Tissue'] != tissue]
    model = GradientBoostingClassifier(random_state=1234)
    model.fit(Other_tissues_data[relevant_cols], Other_tissues_data['Pathogenic_Mutation'])

    y_pred = model.predict(Tissue_data[relevant_cols])  # predict the test data
    predictions_proba = model.predict_proba(Tissue_data[relevant_cols])
    pred_true = predictions_proba[:, 1]
    prec, recall, _ = precision_recall_curve(Tissue_data['Pathogenic_Mutation'], pred_true)  # pred[:, 1], pos_label=model.classes_[1]
    fpr, tpr, _ = roc_curve(Tissue_data['Pathogenic_Mutation'], pred_true)
    probability_df = pd.DataFrame()
    probability_df['Label'] = Tissue_data['Pathogenic_Mutation']
    probability_df['Predicted_probability'] = pred_true
    probability_df.to_csv(root_directory + tissue + '.csv', index=None)
    pr_auc1 = metrics.auc(recall, prec)#average_precision_score(Tissue_data['Pathogenic_Mutation'], pred_true)
    roc_auc = metrics.auc(fpr, tpr)
    print('ROC = ', roc_auc)
    print('PR_AUC = ', pr_auc1)
    print('\n')

    # ax2.plot(recall, prec, label='%s PR-AUC = %0.2f' % (tissue, pr_auc1))
    # ax1.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (tissue, roc_auc))

    ax2.plot(recall, prec, label='%s TRACEvar: Proportion = %0.2f, PR-AUC (area = %0.2f)' % (tissue, pathogenic_proportion, pr_auc1),
             color=colors[tissues_list.index(tissue)])
    ax1.plot(fpr, tpr, label='%s TRACEvar: Proportion = %0.2f, ROC (area = %0.2f)' % (tissue, pathogenic_proportion, roc_auc),
             color=colors[tissues_list.index(tissue)])
    "--------------------------- CADD Comparision -------------------------------"

    Tissue_data['PHRED_percent'] = Tissue_data['CADD | Pathogenic score (PHRED)'].apply(lambda x: x * 0.01)
    Precision_Recall, average_precision, ROC, AUC = cadd_as_score_measurements(Tissue_data['CADD | Pathogenic score (PHRED)'], Tissue_data['Pathogenic_Mutation'])
    ax2.plot(Precision_Recall[:, 0], Precision_Recall[:, 1], 'r--', color=colors[tissues_list.index(tissue)],
             label='%s CADD: Proportion = %0.2f, PR-AUC (area = %0.2f)' % (tissue, pathogenic_proportion, average_precision))
    ax1.plot(ROC[:, 0], ROC[:, 1], 'r--', color=colors[tissues_list.index(tissue)],
             label='%s CADD: Proportion = %0.2f, ROC (area = %0.2f)' % (tissue, pathogenic_proportion, AUC))

  
    
    "---------------------------- Results Record ----------------------------"
    
    results_dict['Tissue'].append(tissue)
    results_dict['Method'].append('TRACEvar')
    results_dict['PR_AUC'].append(pr_auc1)
    results_dict['ROC_AUC'].append(roc_auc)
    
    results_dict['Tissue'].append(tissue)
    results_dict['Method'].append('CADD')
    results_dict['PR_AUC'].append(average_precision)
    results_dict['ROC_AUC'].append(AUC)
    
    results_dict['Tissue'].append(tissue)
    results_dict['Method'].append('Pathogenic_proportion')
    results_dict['PR_AUC'].append(pathogenic_proportion)
    results_dict['ROC_AUC'].append(0.5)
    
    Results_df = pd.DataFrame.from_dict(results_dict, orient='columns')
    
    return tissue, strftime("%Y-%m-%d %H:%M:%S", gmtime()), Results_df


"---------------------------- Multi-processing -----------------------"


def driver_func_shap():
    PROCESSES = 22
    results_df_list = []

    with mp.Pool(PROCESSES) as pool:
        results = [pool.apply_async(transfear_learning_tissue, (tissue,)) for tissue in tissues_list]

        for r in results:
            
            results_tuple = r.get(timeout=None)
            print('@', results_tuple[0], ' finished', results_tuple[1])
            results_df_list.append(results_tuple[2])
            
    All_Results = pd.concat(results_df_list)
    path = os.path.join(root_directory, 'Transfer_learning_hg37_ALL_Results_New.csv')
    All_Results.to_csv(path)

    ax1.plot([0, 1], [0, 1], 'r--')
    ax1.set_xlabel('1-Specificity(False Positive Rate)')
    ax1.set_ylabel('Sensitivity(True Positive Rate)')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right", fontsize='small')

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall curve')
    ax2.legend(loc="lower right", fontsize='small')
    fig.suptitle('Transfer_learning Full')
    path = os.path.join(root_directory, 'Transfer_learning_hg37_ALL_New.pdf')
    plt.savefig(path)
    
if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(50, 30))

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    driver_func_shap()
    
