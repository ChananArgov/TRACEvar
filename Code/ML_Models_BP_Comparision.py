



import ast
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
import pickle
from scipy import interp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import catboost as ctb
from sklearn.neural_network import MLPClassifier

"------------------------------------------  Load Data ----------------------------------"

path = os.path.join('..', '..', 'Data', 'Runing_ML_data', 'New_Verssion_data', 'hg37_Dataset',  'Full_Slim_Dataset_hg37-v1.6.csv')
Variants_data = pd.read_csv(path, engine='python')
print(Variants_data)

filename = os.path.join('..', '..', 'Data', 'TRACE_data', 'trace_features.pkl')
infile = open(filename,'rb')
trace_features = pickle.load(infile)
infile.close()

path = os.path.join('..', '..', 'Final_Results', 'Multiple_Models', 'Best_Parameters',  'Best_Parameters_Multiple_ML_0.csv')
Best_param = pd.read_csv(path)
models_list = Best_param['ML_Model'].unique()
models_dict = {'xgboost': XGBClassifier, 'SVM': SVC, 'GBM': GradientBoostingClassifier, 'Random Forest': RandomForestClassifier, 'AdaBoost': AdaBoostClassifier, 'catboost': ctb.CatBoostClassifier, 'Neural Network': MLPClassifier}
"----------------------------------- Functions ------------------------------------------"

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
        if fraction >= threshold or fold > (1/threshold - 1):
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
    Shafeled_df = Folds_df.sample(frac=1, random_state= 1234).reset_index(drop=True)
    inds = Shafeled_df.index
    new_fold = [None]*len(Folds_df)
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


def finding_hyperparameter(X_data, Y_data, pathogenic_proportion, tissue, dataset):#, preprocessor):
    models = [{'label': 'xgboost',
               'model': XGBClassifier(learning_rate=0.1, n_estimators=100, objective='binary:logistic', verbosity=0,
                                      nthread=1, random_state=1234)
                  , 'params': {
            'eta': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
            'min_child_weight': [1, 3, 5, 10],
            'gamma': [0, 0.5, 1, 1.5, 2, 2.5, 5],
            'subsample': [0.6, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.2, 0.3, 0.4, 0.6, 0.8, 1],
            'max_depth': [3, 4, 5, 6, 7, 10, 15],
            'alpha': [1, 2, 3, 4], 'random_state':[1234],
        'learning_rate': [0.08, 0.06, 0.04, 0.02, 0.01],
        'scale_pos_weight': [1/pathogenic_proportion, 2/pathogenic_proportion, 1000, 100]}},

              {'label': 'SVM', 'model': SVC(C=1), 'params': {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000], "random_state": [1234]}},

              {'label': 'GBM', 'model': GradientBoostingClassifier(), 'params': {'learning_rate':[0.15, 0.1, 0.05, 0.01, 0.005, 0.001], 'n_estimators':[100, 250, 500, 750, 1000, 1250, 1500, 1750], 'min_samples_split':[2, 4, 6, 8, 10, 20, 40, 60, 100], 'min_samples_leaf':[1, 3, 5, 7, 9], 'max_features':[2, 3, 4, 5, 6, 7], "random_state": [1234]}},

              {'label': 'Random Forest', 'model': RandomForestClassifier(),
               'params': {
                   'bootstrap': [True, False],
                   'max_depth': [10, 20, 30, 40, 50, None],
                   'min_samples_leaf': [1, 2, 3, 4, 5, 10],
                   'min_samples_split': [2, 5, 10],
                   'n_estimators': [100, 150, 200, 300, 400, 600, 2000], 'random_state':[1234]}},
              {'label': 'AdaBoost', 'model': AdaBoostClassifier(),
               'params': {
                   'n_estimators': [50, 100, 200, 500, 1000],
                   'learning_rate': [0.001, 0.1, 0.2, 0.5, 1.0]}},
              {'label': 'catboost', 'model': ctb.CatBoostClassifier(),
               'params': {
                   'learning_rate': [0.001, 0.01, 0.1, 0.5],
                   'depth': [4, 6, 10],
                   'l2_leaf_reg': [1, 3, 5, 7, 9]}},
              {'label': 'Neural Network', 'model': MLPClassifier(),
               'params': {
                   'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                   'activation': ['tanh', 'relu'],
                   'solver': ['sgd', 'adam'],
                   'alpha': [0.0001, 0.1, 0.05, 1],
                   'learning_rate': ['constant', 'adaptive']}}
              ]

    best_parameters_dict = {'Tissue':[], 'Dataset': [], 'ML_Model':[], 'Best_Parameters':[]}
    for m in models:
        # m = models[1]
        print('------------', m['label'], '---------------')

        folds = 3
        param_comb = 15

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

        random_search = RandomizedSearchCV(m['model'], param_distributions=m['params'], n_iter=param_comb,
                                           scoring='average_precision', n_jobs=4, cv=skf.split(X_data, Y_data), verbose=3,
                                           random_state=1234)

        random_search.fit(X_data, Y_data)

        print(random_search.best_params_)

        best_parameters_dict['Tissue'].append(tissue)
        best_parameters_dict['Dataset'].append(data_set)
        best_parameters_dict['ML_Model'].append(m['label'])
        best_parameters_dict['Best_Parameters'].append(random_search.best_params_)

    Best_parameters_df = pd.DataFrame.from_dict(best_parameters_dict)
    print(Best_parameters_df)
    return Best_parameters_df


y_columns = Variants_data.columns[Variants_data.columns.str.contains(pat = 'disease_causing')].tolist()
print(y_columns)
cols = Variants_data.columns

def prepare_data(Variants_data, y_columns, y):
    non_relevant_columns = ['VariationID', 'OMIMs', 'Manifested_Tissues', '#Chr', 'Pos', 'ConsDetail', 'motifEName', 'FeatureID', 'GeneName', 'CCDS', 'Intron', 'Exon', 'SIFTcat', 'PolyPhenCat', 'bStatistic', 'targetScan', 'dbscSNV-rf_score', 'oAA', 'Ref', 'nAA', 'Alt']# it will be good to replace oAA and nAA with blssuom64 matrix. What bStatistic doing?
    non_relevant_columns = non_relevant_columns + y_columns
    print(non_relevant_columns)

    relevant_columns = [x for x in cols if x not in non_relevant_columns]
    relevant_columns.append(y)
    print(relevant_columns)
    Relevant_data = Variants_data[relevant_columns]
    print(Relevant_data)

    one_hot_columns = ['Type', 'AnnoType', 'Consequence', 'Domain', 'Dst2SplType', 'EnsembleRegulatoryFeature']

    # Get one hot encoding of columns B
    one_hot = pd.get_dummies(Relevant_data[one_hot_columns])
    # Drop column B as it is now encoded
    Relevant_data = Relevant_data.drop(one_hot_columns, axis = 1)
    # Join the encoded df
    Relevant_data = Relevant_data.join(one_hot)
    print(Relevant_data)
    cHmm_columns = Variants_data.columns[Variants_data.columns.str.contains(pat = 'cHmm_E')].tolist()
    fill_zero_columns = ['motifECount', 'motifEHIPos', 'motifEScoreChng', 'mirSVR-Score', 'mirSVR-E', 'mirSVR-Aln', 'tOverlapMotifs', 'motifDist'] + cHmm_columns # motifs with high number of nan 97%
    Relevant_data[fill_zero_columns] = Relevant_data[fill_zero_columns].fillna(value=0)
    fill_common_columns = ['cDNApos', 'relcDNApos', 'CDSpos', 'relCDSpos', 'protPos', 'relProtPos', 'Dst2Splice', 'SIFTval', 'PolyPhenVal', 'GerpRS', 'GerpRSpval', 'GerpN', 'GerpS', 'all Enc', 'Grantham', 'All SpliceAI', 'All MMSp', 'Dist2Mutation', 'All 00bp', 'dbscSNV', 'RemapOverlapTF', 'RemapOverlapCL', 'Trace Features'] # Locations, is this right?
    for cl in list(Relevant_data):
        Relevant_data[cl] = Relevant_data[cl].fillna(Relevant_data[cl].value_counts().idxmax())
    return Relevant_data

def plot_mean_auc(tprs, mean_fpr, aucs, dataset, color, ax):

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color=color,
             label= dataset + r' Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2,
                     label=dataset + r' $\pm$ 1 std. dev.')


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
    roc_auc = metrics.auc(ROC[:,1], ROC[:,0])
    return Precision_Recall, average_precision, ROC, AUC

"------------------------------ Train Model ---------------------------------------"
results_dict = {'Tissue':[], 'Data_set':[], 'ML_model':[], 'Fold':[], 'ROC_AUC':[], 'PR_AUC':[]}
best_parameters_dict = {}
best_params_list = []
relevant_models = ['xgboost', 'Random Forest', 'GBM', 'AdaBoost', 'SVM']# 'catboost',

for y in y_columns:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    print("------------- ", y, " ------------------")
    pathogenic_proportion = Variants_data[y].value_counts(normalize=True)[True]
    print('Pathogenic_proportion', pathogenic_proportion)
    Relevant_data = prepare_data(Variants_data, y_columns, y)
    Genes_fraction = Variants_data['GeneID_y'][Variants_data[y] == True].value_counts(normalize=True)
    Genes_fraction, Genes_as_factor = split_data_on_genes(Variants_data, y)
    Folds_df = find_fold(Genes_fraction)
    print(Folds_df)
    Shafeled_Fold_df = random_split(Folds_df)

    folds_list = list(Shafeled_Fold_df['New_Fold'].unique())
    folds_list = [int(x) for x in folds_list if pd.notnull(x)]
    print('folds_list', folds_list)
    # break

    tprs_dict = {'CADD': {model_name: [] for model_name in relevant_models}, 'Full Trace':{model_name: [] for model_name in relevant_models}}
    aucs_dict = {'CADD':{model_name: [] for model_name in relevant_models}, 'Full Trace':{model_name: [] for model_name in relevant_models}}

    y_real_dict = {'CADD': {model_name: [] for model_name in relevant_models},
                 'Full Trace': {model_name: [] for model_name in relevant_models}}
    y_proba_dict = {'CADD': {model_name: [] for model_name in relevant_models},
                 'Full Trace': {model_name: [] for model_name in relevant_models}}

    mean_fpr = np.linspace(0, 1, 100)
    y_real = []
    y_proba = []

    x_test_list = []
    y_test_list = []
    num_genes = Variants_data['GeneID_y'][Variants_data[y]==True].nunique()
    if num_genes >= 20:
        for fold in folds_list:
            print(" Fold: ", fold)

            test_genes = Shafeled_Fold_df['GeneID'][Shafeled_Fold_df['New_Fold'] == fold].tolist()
            print('test_genes', test_genes)
            train_genes = Shafeled_Fold_df['GeneID'][Shafeled_Fold_df['New_Fold'] != fold].tolist()
            print('train_genes', train_genes)
            benign_genes = Variants_data['GeneID_y'][Variants_data[y] == False].tolist()
            print('benign_genes', len(benign_genes))

            Benign_data = Relevant_data[Relevant_data[y] == False]
            Pathogenic_test = Relevant_data[(Relevant_data[y] == True) & (Relevant_data['GeneID_y'].isin(test_genes))]
            Pathogenic_train = Relevant_data[(Relevant_data[y] == True) & (Relevant_data['GeneID_y'].isin(train_genes))]

            test_len =  len(Pathogenic_test)
            train_len = len(Pathogenic_train)
            test_fraction = test_len/(test_len+train_len)
            train_fraction = train_len/(test_len+train_len)
            print('test_fraction: ', test_fraction)
            print('train_fraction: ', train_fraction)


            X_train1, X_test1, y_train1, y_test1 = train_test_split(Benign_data, Benign_data[y], test_size=test_fraction, random_state=fold)

            X_train1 = X_train1[~X_train1['GeneID_y'].isin(test_genes)]
            X_test1 = X_test1[~X_test1['GeneID_y'].isin(train_genes)]
            y_train1 = y_train1[X_train1[~X_train1['GeneID_y'].isin(test_genes)].index]
            y_test1 = y_test1[X_test1[~X_test1['GeneID_y'].isin(train_genes)].index]

            relevant_cols2 = list(Pathogenic_train)
            relevant_cols2 = [x for x in relevant_cols2 if x != y and x != 'GeneID_y']
            print('relevant_cols2', relevant_cols2)

            tissue = y.replace("_disease_causing", "")
            for data_set in ['CADD', 'Full Trace']:

                if data_set == 'CADD':
                    relevant_cols = [x for x in relevant_cols2 if x not in trace_features]
                else:
                    relevant_cols = [x for x in relevant_cols2]

                X_train = pd.concat([X_train1[relevant_cols], Pathogenic_train[relevant_cols]])
                X_test = pd.concat([X_test1[relevant_cols], Pathogenic_test[relevant_cols]])
                y_train = pd.concat([y_train1, Pathogenic_train[y]])
                y_test = pd.concat([y_test1, Pathogenic_test[y]])

                X_train = X_train.sort_index()
                X_test = X_test.sort_index()
                y_train = y_train.sort_index()
                y_test = y_test.sort_index()

                print(y_test)

                for model_name in relevant_models:#['xgboost', 'Random Forest', 'GBM', 'AdaBoost', 'SVM', 'Neural Network']

                    print('-----------------', model_name, tissue, data_set, '------------------')
                    best_parameters = Best_param['Best_Parameters'][(Best_param['Dataset'] == data_set)&(Best_param['Tissue'] == tissue)&(Best_param['ML_Model'] == model_name)].values[0]
                    print(type(best_parameters), best_parameters)

                    best_parameters = ast.literal_eval(best_parameters)
                    if model_name == 'SVM':
                        best_parameters['probability'] = True
                    model = models_dict[model_name]
                    model = model(**best_parameters)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)  # predict the test data
                    y_pred = pd.DataFrame(y_pred)
                    predictions_proba = model.predict_proba(X_test)

                    pred_true = predictions_proba[:, 1]
                    clr = classification_report(y_test, y_pred, output_dict=True)
                    print(clr)
                    precision = clr['True']['precision']
                    recall_1 = clr['True']['recall']
                    f1_score = clr['True']['f1-score']
                    print('@@@ ','precision:', precision, 'recall: ', recall_1, 'f1_score: ', f1_score)

                    prec, recall, _ = precision_recall_curve(y_test, pred_true)  # pred[:, 1], pos_label=model.classes_[1]
                    fpr, tpr, _ = roc_curve(y_test, pred_true)
                    pr_auc1 = average_precision_score(y_test, pred_true)
                    roc_auc = metrics.auc(fpr, tpr)

                    results_dict['Tissue'].append(tissue)
                    results_dict['Data_set'].append(data_set)
                    results_dict['ML_model'].append(model_name)
                    results_dict['Fold'].append(fold)
                    results_dict['ROC_AUC'].append(roc_auc)
                    results_dict['PR_AUC'].append(pr_auc1)

                    tprs_dict[data_set][model_name].append(interp(mean_fpr, fpr, tpr))
                    tprs_dict[data_set][model_name][-1][0] = 0

                    aucs_dict[data_set][model_name].append(roc_auc)

                    y_real_dict[data_set][model_name].append(y_test)
                    y_proba_dict[data_set][model_name].append(pred_true)


                    y_real.append(y_test)
                    y_proba.append(pred_true)

                    x_test_list.append(X_test)
                    y_test_list.append(y_test)
                # break
    colors = {'CADD':['darkred', 'tomato', 'lightsalmon', 'red', 'sandybrown', 'chocolate'], 'Full Trace': ['darkblue', 'royalblue', 'mediumblue', 'blue', 'cadetblue', 'teal']}

    for data_set in ['CADD', 'Full Trace']:#['CADD', 'Full Trace']:
        c = 0
        for model_name in relevant_models:
            print(c)
            color = colors[data_set][c]
            c += 1
            plot_mean_auc(tprs_dict[data_set][model_name], mean_fpr, aucs_dict[data_set][model_name], data_set + ' ' + model_name, color, ax1)
            y_real_model_dataset = np.concatenate(y_real_dict[data_set][model_name])
            y_proba_model_dataset = np.concatenate(y_proba_dict[data_set][model_name])
            precision_model_dataset, recall_model_dataset, _ = precision_recall_curve(y_real_model_dataset, y_proba_model_dataset)
            plt.plot(recall_model_dataset, precision_model_dataset, label=r'%s Mean Precision-Recall (AUC = %0.2f)' % (data_set + ' ' + model_name , average_precision_score(y_real_model_dataset, y_proba_model_dataset)), lw=2, alpha=.8, color=color)#color='grey',

    "----------------------------------- CADD as Score --------------------------"

    Precision_Recall, average_precision, ROC, AUC = cadd_as_score_measurements(Variants_data['PHRED'], Variants_data[y])
    ax2.plot(Precision_Recall[:, 0], Precision_Recall[:, 1], color='pink', label='%s Precision-Recall (AUC = %0.2f)' % ('CADD_Score', average_precision))
    ax1.plot(ROC[:, 0], ROC[:, 1], color='pink', label='%s ROC (area = %0.2f)' % ('CADD_Score', AUC))

    fig.suptitle(tissue)
    ax1.plot([0, 1], [0, 1], 'r--')
    ax1.set_xlabel('1-Specificity(False Positive Rate)')
    ax1.set_ylabel('Sensitivity(True Positive Rate)')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right", fontsize='small')
    ax2.axhline(y=pathogenic_proportion, color='red', linestyle='--',
                label=r'Pathogenic variants frequency = %0.3f' % (pathogenic_proportion))
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall curve')
    ax2.legend(loc="lower right", fontsize='small')
    path = os.path.join('..', '..', 'Final_Results', 'Multiple_Models', 'AUCs', tissue + '_Multiple_Model_AUC_TRY.pdf')
    plt.savefig(path)
    plt.close()
print(results_dict)
Results = pd.DataFrame.from_dict(results_dict)
print(Results)


path = os.path.join('..', '..', 'Final_Results', 'Multiple_Models', 'Comparison_Results', 'Multiple_Models_Results_TRY.csv')
Results.to_csv(path, index=False)
