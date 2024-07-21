"""
This script calculates and plots the feature importance for the multi-tissue model (Fig. 5).
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
import shap

"---------------------------- Load Data -----------------------"

startTime = time.time()
path = os.path.join('..TRACEvar/Multi_tissue_model/Data', 'Transfer_Learning_Data_hg37.csv')
All_tissues_data= pd.read_csv(path)

path = os.path.join('../TRACEvar/Multi_tissue_model/Data', 'Transfer_Learninig_Relevant_Columns_Names_Edited.csv')
Relevant_Cols_df = pd.read_csv(path)
overlap_cols = Relevant_Cols_df['Feature'].tolist()
"---------------------------- Columns Names -----------------------"

rename_dict = dict(zip(overlap_cols, Relevant_Cols_df['Feature Name'].tolist()))
# print(rename_dict)
All_tissues_data.rename(columns=rename_dict, inplace=True)

cols_names = Relevant_Cols_df['Feature Name'].tolist()
# print(cols_names)

relevant_cols = [x for x in cols_names if (x not in  ['Unnamed: 0', 'Tissue', 'Pathogenic_Mutation', 'CADD | GeneID y', 'Segway'])]
# print(relevant_cols)
cadd_cols = [c for c in relevant_cols if 'CADD' in c]
tissue_cols = [c for c in relevant_cols if 'CADD' not in c]

print('cadd_cols', len(cadd_cols))
print('tissue_cols', len(tissue_cols), tissue_cols)

"---------------------------- Preprossecing Data --------------------"
duplicated_tissues = ['Skin - Sun Exposed', 'Heart - Atrial Appendage', 'brain-1', 'brain-0', 'brain-3', 'brain-2', 'Artery - Coronary', 'Artery - Tibial']
non_relevant_tissues = ['Adipose - Subcutaneous', 'Colon - Sigmoid', 'Breast - Mammary Tissue', 'Uterus', 'Adipose - Visceral', 'Esophagus - Gastroesophageal Junction', 'Esophagus - Mucosa', 'Thyroid', 'Artery - Aorta', 'Pituitary', 'Ovary', 'kidney']
duplicated_tissues = duplicated_tissues + non_relevant_tissues
Transfer_Data_set = All_tissues_data[~All_tissues_data['Tissue'].isin(duplicated_tissues)]

print(Transfer_Data_set)
tissues_list = ['brain', 'Heart - Left Ventricle', 'Skin - Not Sun Exposed','Testis', 'Whole Blood',
'Liver', 'Nerve - Tibial','Lung', 'Muscle - Skeletal']
print('tissues_list', len(tissues_list), tissues_list)

'-------------------- Synthetic Dataset --- ---------------------------'

Pathogenic = Transfer_Data_set[Transfer_Data_set['Pathogenic_Mutation'] == True]
counts = len(Pathogenic)
folds = 9
Non_pathogenic = Transfer_Data_set[Transfer_Data_set['Pathogenic_Mutation'] == False].sample(n=counts * folds, axis='index', random_state=1234)
Synthetic_Dataset = pd.concat([Pathogenic, Non_pathogenic])
print(Synthetic_Dataset)

'-------------------- Train Model--- ---------------------------'

model = GradientBoostingClassifier(random_state=1234)#, min_samples_leaf=100, n_estimators=10
model.fit(Synthetic_Dataset[relevant_cols], Synthetic_Dataset['Pathogenic_Mutation'])
print('@ Model trained',  time.time() - startTime)

'-------------------- SHAP Importance ---------------------------'

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Synthetic_Dataset[relevant_cols])
vals = np.abs(shap_values).mean(0)

print('@ Explainer created',  time.time() - startTime)

Feature_importance = pd.DataFrame(list(zip(Synthetic_Dataset[relevant_cols].columns, vals)),columns=['col_name', 'feature_importance_vals'])
Feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
#path = os.path.join('/gpfs0/estiyl/users/shneyour/TRACEvar/Multi_tissue_model/Results', 'Transfer_learning_hg37_SHAP_Importance_Slim.csv')
#Feature_importance.to_csv(path, index=False)
print('@ Feature Importance Saved',  time.time() - startTime)

shap.summary_plot(shap_values, Synthetic_Dataset[relevant_cols], show=False)
print('@ Plot created', startTime - time.time())
plt.tight_layout()
path = os.path.join('../TRACEvar/Multi_tissue_model/Results', 'Transfer_learning_hg37_SHAP_dot2.pdf')
plt.savefig(path)
plt.close()



print('@ Finished',  time.time() - startTime)

shap.summary_plot(shap_values, Synthetic_Dataset[relevant_cols], show=False, plot_type='bar')
path = os.path.join('../TRACEvar/Multi_tissue_model/Results', 'Transfer_learning_hg37_SHAP_bar2.pdf')
plt.tight_layout()

plt.savefig(path)
plt.close()

shap.summary_plot(shap_values, Synthetic_Dataset[relevant_cols], show=False, plot_type='violin')
path = os.path.join('../TRACEvar/Multi_tissue_model/Results', 'Transfer_learning_hg37_SHAP_violin2.pdf')
plt.tight_layout()
plt.savefig(path)
plt.close()
