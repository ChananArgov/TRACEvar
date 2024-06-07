"""
This code aim to show the power of TRACEvar by POC examples.
For each example gene, the model train six different tissue models to predict the gene pathogenic variants.
"""

from sklearn.ensemble import GradientBoostingClassifier
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report
import pickle
import ast

"----------------------------------------- Load Data --------------------------------"

root_directory = 'Your relevant directory'
path = os.path.join(root_directory, 'full_dataset_2_stars_2023.csv')
Variants_data = pd.read_csv(path, engine='python')
print(Variants_data)

path = os.path.join(root_directory, 'new_parameters_file.csv')
Best_param = pd.read_csv(path, engine='python')
print(Best_param)

path = os.path.join(root_directory, 'Relevant_Columns_Names_Edited_2.csv')
Relevant_Cols_df = pd.read_csv(path)
overlap_cols = Relevant_Cols_df['Feature'].tolist()
print(overlap_cols)
rename_dict = dict(zip(overlap_cols, Relevant_Cols_df['Feature Name'].tolist()))
overlap_cols_names  = Relevant_Cols_df['Feature Name'].tolist()

"----------------------------------------- Data Preprocessing --------------------------------"

y_columns = Variants_data.columns[Variants_data.columns.str.contains(pat = 'disease_causing')].tolist()
cols = list(Variants_data)
non_relevant_columns = ['VariationID', 'OMIMs', 'Manifested_Tissues', '#Chr', 'Pos', 'ConsDetail', 'motifEName', 'FeatureID', 'GeneID_y', 'GeneName', 'CCDS', 'Intron', 'Exon', 'SIFTcat', 'PolyPhenCat', 'bStatistic', 'targetScan', 'dbscSNV-rf_score', 'oAA', 'Ref', 'nAA', 'Alt']# it will be good to replace oAA and nAA with blssuom64 matrix. What bStatistic doing?
non_relevant_columns = non_relevant_columns + y_columns
print(non_relevant_columns)

relevant_columns = [x for x in cols if (x not in non_relevant_columns) and (x in overlap_cols)]
# relevant_columns.append(y)
print(relevant_columns)
Relevant_data = Variants_data[relevant_columns]
print(Relevant_data)

def preprocessing_new(Relevant_Data):
    
    "---------------------- One Hot Columns -------------------------"
    
    one_hot_columns = ['Type', 'AnnoType', 'Consequence', 'Domain', 'Dst2SplType']

    one_hot = pd.get_dummies(Relevant_Data[one_hot_columns])
    Relevant_Data = Relevant_Data.drop(one_hot_columns, axis=1)
    Relevant_Data = Relevant_Data.join(one_hot)
    
    "---------------------- Missing Values Imputation ---------------"
    
    special_imputation_cols = {'SIFTval':1, 'GC':0.42, 'CpG':0.02, 'priPhCons':0.115, 'mamPhCons':0.079, 'verPhCons':0.094,'priPhyloP':-0.033, 'mamPhyloP':-0.038, 'verPhyloP':0.017, 'GerpN':1.91, 'GerpS':-0.2}
    
    for cl in special_imputation_cols:
        Relevant_Data[cl] = Relevant_Data[cl].fillna(special_imputation_cols[cl])
        
    Relevant_Data.fillna(0, inplace=True)
    
    return Relevant_Data

Relevant_data = preprocessing_new(Relevant_data)
Relevant_data.rename(columns=rename_dict, inplace=True)

"-----------------------------------------Choosing Gene Examples --------------------------------"

gene_name = 'AHDC1' # disease gene only in skin
tissues = ['Skin - Not Sun Exposed', 'Heart - Left Ventricle', 'brain',  'Whole Blood', 'kidney', 'Testis', 'Lung']#, 'Muscle - Skeletal'

# gene_name = 'CLN3' # disease gene only in brain
# tissues = ['Skin - Not Sun Exposed', 'Heart - Left Ventricle', 'brain',  'Whole Blood', 'kidney', 'Testis', 'Lung']#, 'Muscle - Skeletal'

relevant_y_cols = [t + '_disease_causing' for t in tissues]
print(relevant_y_cols)

X_train = Relevant_data[Variants_data['GeneName'] != gene_name]
X_test = Relevant_data[Variants_data['GeneName'] == gene_name]
pathogenicity_list = []

"----------------------------------------- Running Six Tissue-Specific Models --------------------------------"

for y in relevant_y_cols:
    tissue = tissues[relevant_y_cols.index(y)]
    print('-------------------', tissue, '--------------------------')
    best_parameters = Best_param['Best_Parameters'][(Best_param['Dataset'] == 'Full TRACE')&(Best_param['Tissue'] == tissue.strip())&(Best_param['ML_Model'] == 'GBM')].values[0]
    best_parameters = ast.literal_eval(best_parameters)
    model = GradientBoostingClassifier(**best_parameters)
    y_train = Variants_data[y][Variants_data['GeneName'] != gene_name]
    y_test = Variants_data[y][Variants_data['GeneName'] == gene_name]
    model.fit(X_train, y_train)
    predictions_proba = model.predict_proba(X_test)
    pred_true = predictions_proba[:, 1]
    print(pred_true)
    y_pred = model.predict(X_test)
    clr = classification_report(y_test, y_pred, output_dict=True)
    Pred_true_df = pd.DataFrame({tissue: pred_true})
    print(Pred_true_df)
    pathogenicity_list.append(Pred_true_df)
    print(clr)


Pathogenicity_df = pd.concat(pathogenicity_list, axis=1)
Pathogenicity_df['NewIndex'] = X_test.index
Pathogenicity_df.set_index('NewIndex', inplace=True)
print(Pathogenicity_df)

"----------------------------------------- Saving Results --------------------------------"


interesting_cols = ['VariationID', '#Chr', 'Pos', 'Ref', 'Alt', 'Type', 'Length', 'AnnoType', 'Consequence', 'GeneName', 'cDNApos', 'protPos', 'PHRED']
interesting_cols.extend(relevant_y_cols)
Pathogenicity_df = pd.concat([Variants_data[interesting_cols], Pathogenicity_df], axis=1, join='inner')
print(Pathogenicity_df)

path = os.path.join(root_directory, gene_name + '_Predictions_Example.csv')
Pathogenicity_df.to_csv(path)

