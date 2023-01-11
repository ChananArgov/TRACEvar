"""
The aim of this script is to create TRACEvar models and shap explainers.
To this end, you need first to place the files bellow in the 'Data' folder:
1. TRACEvar features (Full_Slim_Dataset_hg37-v1.6.csv).
2. TRACE file (df_complete_dataset.csv).
3. Best parameters for ML models file (Best_Parameters_New_17.csv).
4. Columns names file (Relevant_Columns_Names_Edited_2).
"""

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap  # model interpretation package
pd.options.mode.chained_assignment = None
import pickle
import multiprocessing as mp
from time import gmtime, strftime
import ast

"------------------------------------------  Load Data ----------------------------------"

path = '../Data/Full_Slim_Dataset_hg37-v1.6.csv'
Variants_data = pd.read_csv(path, engine='python')#low_memory=False,

path = '../Data/Best_Parameters_New_17.csv'
Best_param = pd.read_csv(path, engine='python')#low_memory=False,
print(Best_param)

"------------------------------------------  Relevant cols ----------------------------------"

path = '../Data/Relevant_Columns_Names_Edited_2.csv'
Relevant_Cols_df = pd.read_csv(path)
overlap_cols = Relevant_Cols_df['Feature'].tolist()
rename_dict = dict(zip(overlap_cols, Relevant_Cols_df['Feature Name'].tolist()))

"----------------------------------- Functions ------------------------------------------"
cols = overlap_cols

def preprocessing_new(Variants_data, y_columns, y):

    "---------------------- Relevant Columns -------------------------"

    non_relevant_columns = ['VariationID', 'OMIMs', 'Manifested_Tissues', '#Chr', 'Pos', 'ConsDetail', 'motifEName', 'FeatureID', 'GeneName', 'CCDS', 'Intron', 'Exon', 'SIFTcat', 'PolyPhenCat', 'bStatistic', 'targetScan', 'dbscSNV-rf_score', 'oAA', 'Ref', 'nAA', 'Alt', 'Segway']# it will be good to replace oAA and nAA with blssuom64 matrix. What bStatistic doing?
    non_relevant_columns = non_relevant_columns + y_columns
    relevant_columns = [x for x in cols if (x not in non_relevant_columns) and (x in list(Variants_data))]
    relevant_columns.append(y)

    Relevant_Data = Variants_data[relevant_columns]

    
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

"------------------------------ Main ---------------------------------------"

results_dict = {'Tissue':[], 'Data_set':[], 'Fold':[], 'ROC_AUC':[], 'PR_AUC':[]}
best_parameters_dict = {}

y_columns = [  'brain_disease_causing', 'Heart - Left Ventricle_disease_causing', 'brain-1_disease_causing', 'Skin - Not Sun Exposed_disease_causing', 'Pituitary_disease_causing', 'Ovary_disease_causing',
              'brain-3_disease_causing', 'Testis_disease_causing', 'Whole Blood_disease_causing', 'brain-2_disease_causing',
                 'brain-0_disease_causing', 'Liver_disease_causing', 'Nerve - Tibial_disease_causing', 'kidney_disease_causing','Lung_disease_causing', 'Muscle - Skeletal_disease_causing','Artery - Aorta_disease_causing']

def create_model_and_shap(y):

    print("------------- ", y, " ------------------")
    pathogenic_proportion = Variants_data[y].value_counts(normalize=True)[True]
    print('Pathogenic_proportion', pathogenic_proportion)

    Relevant_data = preprocessing_new(Variants_data, y_columns, y)
    Relevant_data.rename(columns=rename_dict, inplace=True)
    relevant_cols2 = list(Relevant_data)
    relevant_cols = [x for x in relevant_cols2 if x != y and x != 'CADD | GeneID y' and x not in y_columns and x != 'CADD | Variant location (CDSpos)']

    tissue = y.replace("_disease_causing", "")
    path = '../Trained Models/' + tissue + '_Features_Order.csv'
    Features_Order = pd.DataFrame(relevant_cols)
    Features_Order.to_csv(path)

    features_dict = {feature: Relevant_data[feature].value_counts().idxmax() for feature in Relevant_data[relevant_cols]}
    path = '../Trained Models/' + tissue.strip() + '_Features_dict.pkl'

    with open(path, 'wb') as handle:
        pickle.dump(features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    "------------------------------- Model Training ---------------------"

    best_parameters = Best_param['Best_Parameters'][(Best_param['Dataset'] == 'Full Trace') & (Best_param['Tissue'] == tissue.strip()) & (Best_param['ML_Model'] == 'Random Forest')].values[0]
    best_parameters = ast.literal_eval(best_parameters)
    model = RandomForestClassifier(**best_parameters)
    model.fit(Relevant_data[relevant_cols], Relevant_data[y])  # train the model

    print(tissue + ' Model trained')
    path = '../Trained Models/' + tissue.strip() + '_RF_Model.pkl'
    with open(path, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    "------------------------------- SHAP Part --------------------------"


    explainer = shap.TreeExplainer(model)

    print(tissue + ' Explainer calculated')

    path = '../Trained Models/' + tissue.strip() + '_RF_Explainer.pkl'
    with open(path, 'wb') as handle:
        pickle.dump(explainer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    shap_values = explainer.shap_values(Relevant_data[relevant_cols], check_additivity=False)
    shap.summary_plot(shap_values[1], Relevant_data[relevant_cols], show=False)  # Worning check_additivity=False
    plt.tight_layout()

    path = '../Trained Models/' + tissue + '_Shap.jpg'

    plt.savefig(path)
    plt.close()
    print(tissue + 'Importance Fig Saved')
    return tissue, strftime("%Y-%m-%d %H:%M:%S", gmtime())
"-------------------------------- Multiprocessing ----------------------------------"


def driver_func():
    PROCESSES = 1  # Here you can use number of CPUs for multiprocessing

    with mp.Pool(PROCESSES) as pool:
        results = [pool.apply_async(create_model_and_shap, (y,)) for y in y_columns[:1]]# rf_best_parameters_dict.keys()
        for r in results:
            results_tuple = r.get(timeout=None)
            print('@', results_tuple[0], ' finished', results_tuple[1])


if __name__ == '__main__':
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    driver_func()