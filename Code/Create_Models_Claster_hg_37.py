
import time
import matplotlib.pyplot as plt
# import seaborn as sns
import os
import pandas as pd

#import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap  # package used to calculate Shap values
pd.options.mode.chained_assignment = None
import pickle

"---------------------------- Load Data ------------------------------"
general_path = 'enter your repository'

path = os.path.join('TRACE_data', 'Full_Slim_Dataset_hg37-v1.6.csv')
Slim_dataset = pd.read_csv(path, engine='python')#low_memory=False,
print(Slim_dataset)

filename = os.path.join('RF_best_parameters_dict.pkl')
infile = open(filename,'rb')
rf_best_parameters_dict = pickle.load(infile)
infile.close()

"-------------------------- Data PreProcessing ------------------------"

def preprocessing_data(Relevant_Data):

    one_hot_columns = ['Type', 'AnnoType', 'Consequence', 'Domain', 'Dst2SplType']  # , 'EnsembleRegulatoryFeature'

    # Get one hot encoding of columns B
    one_hot = pd.get_dummies(Relevant_Data[one_hot_columns])
    # Drop column B as it is now encoded
    Relevant_Data = Relevant_Data.drop(one_hot_columns, axis=1)
    # Join the encoded df
    Relevant_Data = Relevant_Data.join(one_hot)
    # print(relevant_data_1)
    cHmm_columns = Slim_dataset.columns[Slim_dataset.columns.str.contains(pat='cHmm_E')].tolist()
    fill_zero_columns = ['motifECount', 'motifEHIPos', 'motifEScoreChng', 'mirSVR-Score', 'mirSVR-E', 'mirSVR-Aln',
                         'tOverlapMotifs', 'motifDist'] + cHmm_columns  # motifs with high number of nan 97%
    Relevant_Data[fill_zero_columns] = Relevant_Data[fill_zero_columns].fillna(value=0)
    fill_common_columns = ['cDNApos', 'relcDNApos', 'CDSpos', 'relCDSpos', 'protPos', 'relProtPos', 'Dst2Splice',
                           'SIFTval', 'PolyPhenVal', 'GerpRS', 'GerpRSpval', 'GerpN', 'GerpS', 'all Enc',
                           'Grantham', 'All SpliceAI', 'All MMSp', 'Dist2Mutation', 'All 00bp', 'dbscSNV',
                           'RemapOverlapTF', 'RemapOverlapCL', 'Trace Features']  # Locations, is this right?
    for cl in list(Relevant_Data):
        Relevant_Data[cl] = Relevant_Data[cl].fillna(Relevant_Data[cl].value_counts().idxmax())
    return Relevant_Data


y_columns = Slim_dataset.columns[Slim_dataset.columns.str.contains(pat = 'disease_causing')].tolist()
print(y_columns)
non_relevant_columns = ['VariationID', 'OMIMs', 'Manifested_Tissues', '#Chr', 'Pos', 'ConsDetail', 'motifEName', 'GeneID_y', 'FeatureID', 'GeneName', 'CCDS', 'Intron', 'Exon', 'SIFTcat', 'PolyPhenCat', 'bStatistic', 'targetScan', 'dbscSNV-rf_score', 'oAA', 'Ref', 'nAA', 'Alt', 'Segway']# it will be good to replace oAA and nAA with blssuom64 matrix. What bStatistic doing?
non_relevant_columns = non_relevant_columns + y_columns
non_relevant_patient = ['#Chr', 'Pos', 'ConsDetail', 'motifEName', 'GeneID_y', 'FeatureID', 'GeneName', 'CCDS', 'Intron', 'Exon', 'SIFTcat', 'PolyPhenCat', 'bStatistic', 'targetScan', 'dbscSNV-rf_score', 'oAA', 'Ref', 'nAA', 'Alt', 'Segway']# it will be good to replace oAA and nAA with blssuom64 matrix. What bStatistic doing?


cols = Slim_dataset.columns
relevant_columns = [x for x in cols if x not in non_relevant_columns]
Slim_Relevant = Slim_dataset[relevant_columns]
Slim_Relevant = preprocessing_data(Slim_Relevant)

relevant_y = [  'brain-1_disease_causing', 'Skin - Not Sun Exposed_disease_causing', 'Pituitary_disease_causing', 'Ovary_disease_causing', 
              'brain-3_disease_causing', 'Testis_disease_causing', 'Whole Blood_disease_causing', 'brain-2_disease_causing', 
              'brain_disease_causing']#'Heart - Left Ventricle_disease_causing',  'brain-0_disease_causing', 'Liver_disease_causing', 'Nerve - Tibial_disease_causing', 'kidney_disease_causing','Lung_disease_causing', 'Muscle - Skeletal_disease_causing']#,'Artery - Aorta_disease_causing', 

"--------------------------------- Creating Models And Explainers -------------------------------------------"

for y in relevant_y:
    tissue = y.replace("_disease_causing", "")

    print('-----------------', tissue, '-----------------')
    try:
        best_parameters = rf_best_parameters_dict[tissue.strip()][1]
        model = RandomForestClassifier(**best_parameters)
        model.fit(Slim_Relevant, Slim_dataset[y])
        print(tissue + ' Model trained')
        path = general_path + tissue.strip() + '_RF_Model.pkl'
        with open(path, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        explainer = shap.TreeExplainer(model)
        path = general_path + tissue.strip() + '_RF_Explainer.pkl'

        with open(path, 'wb') as handle:
            pickle.dump(explainer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print(tissue, ' have no validated model.')

    features_dict = {feature: Slim_Relevant[feature].value_counts().idxmax() for feature in Slim_Relevant}
    path = general_path + tissue.strip() + '_Features_dict.pkl'

    with open(path, 'wb') as handle:
        pickle.dump(features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    "------------------------ Shap ---------------------------------------"
    shap_values = explainer.shap_values(Slim_Relevant)
    vals = np.abs(shap_values).mean(0)
    print('vals', len(vals), vals)

    Feature_importance = pd.DataFrame(list(zip(Slim_Relevant.columns, sum(vals))),
                                          columns=['col_name', 'feature_importance_vals'])
    Feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    path = general_path + tissue.strip() + '_Shap_Importance_New_hg37.csv'
    Feature_importance.to_csv(path, index=False)
    print('Shap_Feature importance')
    print(Feature_importance.head())
    
    shap.summary_plot(shap_values[1], Slim_Relevant, plot_type="dot", show=False)
    plt.subplots_adjust(left=0.5, right=1, bottom=0.19, top=0.82)
    plt.suptitle(tissue + 'RF SHAP Feature Importance')
    file_name = tissue + ' SHAP Feature Importance' + '.pdf'  # + '_XGB'
    path = os.path.join('..', '..', 'Final_Results', 'Random_Forest', 'Shap_Importance', file_name)
    path = general_path + file_name

    plt.savefig(path, dpi=100)
    plt.close()
    print(tissue + ' SHAP plot Saved')

