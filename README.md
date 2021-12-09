# TRACEvar
TRACEvar is a pathogenic variant prioritization tool, based on tissue-specific ML models.

# TRACEvar Webtool
[TRACEvar](https://netbio.bgu.ac.il/tracevar/) webtool allows you to upload your variant file in vcf format and get TRACEvar pathogenicity score for each variant.

# Dataset

TRACEvar features dataset could be find in XXXXXX.

# Download

To use TRACEvar on your devise you first need to download all the project files to your working directory.
The features dataset file needs to be placed in the 'Data' folder.

# Usage

To create TRACEvar tissue-specific Random-Forest (RF) models, you need to run the Main.py script in the 'Code' folder. This script will create 5 files for each tissue model in the 'Output' folder, 3 python [pickel](https://docs.python.org/3/library/pickle.html) (pkl) including the trained model, a dictionary with all the model input features, [shap](https://shap.readthedocs.io/en/latest/index.html) explainer, csv file with shap feature importance mean values and shap summery plot in pdf format showing the top 20 features for each RF model.

# Dependencies
TRACEvar runing on python 3.8 and requir the pakages: sklearn, pandas, ast, matplotlib, os, numpy, shap, pickle. Use pip for pakages installation.
TRACEvar scripts and dataset are avalable for genome verssion hg37, it will be avalable in the future for hg38.
To run TRACEvar on your own dataset you can run TRACEvar webtool(see above), alternatively you will need to get the CADD features for your variants and TRACE features of the relevant genes, and to combine tham. To this end, first upload your dataset in vcf format to [CADD](https://cadd.gs.washington.edu/score) hg37 V1.6, choose include annotations bottom to get all CADD features. Second download TRACE features from YYYYY, and lastly combine the feature datasets on the ensamble gene id column. Now you can run TRACEvar following the instruction in the 'usage' section.

# Cite
Please cite  TRACEvar: Prioritizing and interpreting pathogenic variants that underlie hereditary diseases in tissue contexts. Argov et al, submited. 

# Additional files

In the Code folder you can find the following py scripts:

1. Main.py - This file creating the randome-forest tissue-specific TRACEvar models, shap explainers and feature importance shap values.
2. ML_Models_BP_Comparision.py - This script comparing between the eight ML methods, recording the precision-recall AUC, ROC-AUC, training and predicting time. 
3. Feature_Gruops_Tissue_Specific.py - Taking feature importance shap values across the 17 models, finding the most contributing feature groups (see article). 
4. Prioritize_Candidate_Variants.py.py - Prioritizing candidate variants based on TRACEvar tissue-specific models.
5. Interpretation_for_Specific_Variant.py -  Explains variant prediction based on SHAP explainer. 
