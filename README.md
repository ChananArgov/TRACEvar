# TRACEvar
TRACEvar is a pathogenic variant prioritization tool that uses tissue-specific ML models. Given data of variants and the identity of the tissue affected by the disease, TRACEvar computes a pathogenicity score for each variant in the affected tissue by using random forest. [TRACEvar](https://netbio.bgu.ac.il/tracevar/) is also available online . 

# Dataset
The dataset contains TRACEvar features and variant labels per tissue, and can be found here.

# Download
To use TRACEvar on your device download all the project files and folders to your work directory.
Place the dataset file in the 'Data' folder.

# TRACEvar usage for model contruction
To create TRACEvar tissue-specific random forest models, run the Main.py script in the 'Code' folder. For each tissue model, the script will create 5 files in the 'Output' folder, as follows:
1. a python [pickle](https://docs.python.org/3/library/pickle.html) (pkl) that include the trained model.
2. a pkl file containing a dictionary of the model input features.
3. a pkl file containing the [SHAP](https://shap.readthedocs.io/en/latest/index.html) explainer.
4. a csv file containing the SHAP feature importance mean values.
5. a pdf file containing the SHAP summary plot that shows the top 20 most important features of the RF model.

# Dependencies for model contruction
TRACEvar requires python 3.8 and the following packages: sklearn, pandas, ast, matplotlib, os, numpy, shap, pickle. Use pip for packages installation.
TRACEvar scripts and dataset are available for human genome version hg37.

# TRACEvar usage for variant prioritization
1. Create models as described in 'TRACEvar usage for model contruction' above.
2. Given a list of variants in a VCF format, obtain their CADD features as follows: (i) upload the VCF file to [CADD](https://cadd.gs.washington.edu/score) hg37 V1.6; (ii) select 'include annotations' (appears at the bottom); (iii) obtain CADD features per variant.
3. Combine CADD with tissue-specific features by gene id column. 
4. Run TRACEvar. XXX

# Additional files
The Code folder contains the following python scripts:
1. Main.py - script that creates the random forest tissue-specific TRACEvar models, shap explainers and feature importance shap values.
2. ML_Models_BP_Comparision.py - script that comparing the performance of eight different ML methods per tissue. Performance is measured by auROC, auPRC, training time and prediction time. 
3. Feature_Gruops_Tissue_Specific.py - script that uses SHAP feature importance values in 17 tissue models to calculate the contribution of each feature group (see article). 
4. Prioritize_Candidate_Variants.py.py - script that prioritizes variants based on TRACEvar tissue-specific models.
5. Interpretation_for_Specific_Variant.py - script that creates the SHAP decision plot for a specific variant. 

# Cite
Please cite 'TRACEvar: Prioritizing and interpreting pathogenic variants that underlie hereditary diseases in tissue contexts. Argov et al, submitted.'

# Contact
Esti Yeger-Lotem, estiyl@bgu.ac.il
Chanan Argov, chanana@post.bgu.ac.il
