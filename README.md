# TRACEvar
TRACEvar is a pathogenic variant prioritization tool, based on tissue-specific ML models.

TRACEvar features dataset could be find in XXXXXX.

In the Code folder you can find the following py files:

1. Find_Best_ML_Paraneters.py - This file taking TRACEvar dataset and finding the best-parameter (BP) for tuning eight ML methods for tissue-specific variant prioritizing.
2. Compare_ML_Methods.py - Based on TRACEvar dataset and the BP from 1, this script comparing between the eight ML methods, recording the precision-recall AUC, ROC-AUC, training and predicting time. 
3. Create_TRACEvar_Models_Explainers.py - This file taking TRACEvar dataset + BP, creating 17 tissue-specific RF models and SHAP explainers this script also contains the feature importance (FI) calculation for each of the models.
4. Feature_Importance_Analysis.py - Taking FI across the 17 models and analysing it. 
5. Prioritize_Cabdidate_Variants_And_Interpetate.py - Prioritizing candidate variants based on the tissue-specific PF models, and explains variant prediction based on SHAP explainer. 
