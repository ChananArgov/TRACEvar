

# TRACEvar
TRACEvar is a pathogenic variant prioritization tool that uses tissue-specific ML models. Given data of variants and the identity of the tissue affected by the disease, TRACEvar computes a pathogenicity score for each variant in the affected tissue by using GBM. [TRACEvar](https://netbio.bgu.ac.il/tracevar/) is also available online. 

<img src="TRACEvar MSB.png" alt="TRACEvar concept figure">


# Dataset
The dataset contains TRACEvar features and variant labels per tissue, and can be found [here](https://zenodo.org/record/5769155#.Yh9sEOhBwuU).

# Download
To use TRACEvar on your device download all the project files and folders to your work directory.
Place the dataset file in the 'Data' folder.

# TRACEvar usage for model construction
To create TRACEvar tissue-specific random forest models, run the Create_TRACEvar_Models_and_SHAP_Explainers.py script in the 'MSB Code' folder. For each tissue model, the script will create 5 files in the 'Output' folder, as follows:
1. a python [pickle](https://docs.python.org/3/library/pickle.html) (pkl) that include the trained model.
2. a pkl file containing a dictionary of the model input features.
3. a pkl file containing the [SHAP](https://shap.readthedocs.io/en/latest/index.html) explainer.
4. a csv file containing the SHAP feature importance mean values.
5. a pdf file containing the SHAP summary plot that shows the top 20 most important features of the RF model.

# Dependencies for model construction
TRACEvar requires python 3.8 and the following packages: sklearn, pandas, ast, matplotlib, os, numpy, shap, pickle. Use pip for packages installation.
TRACEvar scripts and dataset are available for human genome version hg37, hg38.

# TRACEvar usage for variant prioritization
1. Create models as described in 'TRACEvar usage for model construction' above.
2. Given a list of variants in a VCF format, obtain their CADD features as follows: (i) upload the VCF file to [CADD](https://cadd.gs.washington.edu/score) hg37 V1.6 or hg38 V1.6; (ii) select 'include annotations' (appears at the bottom); (iii) obtain CADD features per variant.
3. Run Prioritize_Variants_By_TRACEvar.py script (the input is detailed at the top of the script).


# TRACEvar usage for variant interpretation
In order to obtain an interpretation for a specific variant, you must first run the 'TRACEvar usage for variant prioritization' section.
Then, run the Variant_Interpretation_by_TRACEvar.py script, with the relevant variant index (the input is detailed at the top of the script).




<h2>Analysis scripts</h2>
The updated code is in the 'MSB Scripts' folder. 
Previous version exists in the 'Analysis scripts' folder.
<table>
  <tr>
    <th>Notebook name</th>
    <th>Contant</th>
    <th>Manuscript figures</th>
  </tr>
  <tr>
    <td>POC_One_gene</td>
    <td>POC examples ML</td>
    <td></td>
  </tr>
  <tr>
    <td>POC_Plot</td>
    <td>POC plots</td>
    <td>Fig. 2A-B</td>
  </tr>
  <tr>
    <td>Compare_code_test</td>
    <td>comparing TRACEvar performance to common genetic tools</td>
    <td>Fig. 2C-D</td>
  </tr>
  <tr>
    <td>Methods_Comparison_Analysis</td>
    <td>Methods comparison analysis and plots</td>
    <td>Fig 2. E-F</td>
  </tr>
   <tr>
    <td>single_variant_analysis</td>
    <td>Scoring and interpretation for single variant</td>
    <td>Fig 3.B-D</td>
  </tr>
   <tr>
    <td>Create_TRACEvar_Models_and_SHAP_Explainers</td>
    <td>Model creation, feature importance analysis for each TRACEvar 14 model</td>
    <td>Fig. 4A</td>
  </tr>
  <tr>
    <td>feature_importance_analysis</td>
    <td>Feature importance analysis</td>
    <td>Fig. 4B-C</td>
  </tr>
  <tr>
    <td>Multi_tissue_models</td>
    <td>Training and evaluating TRAVEvar multi-tissue model</td>
    <td>Fig. 5</td>
   <tr>
    <td>SHAP_Analysis_Multi-Tissue_Model</td>
    <td>Multi-tissue model feature importance analysis</td>
    <td>Fig. 5C</td>
  </tr>
</table>

</body>
</html>


# Cite
Please cite 'Tissue-aware interpretation of genetic variants advances the etiology of rare diseases. Argov et al, submitted.'

# Contact
Esti Yeger-Lotem, estiyl@bgu.ac.il
Chanan Argov, chanana@post.bgu.ac.il
