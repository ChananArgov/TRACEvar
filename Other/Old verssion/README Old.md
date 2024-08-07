

# TRACEvar
TRACEvar is a pathogenic variant prioritization tool that uses tissue-specific ML models. Given data of variants and the identity of the tissue affected by the disease, TRACEvar computes a pathogenicity score for each variant in the affected tissue by using random forest. [TRACEvar](https://netbio.bgu.ac.il/tracevar/) is also available online. 

<img src="TRACEvar concept figure MSB.png.jpg" alt="TRACEvar concept figure">


# Dataset
The dataset contains TRACEvar features and variant labels per tissue, and can be found [here](https://zenodo.org/record/5769155#.Yh9sEOhBwuU).

# Download
To use TRACEvar on your device download all the project files and folders to your work directory.
Place the dataset file in the 'Data' folder.

# TRACEvar usage for model construction
To create TRACEvar tissue-specific random forest models, run the Create_TRACEvar_Models_and_SHAP_Explainers.py script in the 'New Code' folder. For each tissue model, the script will create 5 files in the 'Output' folder, as follows:
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

<table>
  <tr>
    <th>Notebook name</th>
    <th>Contant</th>
    <th>Manuscript figures</th>
  </tr>
  <tr>
    <td>One_Gene_Model_POC</td>
    <td>POC examples ML</td>
    <td></td>
  </tr>
  <tr>
    <td>One_Gene_POC_Figs</td>
    <td>POC plots</td>
    <td>Fig. 2A-B</td>
  </tr>
  <tr>
    <td>Compare_TRACEvar_to_Common_Scores</td>
    <td>comparing TRACEvar performance to common genetic tools</td>
    <td>Fig. 2C-D</td>
  </tr>
  <tr>
    <td>Methods_Comparison_Analysis</td>
    <td>Methods comparison analysis and plots</td>
    <td>Fig 2. E-F</td>
  </tr>
   <tr>
    <td>SHAP_Importance_Slim</td>
    <td>Feature importance for each TRACEvar 17 model</td>
    <td>Fig. 3A</td>
  </tr>
  <tr>
    <td>Feature Importance Analysis</td>
    <td>Feature importance analysis and plots</td>
    <td>Fig. 3B-C</td>
  </tr>
  <tr>
    <td>Transfear_Learning_hg37_Multiprossesing_Doc</td>
    <td>Training and evaluating TRAVEvar multi-tissue model</td>
    <td></td>
  <tr>
    <td>Analyze transfer learning results</td>
    <td>Multi-tissue model performance</td>
    <td>Fig. 4B</td>
  </tr>
   <tr>
    <td>Transfer Learning Feature Importance</td>
    <td>Multi-tissue model feature importance analysis</td>
    <td>Fig. 4C</td>
  </tr>
  <tr>
    <td>Patient Variants Prioritization</td>
    <td>Patient variant prioritization and interpretation</td>
    <td>Fig. 5B-D</td>
  </tr>
  <tr>
    <td>Analyze Patients Rank</td>
    <td>Statistics and comparision of TRACEvar patients variants prioritization results</td>
    <td>Fig. 5A, Fig. S4</td>
  </tr>
</table>

</body>
</html>


# Cite
Please cite 'TRACEvar: Prioritizing and interpreting pathogenic variants that underlie hereditary diseases in tissue contexts. Argov et al, submitted.'

# Contact
Esti Yeger-Lotem, estiyl@bgu.ac.il
Chanan Argov, chanana@post.bgu.ac.il
