# Orange3 -  Machine Learning comparing different algorithms, low-code

Use Orange 3 to compare several machine-learning models. Load the data from a parquet file and make it usable in Orange3


for details check out the **Medium article**:

https://medium.com/p/d185214037af

The most important files:

* **ML_Cross_Validation.ows** = Orange3 workflow with initial comparison
* **ML_Cross_Validation_vtreat.ows** = Orange3 workflow using data preparation with vtreat (cf. article)
* orange_import_parquet_file.py = Python script to import .parquet file and prepare it for Orange3 (included in workflow)
* orange_import_parquet_create_vtreat_model.py = import .parquet and create vtreat data preparation plan
* orange_import_parquet_apply_vtreat_model.py = import .parquet file and apply vtreat plan

#### further articles you might be interested in (setting up conda and using vtreat):

- KNIME and Python — Setting up and managing Conda environments (https://medium.com/p/2ac217792539)
- Medium: Data preparation for Machine Learning with KNIME and the Python “vtreat” package (https://medium.com/p/efcaf58fa783)

# 
conda create --name py_orange python=3.9

conda activate py_orange

conda install pip

conda config --add channels conda-forge

conda update -n py_orange --all

conda install -n py_orange -c conda-forge xgboost

conda install -n py_orange -c conda-forge catboost

pip install vtreat

#
![Screenshot of an Orange3 workflow for Machine Learning](Orange3_workflow.png)
