from Orange.data import Domain, Table

# Orange3 - Machine Learning comparing different algorithms, low-code
# https://medium.com/p/d185214037af

# Medium: Data preparation for Machine Learning with KNIME and the Python “vtreat” package
# https://medium.com/p/efcaf58fa783

# conda install -n py_orange -c conda-forge vtreat

import pyarrow.parquet as pq

import Orange
import pandas as pd
import numpy as np
import pickle

import vtreat

# var_path_data = "C:\\Users\\a123456\\knime-workspace\\orange\\orange_machine_learning\\"
var_path_data = "/Users/m_lauber/Dropbox/knime-workspace/orange/orange_machine_learning/"
df = pq.read_table(var_path_data + "test.parquet").to_pandas()

# set the path for the pickel file
path = var_path_data + 'vtreat_model.pkl'
# Save object as pickle file
vtreat_model_load = pickle.load(open(path, 'rb'))

# apply the loaded model to the test data (or any new data)
d_prepared = vtreat_model_load.transform(df)


def pandas_to_orange(df):
    # Preprocess the date column
    for col in df.columns:
        if df[col].dtype == "datetime64[ns]":
            df[col] = df[col].astype(str)

    # Determine variable types
    def get_variable(col):
        if np.issubdtype(df[col].dtype, np.number):
            return Orange.data.ContinuousVariable(col)
        elif df[col].dtype == np.dtype('O'):
            if df[col].apply(lambda x: isinstance(x, str)).all():
                return Orange.data.StringVariable(col)
            else:
                return Orange.data.DiscreteVariable(col, values=df[col].unique().tolist())
        else:
            raise ValueError(f"Unsupported dtype {df[col].dtype} for column {col}")

    feature_vars = [get_variable(col) for col in df.columns if not isinstance(get_variable(col), Orange.data.StringVariable)]
    meta_vars = [get_variable(col) for col in df.columns if isinstance(get_variable(col), Orange.data.StringVariable)]

    domain = Orange.data.Domain(feature_vars, metas=meta_vars)

    # Convert non-string columns to a NumPy array
    non_string_data = df.select_dtypes(exclude='object').values
    string_data = df.select_dtypes(include='object').astype(str).values

    # Create a table for non-string columns
    table = Orange.data.Table.from_numpy(domain, X=non_string_data, metas=string_data)

    return table


# Convert pandas DataFrame to Orange data table
out_data = pandas_to_orange(d_prepared)