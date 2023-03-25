import pandas as pd
import Orange

import numpy as np
import pyarrow.parquet as pq


# var_path_data = "C:\\Users\\a123456\\knime-workspace\\orange\\orange_machine_learning\\"
var_path_data = "/Users/m_lauber/Dropbox/knime-workspace/orange/orange_machine_learning/"
df = pq.read_table(var_path_data + "train.parquet").to_pandas()

features = [feat for feat in df.columns]

num_cols = df[features].select_dtypes(include='number').columns.tolist()
cat_cols = df[features].select_dtypes(exclude='number').columns.tolist()

df[cat_cols] = df[cat_cols].astype(str)

print(df.head())


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
            raise ValueError(
                f"Unsupported dtype {df[col].dtype} for column {col}")

    feature_vars = [get_variable(col) for col in df.columns if not isinstance(
        get_variable(col), Orange.data.StringVariable)]
    meta_vars = [get_variable(col) for col in df.columns if isinstance(
        get_variable(col), Orange.data.StringVariable)]

    domain = Orange.data.Domain(feature_vars, metas=meta_vars)

    # Convert non-string columns to a NumPy array
    non_string_data = df.select_dtypes(exclude='object').values
    string_data = df.select_dtypes(include='object').astype(str).values

    # Create a table for non-string columns
    table = Orange.data.Table.from_numpy(
        domain, X=non_string_data, metas=string_data)

    return table


# Convert the pandas DataFrame to an Orange Table
out_data = pandas_to_orange(df)

# print(type(out_table))

# Now you can use the out_table with Orange3
