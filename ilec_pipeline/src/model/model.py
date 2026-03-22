# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "xgboost==3.2.0",
#   "scikit-learn==1.8",
# ]
# ///
import pandas as pd, numpy as np
from pyspark.sql import functions as F
import xgboost, sklearn

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
tbl_ilec_data = (
    spark.read.table(f"{catalog}.{schema}.ilec_data")
    .filter(
        (F.trim(F.col("Insurance_Plan")) == "TERM") &
        (F.col("SOA_Post_Lvl_Ind") != F.lit("PLT")) &
        (F.col("ExpDth_VBT2015wMI_Cnt") > F.lit(0.0))
    )
    .withColumn("DATASET",
        F.when(F.col("Observation_Year") <= F.lit(2017), "TRAIN")
         .otherwise("TEST")
    )
)

tbl_ilec_data.count()

# COMMAND ----------

def agg_data(df : F.DataFrame) -> F.DataFrame:
    return (
        df
        .groupBy(["Sex", "Smoker_Status", "Attained_Age", "Face_Amount_Band"])
        .agg(
            F.sum("Death_Count").alias("Death_Count"),
            F.sum("ExpDth_VBT2015wMI_Cnt").alias("ExpDth_VBT2015wMI_Cnt")
        )
    )

tbl_train = (
    tbl_ilec_data
    .filter(F.col("DATASET") == F.lit("TRAIN"))
)

tbl_test = (
    tbl_ilec_data
    .filter(F.col("DATASET") == F.lit("TEST"))
)

df_train = agg_data(tbl_train).toPandas()
df_test = agg_data(tbl_test).toPandas()


# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

transformers = [
    (
        "sex_ohe",
        OneHotEncoder(
            categories=[["M", "F"]],
            drop="first",
            sparse_output=False,
            handle_unknown="ignore",
        ),
        ["Sex"],
    ),
    (
        "smoker_ohe",
        OneHotEncoder(
            categories=[["NS", "S", "U"]],
            drop="first",
            sparse_output=False,
            handle_unknown="ignore",
        ),
        ["Smoker_Status"],
    ),
    (
        "face_amount_ord",
        OrdinalEncoder(
            categories=[[
                '01: 0 - 9,999',
                '02: 10,000 - 24,999',
                '03: 25,000 - 49,999',
                '04: 50,000 - 99,999',
                '05: 100,000 - 249,999',
                '06: 250,000 - 499,999',
                '07: 500,000 - 999,999',
                '08: 1,000,000 - 2,499,999',
                '09: 2,500,000 - 4,999,999',
                '10: 5,000,000 - 9,999,999',
                '11: 10,000,000+',
            ]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ),
        ["Face_Amount_Band"],
    ),
    ("num", "passthrough", ["Attained_Age"]),
]

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    verbose_feature_names_out=False,
)


# COMMAND ----------

X_train = preprocessor.fit_transform(df_train)
offset_train = np.log(df_train["ExpDth_VBT2015wMI_Cnt"])
y_train = df_train["Death_Count"]

X_val = preprocessor.transform(df_test)
offset_val = np.log(df_test["ExpDth_VBT2015wMI_Cnt"])
y_val = df_test["Death_Count"]

# COMMAND ----------

import xgboost as xgb

# 1. Setup Parameters (Native API style)
params = {
    'objective': 'count:poisson',
    'learning_rate': 0.01,
    'gamma': 1,
    'max_depth': 3,
    'tree_method': 'hist' # Recommended for speed/modern performance
}

# 2. Wrap data in DMatrix (including margins)
dtrain = xgb.DMatrix(X_train, label=y_train, base_margin=offset_train)
dval = xgb.DMatrix(X_val, label=y_val, base_margin=offset_val)

# 3. Train using the native xgb.train function
# evals is a list of pairs (DMatrix, name) used for early stopping
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=[(dval, 'validation')],
    early_stopping_rounds=10,
    verbose_eval=50,

)

# 4. Predict
# The booster uses the base_margin already embedded in dtest
preds = bst.predict(dval)

np.sum(preds) / np.sum(y_val)

# COMMAND ----------

import mlflow
import joblib

class PoissonXgbModel(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        self.booster = xgb.Booster()
        self.booster.load_model(context.artifacts["booster"])

    def predict(self, context, model_input, params=None):
        
        import numpy as np

        FEATURE_COLS = ["Sex", "Smoker_Status", "Face_Amount_Band", "Attained_Age"]
        OFFSET_COL = "ExpDth_VBT2015wMI_Cnt"
      
        if not isinstance(model_input, pd.DataFrame):
            raise Exception(f"Expected model_input to be pandas DataFrame, got: {str(type(model_input))}")
        
        if OFFSET_COL in model_input.columns:
            offset_col = np.log(model_input[OFFSET_COL].to_numpy())
        else:
            offset_col = np.zeros(())
                
        X = model_input[FEATURE_COLS]
        X_proc = self.preprocessor.transform(X)

        if offset_col is not None:
            base_margin = np.log(model_input[OFFSET_COL].to_numpy())
            dmat = xgb.DMatrix(X_proc, base_margin=base_margin)
        else:
            dmat = xgb.DMatrix(X_proc)

        preds = self.booster.predict(dmat)

        # Return DataFrame for stable serving output schema
        return pd.DataFrame({"prediction": preds})
