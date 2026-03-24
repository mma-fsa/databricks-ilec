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
import tempfile, os, mlflow, joblib

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

preprocessor.get_feature_names_out()
array(['Sex_F', 'Smoker_Status_S', 'Smoker_Status_U', 'Face_Amount_Band',
       'Attained_Age'], dtype=object)

# COMMAND ----------

preprocessor.output_indices_

# COMMAND ----------

class PoissonXgbModel(mlflow.pyfunc.PythonModel):

    INPUT_COLS = [
        "Sex",
        "Smoker_Status",
        "Face_Amount_Band",
        "Attained_Age"
    ]
    OFFSET_COL = "ExpDth_VBT2015wMI_Cnt"  

    def load_context(self, context):
        self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        self.booster = xgb.Booster()
        self.booster.load_model(context.artifacts["booster"])

    def _load_from_memory(self, 
                          bst : xgboost.Booster, 
                          preproc : sklearn.base.TransformerMixin):
        self.preprocessor = preproc
        self.booster = bst

    def predict(self, context, model_input : pd.DataFrame, params=None):
        
        import numpy as np

        if not isinstance(model_input, pd.DataFrame):
            raise Exception(f"Expected model_input to be pandas DataFrame, got: {str(type(model_input))}")
        
        offset_col = PoissonXgbModel.OFFSET_COL

        if PoissonXgbModel.OFFSET_COL in model_input.columns:
            offset_col = np.log(model_input[offset_col].to_numpy())
        else:
            offset_col = np.zeros((model_input.shape[0]))

        X_proc = self.preprocessor.transform(model_input)
        dmat = xgb.DMatrix(X_proc, base_margin=offset_col)

        preds = self.booster.predict(dmat)

        # Return DataFrame for stable serving output schema
        return pd.DataFrame({"prediction": preds})

# write model artfifacts 
def serialize_artifacts(tmpdir : tempfile.TemporaryDirectory):
    booster_path = os.path.join(tmpdir, "model.json")
    bst.save_model(booster_path)

    preproc_path = os.path.join(tmpdir, "preprocessor.joblib")
    joblib.dump(preprocessor, preproc_path)

    return (booster_path, preproc_path)

# test serialization
run_model = PoissonXgbModel()
with tempfile.TemporaryDirectory() as tmpdir:
    booster_path, preproc_path = serialize_artifacts(tmpdir)
    new_bst = xgboost.Booster()
    new_bst.load_model(booster_path)
    new_preproc = joblib.load(preproc_path)
    run_model._load_from_memory(
        new_bst,
        new_preproc,
    )
    df_test_run = run_model.predict({}, df_train) 

df_test_run["prediction"].sum() / df_train["Death_Count"].sum()

# COMMAND ----------

with tempfile.TemporaryDirectory() as tmpdir:
    
    input_example = df_train[PoissonXgbModel.INPUT_COLS].head(5).copy()
    
    signature = infer_signature(
        input_example,
        pd.DataFrame({"prediction": np.asarray(df_test_run[: len(input_example)])}),
        params={"offset_col": "offset"}  # optional param schema
    )

    with mlflow.start_run() as run:
        # Useful metadata / metrics
        mlflow.log_params(params)
        mlflow.log_metric("val_sum_pred_over_sum_actual", float(np.sum(val_preds) / np.sum(y_val)))
        mlflow.log_metric("best_iteration", int(bst.best_iteration))

        # Optional: also log the native booster flavor
        mlflow.xgboost.log_model(
            xgb_model=bst,
            name="xgb_native",
            model_format="json",
        )

        # Deployable pyfunc
        pyfunc_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=PoissonXgbModel(),
            artifacts={
                "booster": booster_path,
                "preprocessor": preproc_path,
            },
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"xgboost=={xgb.__version__}",
                f"scikit-learn=={__import__('sklearn').__version__}",
                f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}",
                f"joblib=={joblib.__version__}",
            ],
            input_example=input_example,
            signature=signature,
        )

        print("PyFunc model URI:", pyfunc_info.model_uri)
        print("Run ID:", run.info.run_id)
