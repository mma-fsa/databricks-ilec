# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "model_environment.yaml"
# environment_version = "5"
# ///
from src.model.util.glm import PoissonGLMFactorAnalysis, set_df_categoricals
from src.model.util.tree import PoissonDecisionTree

# COMMAND ----------

import pandas as pd, numpy as np
from pyspark.sql import functions as F
import formulaic as frm, sklearn as sk, glum as glm
from sklearn.metrics import d2_tweedie_score
import tempfile, os, mlflow, joblib

# COMMAND ----------

def safe_get(var_name, default):
    try:
        return dbutils.widgets.get(var_name)
    except:
        return default

catalog = safe_get("catalog", "workspace")
schema = safe_get("schema", "mlops_dev")

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
        .groupBy(["Observation_Year", "Issue_Year", "Issue_Age", "Sex", "Smoker_Status", "Attained_Age", "Face_Amount_Band"])
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

x_mat_formula = frm.Formula(" ~ cr(Attained_Age, df=4, lower_bound=18, upper_bound=90 )*Smoker_Status*Sex + Face_Amount_Band - 1")

#x_mat_formula = frm.Formula(" ~ Attained_Age*Smoker_Status*Sex + Face_Amount_Band - 1")

# COMMAND ----------

default_levels = {
    "Sex": "M",
    "Smoker_Status": "NS",
    "Face_Amount_Band" : "05: 100,000 - 249,999" 
}

X_train = x_mat_formula.get_model_matrix(
    set_df_categoricals(df_train, default_levels)
)
offset_train = np.log(df_train["ExpDth_VBT2015wMI_Cnt"])
y_train = df_train["Death_Count"]

X_val = x_mat_formula.get_model_matrix(
    set_df_categoricals(df_test, default_levels)
)
offset_val = np.log(df_test["ExpDth_VBT2015wMI_Cnt"])
y_val = df_test["Death_Count"]

# COMMAND ----------

glmnet = glm.GeneralizedLinearRegressor(
    family="poisson",
    alpha_search=True,
    min_alpha_ratio=1e-6
)
glmnet.fit(
    X_train,
    y = y_train,
    offset = offset_train
)

# COMMAND ----------

df_train["model_pred"] = glmnet.predict(X_train, offset=offset_train)
dtree = PoissonDecisionTree("Death_Count", "model_pred")
dtree.fit(df_train.drop("ExpDth_VBT2015wMI_Cnt", axis=1))
print(dtree)


# COMMAND ----------

coef_at_idx = np.zeros(glmnet.coef_path_.shape[1] + 1)
coef_at_idx[0] = glmnet.intercept_path_[99]
coef_at_idx[1:] = glmnet.coef_path_[99, :]

coef_at_idx - glmnet.coef_table().to_numpy()


# COMMAND ----------



# COMMAND ----------

factor_analysis = PoissonGLMFactorAnalysis(
    glmnet.coef_table(),
    X_train.model_spec)

display(factor_analysis.get_factor_map())


# COMMAND ----------

factors = factor_analysis.get_factor_analysis(df_train)
factors[0]

# COMMAND ----------

factors[2]

# COMMAND ----------

display(factor_analysis.append_factor_preds(df_train))

# COMMAND ----------

d2_tweedie_score(
    y_train,
    glmnet.predict(X_train, offset=offset_train, alpha_index=99),
    power=1
)

# COMMAND ----------

y_preds = glmnet.predict(X_train, offset=offset_train, alpha_index=99)

dtree_train = DecisionTreeRegressor(
    criterion=
)

# COMMAND ----------

glmnet.alpha = 0.04
glmnet.aic(X_train, y_train)

# COMMAND ----------

glmnet.coef_path_

# COMMAND ----------

np.sum(y_val) / np.sum(glmnet.predict(
    X_val,
    offset=offset_val,
    alpha_index=40
))

# COMMAND ----------

glmnet.coef_table()

# COMMAND ----------

glmnet.alp

# COMMAND ----------

np.where(
    glmnet.
)

 

# COMMAND ----------

glmnet.coef_path_[-1,:,90,:]

# COMMAND ----------

# standardized
glmnet.coef_table()

# COMMAND ----------

# not standardized
glmnet.coef_table()

# COMMAND ----------



# COMMAND ----------

glmnet.coef_table()

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
        
        offset_col = "ExpDth_VBT2015wMI_Cnt"

        if offset_col in model_input.columns.to_list():
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
    
    signature = mlflow.models.infer_signature(
        input_example,
        pd.DataFrame(df_test_run[: len(input_example)]),
        params={}  # optional param schema
    )

    with mlflow.start_run() as run:
        # Useful metadata / metrics
        mlflow.log_params(params)
        ae_ratio_train = float(df_test_run["prediction"].sum() / np.sum(y_train))
        mlflow.log_metric("val_sum_pred_over_sum_actual", ae_ratio_train)
        mlflow.log_metric("best_iteration", int(bst.best_iteration))

        # Optional: also log the native booster flavor
        mlflow.xgboost.log_model(
            xgb_model=bst,
            name="xgb_native",
            model_format="json",
        )

        # Deployable pyfunc
        booster_path, preproc_path = serialize_artifacts(tmpdir)
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

# COMMAND ----------

signature

# COMMAND ----------


