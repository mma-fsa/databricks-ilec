# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "model_environment.yaml"
# environment_version = "5"
# ///
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC from src.model.util.glm import PoissonGLMFactorAnalysis, set_df_categoricals, calc_path_stats
# MAGIC from src.model.util.tree import PoissonDecisionTree

# COMMAND ----------

import pandas as pd, numpy as np
from pyspark.sql import functions as F
import formulaic as frm, sklearn as sk, glum as glm
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

MODEL_FORMULA = " ~ cr(Attained_Age, df=4, lower_bound=18, upper_bound=90 )*Smoker_Status*Sex + Face_Amount_Band"

# glmnet handles intercept for us
x_mat_formula = frm.Formula(MODEL_FORMULA + " - 1")

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

X_test = x_mat_formula.get_model_matrix(
    set_df_categoricals(df_test, default_levels)
)
offset_test = np.log(df_test["ExpDth_VBT2015wMI_Cnt"])
y_test = df_test["Death_Count"]

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
dtree_train = PoissonDecisionTree("Death_Count", "model_pred")
dtree_train.fit(df_train.drop("ExpDth_VBT2015wMI_Cnt", axis=1))
print(dtree_train)


# COMMAND ----------

df_path_stats_train = calc_path_stats(glmnet, X_train, offset_train, y_train)
df_path_stats_train.plot("alpha_index", "d2")

# COMMAND ----------

df_test["model_pred"] = glmnet.predict(X_test, offset=offset_test)
dtree_test = PoissonDecisionTree("Death_Count", "model_pred")
dtree_test.fit(df_test.drop("ExpDth_VBT2015wMI_Cnt", axis=1))
print(dtree_test)

# COMMAND ----------

df_path_stats_test = calc_path_stats(glmnet, X_test, offset_test, y_test)
df_path_stats_test.plot("alpha_index", "d2")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Factor Analysis

# COMMAND ----------

model_spec = X_train.model_spec
factor_analysis = PoissonGLMFactorAnalysis(
    glmnet.coef_table(),
    model_spec)

# check that factors are mapped correctly
display(factor_analysis.get_factor_map())

# COMMAND ----------

factors = factor_analysis.get_factor_analysis(df_train)

df_factor_chk = df_train.copy()
df_factor_chk = factor_analysis.append_factor_preds(
    df_factor_chk
).rename({"model_pred" : "model_pred_factor"}, axis=1)

df_factor_chk["model_pred_glm"] = glmnet.predict(
    X_train, alpha_index=99
)

check_preds = np.allclose(
    df_factor_chk["model_pred_factor"],
    df_factor_chk["model_pred_glm"]
)

if not check_preds:
    raise Exception("Factors do not match glm predictions")
else:
    print("factor analysis check passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bundle Model into Pyfunc

# COMMAND ----------

class PoissonGLM(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        self.model_spec = joblib.load(context.artifacts["model_spec"])
        self.glmnet = joblib.load(context.artifacts["glmnet"])
    
    def _load_from_memory(self, model_spec:frm.ModelSpec, glmnet:glm.GeneralizedLinearRegressor):
        self.model_spec = model_spec
        self.glmnet = glmnet

    def predict(self, context, model_input : pd.DataFrame, params=None):
        
        import numpy as np

        if not isinstance(model_input, pd.DataFrame):
            raise Exception(f"Expected model_input to be pandas DataFrame, got: {str(type(model_input))}")

        X_mat = self.model_spec.get_model_matrix(model_input)
        preds = self.glmnet.predict(
            X_mat, alpha_index=self.BEST_ALPHA
        )

        return pd.DataFrame({"prediction": preds})


# COMMAND ----------

# set additional model parameters
PoissonGLM.INPUT_COLS = list(model_spec.variables_by_source["data"])
PoissonGLM.BEST_ALPHA = 99

# write model artfifacts 
def serialize_artifacts(tmpdir : tempfile.TemporaryDirectory):
    
    model_spec_path = os.path.join(tmpdir, "model_spec.joblib")
    joblib.dump(model_spec, model_spec_path)

    glmnet_path = os.path.join(tmpdir, "glmnet.joblib")
    joblib.dump(glmnet, glmnet_path)
    
    return (model_spec_path, glmnet_path)

# test model predictions + serialization
run_model = PoissonGLM()
with tempfile.TemporaryDirectory() as tmpdir:
    model_spec_path, glmnet_path = serialize_artifacts(tmpdir)
    new_model_spec = joblib.load(model_spec_path)
    new_glmnet = joblib.load(glmnet_path)
    run_model._load_from_memory(
        new_model_spec,
        new_glmnet,
    )
    df_test_run = run_model.predict({}, df_train) 

y_train_preds = np.sum(
    df_test_run["prediction"] * df_train["ExpDth_VBT2015wMI_Cnt"]
)

ae_train = y_train_preds / df_train["Death_Count"].sum()
ae_train

# COMMAND ----------

with tempfile.TemporaryDirectory() as tmpdir:
    
    input_example = df_train[PoissonGLM.INPUT_COLS].head(5).copy()
    
    signature = mlflow.models.infer_signature(
        input_example,
        pd.DataFrame(df_test_run[: len(input_example)]),
        params={}  # optional param schema
    )

    with mlflow.start_run() as run:
        
        # Useful metadata / metrics
        mlflow.log_params({
            "model_formula" : MODEL_FORMULA
        })
    
        mlflow.log_metric("ae_train", ae_train)

        # model dependencies
        model_spec_path, glmnet_path = serialize_artifacts(tmpdir)
        
        # Deployable pyfunc
        pyfunc_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=PoissonGLM(),
            artifacts={
                "model_spec": model_spec_path,
                "glmnet": glmnet_path,
            },
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"formulaic=={frm.__version__}",
                f"glum=={glm.__version__}",
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

from pyspark.sql import functions as F
URI = pyfunc_info.model_uri
predict_py = mlflow.pyfunc.load_model(URI)
predict_udf = mlflow.pyfunc.spark_udf(spark, URI, env_manager="local")

# COMMAND ----------

input_cols = getattr(predict_py.unwrap_python_model(), "INPUT_COLS")
input_cols

# COMMAND ----------

tbl_w_preds = tbl_ilec_data.withColumn(
        "prediction",
        predict_udf(
            F.struct(*input_cols)
        )
    )

output_table_full_name = "workspace.mlops_dev.test_glm_preds"

(
    tbl_w_preds
    .write
    .mode("overwrite")
    .saveAsTable(output_table_full_name)
)

# COMMAND ----------

(
    spark.read.table("workspace.mlops_dev.test_glm_preds")
    .withColumn("model_pred_cnt", 
                F.col("prediction") * F.col("ExpDth_VBT2015wMI_Cnt"))
    .agg(
        F.sum("model_pred_cnt").alias("model_pred_cnt"),
        F.sum("Death_Count").alias("Death_Count")
    )
    .withColumn("AE", F.col("Death_Count") / F.col("model_pred_cnt"))
).toPandas()
