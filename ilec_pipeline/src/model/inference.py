# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "/Workspace/Users/mr.anderson.1725@gmail.com/databricks-ilec/ilec_pipeline/src/model/model_environment.yaml"
# environment_version = "5"
# ///
import mlflow
from pyspark.sql import functions as F

model_uri = dbutils.widgets.get("model_uri")
predict_py = mlflow.pyfunc.load_model(model_uri)
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri, env_manager="local")

# COMMAND ----------

input_cols = getattr(predict_py.unwrap_python_model(), "INPUT_COLS")
input_cols

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
input_schema = dbutils.widgets.get("input_schema")
input_table = dbutils.widgets.get("input_table")

tbl_ilec_data = (
    spark.read.table(f"{catalog}.{input_schema}.{input_table}")
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

# COMMAND ----------

output_schema = dbutils.widgets.get("output_schema")
output_table = dbutils.widgets.get("output_table")

tbl_w_preds = tbl_ilec_data.withColumn(
        "prediction",
        predict_udf(
            F.struct(*input_cols)
        )
    )

output_table_full_name = f"{catalog}.{output_schema}.{output_table}"
(
    tbl_w_preds
    .write
    .mode("overwrite")
    .saveAsTable(output_table_full_name)
)



# COMMAND ----------

tbl_check = spark.table(output_table_full_name)
(
    tbl_check
    .withColumn("pred_cnt", F.col("prediction") * F.col("ExpDth_VBT2015wMI_Cnt"))
).agg(F.sum(F.col("pred_cnt")), F.sum(F.col("Death_Count"))).toPandas() 

