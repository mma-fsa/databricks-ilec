# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
import mlflow

# COMMAND ----------

model_uri = "models:/workspace.mlops_dev.ilec_pipeline/2"
req_file = mlflow.pyfunc.get_model_dependencies(
    model_uri,
    format="pip",
)
%pip install -r "{req_file}"
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F
model_uri = "models:/workspace.mlops_dev.ilec_pipeline/2"
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri, env_manager="local")

# COMMAND ----------

catalog = "workspace"
schema = "mlops_dev"

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

# COMMAND ----------


tbl_w_preds = tbl_ilec_data.repartition(100).withColumn(
        "prediction",
        predict_udf(
            F.struct(
                "Sex",
                "Smoker_Status",
                "Face_Amount_Band",
                "Attained_Age",
                "ExpDth_VBT2015wMI_Cnt",
            )
        ),
    )

(
    tbl_w_preds
    .write
    .mode("overwrite")
    .saveAsTable("workspace.mlops_dev.term_preds")
)



# COMMAND ----------

tbl_check.limit(100).toPandas()

# COMMAND ----------

tbl_check = spark.table("workspace.mlops_dev.term_preds")

(
    tbl_check
    .withColumn("pred_cnt", F.col("prediction") * F.col("ExpDth_VBT2015wMI_Cnt"))
).agg(F.sum(F.col("pred_cnt")), F.sum(F.col("Death_Count"))).toPandas() 


# COMMAND ----------


