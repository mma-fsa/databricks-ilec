from pyspark import pipelines as dp
from pyspark.sql import functions as F
import util as util

@dp.temporary_view
def ilec_data_import():
    tbl_ilec_raw = (
        spark.read.format("csv")
        .option("header", "true")
        .option("delimiter", "\t")
        .option("inferSchema", "true")
        .load("/Volumes/workspace/default/ilec/soa-ilec/raw_data/ILEC_2012_19 - 20240429.txt")
    )
    return tbl_ilec_raw

@dp.temporary_view
def ilec_data_clean():
    return util.clean_str_cols(spark.read.table("ilec_data_import"))
    
@dp.materialized_view
def ilec_data():
    tbl_vw = (
        spark.read.table("ilec_data_clean")
        .withColumn("Issue_Year_Group",
            F.when(F.col("Issue_Year") <= 1980, "≤1980")
            .when((F.col("Issue_Year") > 1980) & (F.col("Issue_Year") <= 1996), "1981-1996")
            .when((F.col("Issue_Year") >= 1997) & (F.col("Issue_Year") <= 2003), "1997-2003")
            .when((F.col("Issue_Year") >= 2004) & (F.col("Issue_Year") <= 2008), "2004-2008")
            .when(F.col("Issue_Year") >= 2009, "2009+")
            .otherwise("Unknown")
        )
    )

    

    return tbl_vw
