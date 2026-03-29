from pyspark.sql import functions as F
from typing import Dict, List

def clean_str_cols(df : F.DataFrame) -> F.DataFrame:
    tbl_clean = df
    str_cols = [
        x for x in tbl_clean.dtypes
        if x[1] == "string"
    ]
    for col_name, _ in str_cols:
        tbl_clean = (
            tbl_clean
            .withColumn(col_name, F.upper(F.trim(col_name)))
        )
    return tbl_clean

def agg_column_values(df : F.DataFrame, colname : str, mappings : Dict[str, List[str]] ) -> F.DataFrame:
    for tgt_val, src_vals in mappings.items():
        df = (
            df.withColumn(colname,
                F.when(F.col(colname).isin(src_vals), F.lit(tgt_val))
                .otherwise(F.col(colname))
            )
        )
    return df