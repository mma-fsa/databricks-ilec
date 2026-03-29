import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import formulaic as frm
import glum as glm

from src.lib.glm import PoissonGLMFactorAnalysis

TRAINING_DATA = Path(__file__).resolve().parents[1] / "fixtures" / "df_train.parquet"
MODEL_FORMULA = (
    " ~ cr(Attained_Age, df=4, lower_bound=18, upper_bound=90 )*Smoker_Status*Sex"
    " + Face_Amount_Band*Sex + Face_Amount_Band * Smoker_Status - 1"
)

def fit_model(df_train: pd.DataFrame):
    x_mat_formula = frm.Formula(MODEL_FORMULA)
    glmnet = glm.GeneralizedLinearRegressor(
        family="poisson",
        alpha_search=True,
        min_alpha_ratio=1e-6,
    )

    x_train = x_mat_formula.get_model_matrix(df_train)
    offset_train = np.log(df_train["ExpDth_VBT2015wMI_Cnt"])
    y_train = df_train["Death_Count"]

    glmnet.fit(
        x_train,
        y=y_train,
        offset=offset_train,
    )

    return x_train.model_spec, glmnet


def check_factor_analysis() -> bool:
    df_train = pd.read_parquet(TRAINING_DATA)
    model_spec, glmnet = fit_model(df_train)
    model_pred_factors = glmnet.predict(model_spec.get_model_matrix(df_train))

    factor_analysis = PoissonGLMFactorAnalysis(
        glmnet.coef_table(),
        model_spec,
    )

    factor_tables = factor_analysis.get_factor_analysis(df_train)
    df_with_factors, factor_columns = _append_factor_columns(df_train, factor_tables)
    df_with_factors["model_pred"] = df_with_factors.loc[:, factor_columns].prod(axis=1)

    return bool(
        np.allclose(
            df_with_factors["model_pred"],
            model_pred_factors,
            rtol=1e-10,
            atol=1e-10,
        )
    )


@pytest.fixture(scope="module")
def fitted_factor_analysis():
    df_train = pd.read_parquet(TRAINING_DATA)
    model_spec, glmnet = fit_model(df_train)
    return df_train, model_spec, glmnet, PoissonGLMFactorAnalysis(glmnet.coef_table(), model_spec)


def _append_factor_columns(df_train, factor_tables):
    df_with_factors = df_train.copy()
    factor_columns = []

    for factor_table in factor_tables:
        term_group = str(factor_table["term_group"].iat[0])
        factor_column = f"factor_{term_group}"
        join_columns = [
            column_name
            for column_name in factor_table.columns
            if column_name not in {"factor", "term_group"}
        ]

        if join_columns:
            df_with_factors = df_with_factors.merge(
                factor_table.loc[:, join_columns + ["factor"]].rename(columns={"factor": factor_column}),
                on=join_columns,
                how="left",
                validate="many_to_one",
            )
        else:
            df_with_factors[factor_column] = float(factor_table["factor"].iat[0])

        factor_columns.append(factor_column)

    return df_with_factors, factor_columns


def _group_name_for_model_terms(factor_map, expected_model_terms):
    expected_model_terms = set(expected_model_terms)
    for term_group, rows in factor_map.groupby("term_group"):
        if set(rows["model_term"]) == expected_model_terms:
            return term_group
    raise AssertionError(f"No group found for model terms: {sorted(expected_model_terms)}")


def test_create_term_groups_greedily(fitted_factor_analysis):
    _, _, _, factor_analysis = fitted_factor_analysis

    grouped_model_terms = {frozenset(model_terms) for model_terms in factor_analysis.term_groups.values()}

    assert grouped_model_terms == {
        frozenset(["intercept"]),
        frozenset([
            "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90)",
            "Smoker_Status",
            "Sex",
            "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90):Smoker_Status",
            "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90):Sex",
            "Smoker_Status:Sex",
            "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90):Smoker_Status:Sex",
        ]),
        frozenset([
            "Face_Amount_Band",
            "Face_Amount_Band:Sex",
        ]),
        frozenset([
            "Face_Amount_Band:Smoker_Status",
        ]),
    }


def test_get_factor_map_returns_default_group_assignments(fitted_factor_analysis):
    _, _, glmnet, factor_analysis = fitted_factor_analysis

    factor_map = factor_analysis.get_factor_map()

    assert list(factor_map.columns) == ["term", "term_value", "model_term", "term_group"]
    assert factor_map["term"].tolist() == list(glmnet.coef_table().index)
    assert np.allclose(factor_map["term_value"], glmnet.coef_table().to_numpy(), rtol=1e-12, atol=1e-12)

    age_smoker_sex_group = _group_name_for_model_terms(
        factor_map,
        {
            "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90)",
            "Smoker_Status",
            "Sex",
            "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90):Smoker_Status",
            "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90):Sex",
            "Smoker_Status:Sex",
            "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90):Smoker_Status:Sex",
        },
    )
    age_smoker_sex_rows = factor_map.loc[
        factor_map["term_group"] == age_smoker_sex_group,
        "model_term",
    ]
    assert set(age_smoker_sex_rows) == {
        "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90)",
        "Smoker_Status",
        "Sex",
        "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90):Smoker_Status",
        "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90):Sex",
        "Smoker_Status:Sex",
        "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90):Smoker_Status:Sex",
    }


def test_set_factor_map_uses_custom_mapping(fitted_factor_analysis):
    df_train, model_spec, glmnet, factor_analysis = fitted_factor_analysis
    custom_factor_analysis = PoissonGLMFactorAnalysis(glmnet.coef_table(), model_spec)
    custom_map = custom_factor_analysis.get_factor_map()
    default_age_smoker_sex_group = _group_name_for_model_terms(
        custom_map,
        {
            "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90)",
            "Smoker_Status",
            "Sex",
            "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90):Smoker_Status",
            "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90):Sex",
            "Smoker_Status:Sex",
            "cr(Attained_Age, df=4, lower_bound=18, upper_bound=90):Smoker_Status:Sex",
        },
    )
    default_face_band_smoker_group = _group_name_for_model_terms(
        custom_map,
        {"Face_Amount_Band:Smoker_Status"},
    )

    custom_map.loc[
        custom_map["term_group"] == default_age_smoker_sex_group,
        "term_group",
    ] = "Combined_Age_Smoker_Sex"

    face_band_sex_terms = {
        "Face_Amount_Band[T.02: 10,000 - 24,999]",
        "Face_Amount_Band[T.03: 25,000 - 49,999]",
        "Face_Amount_Band[T.04: 50,000 - 99,999]",
        "Face_Amount_Band[T.05: 100,000 - 249,999]",
        "Face_Amount_Band[T.06: 250,000 - 499,999]",
        "Face_Amount_Band[T.07: 500,000 - 999,999]",
        "Face_Amount_Band[T.08: 1,000,000 - 2,499,999]",
        "Face_Amount_Band[T.09: 2,500,000 - 4,999,999]",
        "Face_Amount_Band[T.10: 5,000,000 - 9,999,999]",
        "Face_Amount_Band[T.11: 10,000,000+]",
        "Face_Amount_Band[T.02: 10,000 - 24,999]:Sex[T.M]",
        "Face_Amount_Band[T.03: 25,000 - 49,999]:Sex[T.M]",
        "Face_Amount_Band[T.04: 50,000 - 99,999]:Sex[T.M]",
        "Face_Amount_Band[T.05: 100,000 - 249,999]:Sex[T.M]",
        "Face_Amount_Band[T.06: 250,000 - 499,999]:Sex[T.M]",
        "Face_Amount_Band[T.07: 500,000 - 999,999]:Sex[T.M]",
        "Face_Amount_Band[T.08: 1,000,000 - 2,499,999]:Sex[T.M]",
        "Face_Amount_Band[T.09: 2,500,000 - 4,999,999]:Sex[T.M]",
        "Face_Amount_Band[T.10: 5,000,000 - 9,999,999]:Sex[T.M]",
        "Face_Amount_Band[T.11: 10,000,000+]:Sex[T.M]",
    }
    custom_map.loc[custom_map["term"].isin(face_band_sex_terms), "term_group"] = "Combined_Face_Band_Sex"

    custom_factor_analysis.set_factor_map(custom_map)

    updated_factor_map = custom_factor_analysis.get_factor_map()
    assert set(updated_factor_map["term_group"]) == {
        "intercept",
        "Combined_Age_Smoker_Sex",
        "Combined_Face_Band_Sex",
        default_face_band_smoker_group,
    }

    factor_tables = custom_factor_analysis.get_factor_analysis(df_train)
    assert {table["term_group"].iat[0] for table in factor_tables} == {
        "intercept",
        "Combined_Age_Smoker_Sex",
        "Combined_Face_Band_Sex",
        default_face_band_smoker_group,
    }

    df_with_factors, factor_columns = _append_factor_columns(df_train, factor_tables)
    reconstructed = df_with_factors.loc[:, factor_columns].prod(axis=1)
    direct_predictions = glmnet.predict(model_spec.get_model_matrix(df_train))
    assert np.allclose(reconstructed, direct_predictions, rtol=1e-10, atol=1e-10)


def test_factor_tables_reconstruct_predictions(fitted_factor_analysis):
    df_train, model_spec, glmnet, factor_analysis = fitted_factor_analysis
    factor_tables = factor_analysis.get_factor_analysis(df_train)
    df_with_factors, factor_columns = _append_factor_columns(df_train, factor_tables)

    assert df_with_factors.loc[:, factor_columns].notna().all().all()

    reconstructed = df_with_factors.loc[:, factor_columns].prod(axis=1)
    direct_predictions = glmnet.predict(model_spec.get_model_matrix(df_train))
    assert np.allclose(reconstructed, direct_predictions, rtol=1e-10, atol=1e-10)
    assert check_factor_analysis()