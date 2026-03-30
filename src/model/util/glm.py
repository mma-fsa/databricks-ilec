import pandas as pd, numpy as np
import formulaic as frm
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype
from typing import Dict, Iterable, List, Sequence
import glum as glm

def get_coef_table(glmnet : glm.GeneralizedLinearRegressor, alpha_index : int) -> pd.Series:
    coef_at_idx = np.zeros(glmnet.coef_path_.shape[1] + 1)
    coef_at_idx[0] = glmnet.intercept_path_[alpha_index]
    coef_at_idx[1:] = glmnet.coef_path_[alpha_index, :]

    coef_table = pd.Series(coef_at_idx)
    coef_table.index = glmnet.coef_table().index
    
    return coef_table

def calc_path_stats(glmnet : glm.GeneralizedLinearRegressor, X : pd.DataFrame, offset : pd.Series, y : pd.Series) -> pd.DataFrame:
    from sklearn.metrics import d2_tweedie_score
    scores = []
    for alpha_idx in range(0, int(glmnet.n_alphas)):
        alpha_preds = glmnet.predict(X, offset=offset, alpha_index=alpha_idx)
        scores.append(d2_tweedie_score(
            y,
            alpha_preds,
            power=1
        ))
    
    return pd.DataFrame({
        "alpha_index" : np.arange(0, int(glmnet.n_alphas), 1),
        "d2" : scores
    })

def set_df_categoricals(df: pd.DataFrame, default_levels: Dict[str, str]) -> pd.DataFrame:
    result = df.copy()

    for column_name, default_level in default_levels.items():
        if column_name not in result.columns:
            raise KeyError(f"Column '{column_name}' not found in dataframe.")

        series = result[column_name]

        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = [default_level] + [category for category in series.cat.categories if category != default_level]
            result[column_name] = series.cat.set_categories(categories, ordered=series.cat.ordered)
            continue

        if not (is_object_dtype(series.dtype) or is_string_dtype(series.dtype)):
            raise TypeError(f"Column '{column_name}' must be an object, string, or categorical dtype.")

        non_null = series.dropna()
        if not non_null.map(lambda value: isinstance(value, str)).all():
            raise TypeError(f"Column '{column_name}' must contain only string values.")

        categories = [default_level] + [level for level in pd.Index(non_null).drop_duplicates().tolist() if level != default_level]
        result[column_name] = pd.Categorical(series, categories=categories)

    return result

class PoissonGLMFactorAnalysis():
    
    # do not modify signature of this
    def __init__(self, coef_table : pd.DataFrame, model_spec : frm.ModelSpec):
        self.model_spec = model_spec
        self.data_variables = [str(variable) for variable in model_spec.variables_by_source.get("data", [])]
        self.factor_to_variables = {
            str(factor): self._unique_in_order(
                str(variable)
                for variable in variables
                if str(variable) in self.data_variables
            )
            for factor, variables in model_spec.factor_variables.items()
        }
        self.factor_to_terms = {
            str(factor): [str(term) for term in terms]
            for factor, terms in model_spec.factor_terms.items()
        }
        self.variable_to_terms = {
            str(variable): [str(term) for term in terms]
            for variable, terms in model_spec.variable_terms.items()
            if str(variable) in self.data_variables
        }
        self.term_to_variables = {
            str(term): self._unique_in_order(
                str(variable)
                for variable in model_spec.term_variables[term]
                if str(variable) in self.data_variables
            )
            for term in model_spec.term_factors.keys()
        }
        self.term_to_columns = {
            str(term): list(model_spec.column_names[term_slice])
            for term, term_slice in model_spec.term_slices.items()
        }
        self.coefficients = self._normalize_coef_table(coef_table)
        self.intercept = float(self.coefficients.get("intercept", 0.0))
        self.column_to_term = {"intercept": "intercept"}
        self.column_to_variables = {"intercept": []}
        for term_name, column_names in self.term_to_columns.items():
            for column_name in column_names:
                self.column_to_term[column_name] = term_name
                self.column_to_variables[column_name] = list(self.term_to_variables[term_name])
        self.variable_order = self._build_variable_order()
        self.term_group_variables: Dict[str, List[str]] = {"intercept": []}
        self.term_groups = self._create_term_groups()
        self.group_to_columns: Dict[str, List[str]] = {}
        self.factor_map = self._validate_factor_map(self._build_factor_map(self.term_groups))
        self._set_group_state_from_factor_map(self.factor_map)

    def get_factor_map(self) -> pd.DataFrame:
        return self.factor_map.copy()

    def set_factor_map(self, factor_map: pd.DataFrame) -> None:
        normalized_factor_map = self._validate_factor_map(factor_map)
        self.factor_map = normalized_factor_map
        self._set_group_state_from_factor_map(normalized_factor_map)
    
    # do not modify signature of this
    def get_factor_analysis(self, df_traim : pd.DataFrame) -> List[pd.DataFrame]:
        """
            Returns a list of dataframes. Each dataframe contains a subset of the columns
            present in df_train with an additional column called factor.  Using all the
            returned dataframes, the glm's predictions can be created by multiplying together
            all the factors from each dataframe, after joining to df_train.  The goal is to
            break-down each component of the model into logical groups.  For example an 
            interaction between Sex and Attained Age in the formula will produce a dataframe
            with both Attained_Age and Sex in the factor dataframe, and the factor will contain
            Attained_Age + Sex + Attained_Age:Sex terms.
        """
        df_train = df_traim
        factor_tables: List[pd.DataFrame] = []

        for term_group, group_columns in self.group_to_columns.items():
            group_variables = self.term_group_variables[term_group]

            if term_group == "intercept":
                factor_tables.append(
                    pd.DataFrame(
                        {
                            "factor": [float(np.exp(self.intercept))],
                            "term_group": [term_group],
                        }
                    )
                )
                continue

            pred_grid = self._create_pred_grid(df_train, group_variables)
            model_matrix = self.model_spec.get_model_matrix(pred_grid)
            linear_predictor = model_matrix.loc[:, group_columns].to_numpy() @ self.coefficients.loc[group_columns].to_numpy()

            factor_table = pred_grid.loc[:, group_variables].copy()
            factor_table["factor"] = np.exp(linear_predictor)
            factor_table["term_group"] = term_group
            factor_tables.append(factor_table)

        return factor_tables

    def append_factor_preds(self, df: pd.DataFrame, colname_model_pred = "model_pred") -> pd.DataFrame:
        factor_tables = self.get_factor_analysis(df)
        df_with_factors = df.copy()
        factor_columns: List[str] = []

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

        df_with_factors[colname_model_pred] = df_with_factors.loc[:, factor_columns].prod(axis=1)
        return df_with_factors
    
    # you may modify inputs, do not change the output type
    def _create_term_groups(self) -> Dict[str, List[str]]:
        """
            Used by get_factor_analysis() to map each individual term in frm.ModelSpec.term_slices
            to a logical group.  For example, if there is a cr(Attained_Age, df=4, lower_bound=18, upper_bound=90),
            this produces several columns in the basis expansion, which will need to be mapped back to the Attained_Age
            column in the input. The mapping algorithm should be "greedy" based on the order in the formula, for example,
            the formula part cr(Attained_Age, df=4, lower_bound=18, upper_bound=90 )*Smoker_Status*Sex produces many
            different terms, e.g. cr(Attained_Age, df=4, lower_bound=18, upper_bound=90 ) + Smoker_Status + Sex +
                cr(Attained_Age, df=4, lower_bound=18, upper_bound=90 ):Smoker_Status + ... and these should all
            be mapped to the same group (e.g. "Attained_Age_x_Smoker_Status_x_Sex").  In the term Face_Amount_Band*Sex,
            Face_Amount + Face_Amount_Band:Sex should be mapped to "Face_Amount_Band_x_Sex", do not include the stand alone
            Sex variable, since we've "greedily" claimed it in the first term.
        """
        term_groups: Dict[str, List[str]] = {"intercept": ["intercept"]}
        assigned_terms = set()
        ordered_term_names = [str(term) for term in self.model_spec.terms]
        term_order = {term_name: index for index, term_name in enumerate(ordered_term_names)}

        candidate_terms = sorted(
            [
                term_name
                for term_name in ordered_term_names
                if len(self.term_to_variables[term_name]) > 1
            ],
            key=lambda term_name: (-len(self.term_to_variables[term_name]), term_order[term_name]),
        )

        for candidate_term in candidate_terms:
            if candidate_term in assigned_terms:
                continue

            clause_variables = self._order_variables(self.term_to_variables[candidate_term])
            clause_variable_set = set(clause_variables)
            grouped_terms = [
                term_name
                for term_name in ordered_term_names
                if term_name not in assigned_terms
                and self.term_to_variables[term_name]
                and set(self.term_to_variables[term_name]).issubset(clause_variable_set)
            ]
            if not grouped_terms:
                continue

            group_name = "_x_".join(clause_variables)
            if group_name in term_groups:
                term_groups[group_name].extend(grouped_terms)
            else:
                term_groups[group_name] = grouped_terms
                self.term_group_variables[group_name] = list(clause_variables)
            assigned_terms.update(grouped_terms)

        for term_name in ordered_term_names:
            if term_name in assigned_terms:
                continue

            fallback_variables = self._order_variables(self.term_to_variables[term_name])
            group_name = "_x_".join(fallback_variables) if fallback_variables else term_name
            if group_name in term_groups:
                term_groups[group_name].append(term_name)
            else:
                term_groups[group_name] = [term_name]
                self.term_group_variables[group_name] = list(fallback_variables)

        return term_groups
    
    # you may modify this as needed
    def _create_pred_grid(self, df_train: pd.DataFrame, group_variables: Sequence[str]) -> pd.DataFrame:
        """
            This function is used to enumerate the ranges of variables present in the training data, so that
            for an individual factor table, we have all possible combinations.  Assume numeric variables are 
            integers and only change by 1.  Enumerate all possible string variables based on the input.
        """
        if not group_variables:
            return pd.DataFrame(index=[0])

        grid_index = pd.MultiIndex.from_product(
            [self._enumerate_values(df_train[variable_name]) for variable_name in group_variables],
            names=group_variables,
        )
        pred_grid = grid_index.to_frame(index=False)

        for variable_name in self.data_variables:
            train_series = df_train[variable_name]
            if variable_name in pred_grid.columns:
                pred_grid[variable_name] = self._cast_like_train(pred_grid[variable_name], train_series)
                continue
            pred_grid[variable_name] = self._default_value(train_series)

        return pred_grid.loc[:, self.data_variables]

    def _normalize_coef_table(self, coef_table: pd.DataFrame) -> pd.Series:
        if isinstance(coef_table, pd.Series):
            return coef_table.astype(float)

        if isinstance(coef_table, pd.DataFrame):
            if coef_table.shape[1] == 1:
                return coef_table.iloc[:, 0].astype(float)
            if "coefficient" in coef_table.columns:
                return coef_table.set_index(coef_table.columns[0])["coefficient"].astype(float)

        raise TypeError("coef_table must be a pandas Series or single-column DataFrame.")

    def _build_factor_map(self, term_groups: Dict[str, List[str]]) -> pd.DataFrame:
        rows = []

        for term_group, term_names in term_groups.items():
            for term_name in term_names:
                if term_name == "intercept":
                    rows.append(
                        {
                            "term": "intercept",
                            "term_value": float(self.coefficients.loc["intercept"]),
                            "model_term": "intercept",
                            "term_group": term_group,
                        }
                    )
                    continue

                for column_name in self.term_to_columns[term_name]:
                    rows.append(
                        {
                            "term": column_name,
                            "term_value": float(self.coefficients.loc[column_name]),
                            "model_term": term_name,
                            "term_group": term_group,
                        }
                    )

        factor_map = pd.DataFrame(rows)
        return factor_map.loc[:, ["term", "term_value", "model_term", "term_group"]]

    def _validate_factor_map(self, factor_map: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"term", "term_group"}
        missing_columns = required_columns.difference(factor_map.columns)
        if missing_columns:
            raise ValueError(f"factor_map is missing required columns: {sorted(missing_columns)}")

        normalized = factor_map.copy()
        normalized["term"] = normalized["term"].astype(str)
        normalized["term_group"] = normalized["term_group"].astype(str)

        if normalized["term"].duplicated().any():
            duplicate_terms = normalized.loc[normalized["term"].duplicated(), "term"].tolist()
            raise ValueError(f"factor_map contains duplicate terms: {duplicate_terms}")

        expected_terms = list(self.coefficients.index.astype(str))
        observed_terms = normalized["term"].tolist()
        missing_terms = sorted(set(expected_terms).difference(observed_terms))
        unexpected_terms = sorted(set(observed_terms).difference(expected_terms))
        if missing_terms or unexpected_terms:
            raise ValueError(
                f"factor_map terms must match coefficient terms exactly. Missing={missing_terms}, Unexpected={unexpected_terms}"
            )

        if "term_value" in normalized.columns:
            expected_values = self.coefficients.loc[normalized["term"]].to_numpy(dtype=float)
            provided_values = normalized["term_value"].to_numpy(dtype=float)
            if not np.allclose(provided_values, expected_values, rtol=1e-12, atol=1e-12):
                raise ValueError("factor_map term_value does not match fitted coefficient values.")
        else:
            normalized["term_value"] = self.coefficients.loc[normalized["term"]].to_numpy(dtype=float)

        normalized["model_term"] = normalized["term"].map(self.column_to_term)
        normalized = normalized.loc[:, ["term", "term_value", "model_term", "term_group"]]
        return normalized.sort_values("term", key=lambda series: series.map({term: i for i, term in enumerate(expected_terms)})).reset_index(drop=True)

    def _set_group_state_from_factor_map(self, factor_map: pd.DataFrame) -> None:
        ordered_groups = list(dict.fromkeys(factor_map["term_group"].tolist()))
        self.group_to_columns = {}
        self.term_groups = {}
        self.term_group_variables = {}

        for term_group in ordered_groups:
            group_rows = factor_map.loc[factor_map["term_group"] == term_group]
            group_columns = group_rows["term"].tolist()
            model_terms = self._unique_in_order(group_rows["model_term"].tolist())

            self.group_to_columns[term_group] = group_columns
            self.term_groups[term_group] = model_terms
            self.term_group_variables[term_group] = self._group_variables_for_columns(group_columns)

    def _group_variables_for_columns(self, group_columns: Sequence[str]) -> List[str]:
        variable_set = {
            variable_name
            for column_name in group_columns
            for variable_name in self.column_to_variables[column_name]
        }
        return self._order_variables(variable_set)

    def _build_variable_order(self) -> Dict[str, int]:
        variable_order: Dict[str, int] = {}
        ordered_term_names = [str(term) for term in self.model_spec.terms]

        for term_index, term_name in enumerate(ordered_term_names):
            term_variables = self.term_to_variables[term_name]
            if len(term_variables) != 1:
                continue

            variable_name = term_variables[0]
            variable_order.setdefault(variable_name, term_index)

        fallback_start = len(ordered_term_names)
        for variable_name in self.data_variables:
            if variable_name in variable_order:
                continue

            candidate_indices = [
                index
                for index, term_name in enumerate(ordered_term_names)
                if variable_name in self.term_to_variables[term_name]
            ]
            variable_order[variable_name] = min(candidate_indices, default=fallback_start)

        return variable_order

    def _order_variables(self, variables: Iterable[str]) -> List[str]:
        unique_variables = self._unique_in_order(variable for variable in variables if variable in self.data_variables)
        return sorted(unique_variables, key=lambda variable_name: self.variable_order.get(variable_name, len(self.variable_order)))

    def _enumerate_values(self, series: pd.Series) -> List[object]:
        non_null = series.dropna()
        if is_numeric_dtype(series) and not self._is_categorical_dtype(series.dtype):
            return list(range(int(non_null.min()), int(non_null.max()) + 1))

        values = list(pd.Index(non_null).drop_duplicates())
        try:
            return sorted(values)
        except TypeError:
            return values

    def _cast_like_train(self, series: pd.Series, train_series: pd.Series) -> pd.Series:
        if self._is_categorical_dtype(train_series.dtype):
            return pd.Series(
                pd.Categorical(
                    series,
                    categories=train_series.cat.categories,
                    ordered=train_series.cat.ordered,
                ),
                name=series.name,
            )

        try:
            return series.astype(train_series.dtype)
        except (TypeError, ValueError):
            return series

    def _default_value(self, train_series: pd.Series):
        non_null = train_series.dropna()
        if non_null.empty:
            return np.nan
        return non_null.iloc[0]

    def _is_categorical_dtype(self, dtype) -> bool:
        return isinstance(dtype, pd.CategoricalDtype)

    def _unique_in_order(self, values: Iterable[str]) -> List[str]:
        return list(dict.fromkeys(values))
    
    
