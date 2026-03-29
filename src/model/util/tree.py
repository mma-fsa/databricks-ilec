import pandas as pd, numpy as np
import sklearn as sk, xgboost as xgb
import json

class PoissonDecisionTree():
    
    def __init__(self, actual_col, offset_col):
        self.actual_col = actual_col
        self.offset_col = offset_col
        self._params = {
            'objective': 'count:poisson',
            'max_depth': 3,
            'tree_method': 'exact',
            'grow_policy': 'lossguide',
            'seed':0
        }
        self._preproc = None
        self._bst = None
        
    def fit(self, df : pd.DataFrame):
        
        if not self.actual_col in df.columns:
            raise Exception(f"Missing column (for actuals): {self.actual_col}")
        if not self.offset_col in df.columns:
            raise Exception(f"Missing column (for offset): {self.offset_col}")
        
        str_cols = list(map(
            lambda c: str(c),
            df.select_dtypes(
            include=['object', 'string']).columns
        ))
        
        response_cols = [
            self.actual_col,
            self.offset_col
        ]

        num_cols = list(
            set(df.columns)
            .difference(set(str_cols))
            .difference(set(response_cols))
        )
        
        xfrm = [
            (
                "ohe",
                sk.preprocessing.OneHotEncoder(
                    drop="first", handle_unknown="ignore",
                ),
                str_cols
            ),
            (
                "num", "passthrough", num_cols
            )
        ]
        
        self._preproc = sk.compose.ColumnTransformer(
            transformers=xfrm,
            remainder="drop",
            verbose_feature_names_out=False
        )
        
        X_mat = self._preproc.fit_transform(df)
        offset = df[self.offset_col].to_numpy("float64")
        y = df[self.actual_col].to_numpy("float64")
        f_names = self._preproc.get_feature_names_out().tolist()
        xgb_mat = xgb.DMatrix(
            data=X_mat,
            label=y,
            base_margin = np.log(offset),
            feature_names=f_names
        )
        df_X_mat = pd.DataFrame(
            X_mat,
            columns=f_names
        )
        
        self._bst = xgb.train(
            params = self._params,
            dtrain=xgb_mat,
            num_boost_round=1
        )
        
        tree_json = json.loads(
            self._bst.get_dump(dump_format="json")[0])
        
        self._tree_data = collect_split_stats(
            tree_json, 
            df_X_mat,
            y, 
            offset)
        
        self._tree_df = pd.DataFrame(self._tree_data)
        
    def __str__(self):
        return pretty_print_compact_tree(self._tree_df)
        
    def print_tree(self):
        print(str(self))
        
    def print_nodes(self, digits=4) -> str:
        print(
            pretty_print_split_stats(
                self._tree_df,
                digits=digits
            )
        )

# -------------------------------------------------
# 3) Helpers
# -------------------------------------------------
def poisson_deviance(y, mu):
    """
    Exact Poisson deviance:
      2 * sum( y*log(y/mu) - (y - mu) )
    with y*log(y/mu) defined as 0 when y == 0.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if np.any(mu <= 0):
        raise ValueError("All fitted means mu must be > 0.")
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * np.sum(term - (y - mu))


def node_mu_with_offset(y_node, exposure_node):
    """
    Poisson MLE in a node with log(exposure) as fixed offset:
      mu_i = exposure_i * rate_hat
      rate_hat = sum(y) / sum(exposure)
    """
    total_exp = np.sum(exposure_node)
    if total_exp <= 0:
        raise ValueError("Node has non-positive total exposure.")
    rate_hat = np.sum(y_node) / total_exp
    # keep strictly positive to avoid numerical issues when all y == 0
    rate_hat = max(rate_hat, 1e-15)
    return exposure_node * rate_hat


def split_mask(x_col, split_value, missing_go_to, yes_child_id, no_child_id, missing_child_id):
    """
    XGBoost JSON tree nodes use:
      condition
      split
      yes / no / missing
    For numeric splits, 'yes' is the branch when feature < condition.
    Missing values go to 'missing'.
    """
    is_missing = pd.isna(x_col).to_numpy()
    go_yes = (~is_missing) & (x_col.to_numpy() < split_value)
    go_missing = is_missing

    left = np.zeros(len(x_col), dtype=bool)
    right = np.zeros(len(x_col), dtype=bool)

    # Map yes/no/missing onto left/right according to children ordering
    # We identify left/right by matching child nodeids.
    left_child_id = yes_child_id if yes_child_id in (yes_child_id, no_child_id, missing_child_id) else None
    # But more robustly we'll infer from actual child list outside this helper.

    return go_yes, go_missing


def collect_split_stats(node, X_sub, y_sub, exposure_sub, path_conditions=None, show_missing=False):
    """
    Recursively compute exact Poisson deviance reduction for each split,
    while storing a human-readable chain of split conditions.

    Parameters
    ----------
    node : dict
        Parsed XGBoost JSON node.
    X_sub : pd.DataFrame
        Data reaching this node.
    y_sub : np.ndarray
        Counts reaching this node.
    exposure_sub : np.ndarray
        Exposure reaching this node.
    path_conditions : list[str] | None
        List of human-readable conditions defining the node.

    Returns
    -------
    list[dict]
    """
    if path_conditions is None:
        path_conditions = []

    results = []

    if "children" not in node:
        return results

    feature = node["split"]
    threshold = node["split_condition"]
    yes_id = node["yes"]
    no_id = node["no"]
    missing_id = node["missing"]

    children = node["children"]
    left_child = children[0]
    right_child = children[1]
    left_id = left_child["nodeid"]
    right_id = right_child["nodeid"]

    xcol = X_sub[feature]
    is_missing = pd.isna(xcol).to_numpy()
    go_yes = (~is_missing) & (xcol.to_numpy() < threshold)
    go_no = (~is_missing) & (~go_yes)
    go_missing = is_missing

    left_mask = np.zeros(len(X_sub), dtype=bool)
    right_mask = np.zeros(len(X_sub), dtype=bool)

    if yes_id == left_id:
        left_mask |= go_yes
        yes_text = f"{feature} < {threshold}"
    elif yes_id == right_id:
        right_mask |= go_yes
        yes_text = f"{feature} < {threshold}"
    else:
        raise ValueError("yes child id not found among node children")

    if no_id == left_id:
        left_mask |= go_no
        no_text = f"{feature} >= {threshold}"
    elif no_id == right_id:
        right_mask |= go_no
        no_text = f"{feature} >= {threshold}"
    else:
        raise ValueError("no child id not found among node children")

    if missing_id == left_id:
        left_mask |= go_missing
        missing_text = f"{feature} missing"
    elif missing_id == right_id:
        right_mask |= go_missing
        missing_text = f"{feature} missing"
    else:
        raise ValueError("missing child id not found among node children")

    # If missing follows the same side as yes or no, merge text for readability
    left_conditions = list(path_conditions)
    right_conditions = list(path_conditions)

    left_parts = []
    right_parts = []

    if yes_id == left_id:
        left_parts.append(yes_text)
    if no_id == left_id:
        left_parts.append(no_text)
    if missing_id == left_id and show_missing:
        left_parts.append(missing_text)

    if yes_id == right_id:
        right_parts.append(yes_text)
    if no_id == right_id:
        right_parts.append(no_text)
    if missing_id == right_id and show_missing:
        right_parts.append(missing_text)

    if left_parts:
        left_item = "(" + " OR ".join(left_parts) + ")" if len(left_parts) > 1 else left_parts[0]
        left_conditions.append(left_item)
    if right_parts:
        right_item = "(" + " OR ".join(right_parts) + ")" if len(right_parts) > 1 else right_parts[0]
        right_conditions.append(right_item)

    mu_parent = node_mu_with_offset(y_sub, exposure_sub)
    
    dev_parent = poisson_deviance(y_sub, mu_parent)

    y_left = y_sub[left_mask]
    y_right = y_sub[right_mask]
    exp_left = exposure_sub[left_mask]
    exp_right = exposure_sub[right_mask]

    mu_left = node_mu_with_offset(y_left, exp_left)
    mu_right = node_mu_with_offset(y_right, exp_right)

    dev_left = poisson_deviance(y_left, mu_left)
    dev_right = poisson_deviance(y_right, mu_right)
    dev_children = dev_left + dev_right
    dev_reduction = dev_parent - dev_children

    results.append({
        "nodeid": node["nodeid"],
        "split_label": f"{feature} < {threshold}",
        "path_pretty": " AND ".join(path_conditions) if path_conditions else "ROOT",
        "left_path_pretty": " AND ".join(left_conditions),
        "right_path_pretty": " AND ".join(right_conditions),
        "feature": feature,
        "threshold": threshold,
        "n_parent": len(y_sub),
        "sum_y_parent": float(np.sum(y_sub)),
        "sum_exp_parent": float(np.sum(exposure_sub)),
        "ae_parent": float(np.sum(y_sub)) / float(np.sum(exposure_sub)),
        "dev_parent": float(dev_parent),
        "n_left": int(left_mask.sum()),
        "sum_y_left": float(np.sum(y_left)),
        "sum_exp_left": float(np.sum(exp_left)),
        "ae_left": float(np.sum(y_left)) / float(np.sum(exp_left)),
        "dev_left": float(dev_left),
        "n_right": int(right_mask.sum()),
        "sum_y_right": float(np.sum(y_right)),
        "sum_exp_right": float(np.sum(exp_right)),
        "ae_right": float(np.sum(y_right)) / float(np.sum(exp_right)),
        "dev_right": float(dev_right),
        "dev_children": float(dev_children),
        "dev_reduction": float(dev_reduction),
        "xgb_gain": float(node.get("gain", np.nan)),
        "xgb_cover": float(node.get("cover", np.nan)),
    })

    results.extend(
        collect_split_stats(
            left_child,
            X_sub.iloc[left_mask].copy(),
            y_left,
            exp_left,
            left_conditions
        )
    )

    results.extend(
        collect_split_stats(
            right_child,
            X_sub.iloc[right_mask].copy(),
            y_right,
            exp_right,
            right_conditions
        )
    )

    return results

def pretty_print_split_stats(split_df, digits=4, sort_by_node=True):
    import pandas as pd

    if split_df is None or len(split_df) == 0:
        return "No splits found."

    df = split_df.copy()

    if sort_by_node and "nodeid" in df.columns:
        df = df.sort_values(["nodeid"]).reset_index(drop=True)

    def fmt(x):
        if pd.isna(x):
            return "NA"
        if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer()):
            return f"{int(x)}"
        if isinstance(x, float):
            return f"{x:,.{digits}f}"
        return str(x)

    lines = []

    line = "=" * 120
    lines.append(line)
    lines.append("POISSON SPLIT SUMMARY")
    lines.append(line)

    for _, row in df.iterrows():
        lines.append(f"Node {row['nodeid']}")
        lines.append(f"  Path: {row['path_pretty']}")
        lines.append(f"  Split: {row['feature']} < {fmt(row['threshold'])}")

        lines.append("  Parent")
        lines.append(
            f"    n={fmt(row.get('n_parent'))}  "
            f"actual={fmt(row.get('sum_y_parent'))}  "
            f"exp={fmt(row.get('sum_exp_parent'))}  "
            f"ae={fmt(row.get('ae_parent'))}  "
            f"dev={fmt(row.get('dev_parent'))}"
        )

        lines.append("  Left child")
        lines.append(f"    path: {row['left_path_pretty']}")
        lines.append(
            f"    n={fmt(row.get('n_left'))}  "
            f"actual={fmt(row.get('sum_y_left'))}  "
            f"exp={fmt(row.get('sum_exp_left'))}  "
            f"ae={fmt(row.get('ae_left'))}  "
            f"dev={fmt(row.get('dev_left'))}"
        )

        lines.append("  Right child")
        lines.append(f"    path: {row['right_path_pretty']}")
        lines.append(
            f"    n={fmt(row.get('n_right'))}  "
            f"actual={fmt(row.get('sum_y_right'))}  "
            f"exp={fmt(row.get('sum_exp_right'))}  "
            f"ae={fmt(row.get('ae_right'))}  "
            f"dev={fmt(row.get('dev_right'))}"
        )

        lines.append("  Improvement")
        lines.append(
            f"    children_dev={fmt(row.get('dev_children'))}  "
            f"dev_reduction={fmt(row.get('dev_reduction'))}  "
            f"xgb_gain={fmt(row.get('xgb_gain'))}"
        )
        lines.append("-" * 120)

    return "\n".join(lines)

def pretty_print_compact_tree(split_stats, digits=4, sort_children=True):
    """
    Render a compact tree string from the output of collect_split_stats(...).

    Accepts either:
      - list[dict] returned by collect_split_stats(...)
      - pd.DataFrame built from that list

    Example output:

    root: ae=1.0000   y=10000
      - foo < 12: ae=1.2000   y=4234
        - thud < 3: ae=0.9000   y=100
        - thud >= 3: ae=1.3000   y=402
      - foo >= 12: ae=0.8000   y=5766
    """
    import pandas as pd

    if split_stats is None:
        return "No splits found."

    if isinstance(split_stats, pd.DataFrame):
        df = split_stats.copy()
    else:
        df = pd.DataFrame(split_stats)

    if len(df) == 0:
        return "No splits found."

    def fmt(x):
        if pd.isna(x):
            return "NA"
        if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer()):
            return str(int(x))
        if isinstance(x, float):
            return f"{x:,.{digits}f}"
        return str(x)

    def path_depth(path):
        if path == "ROOT":
            return 0
        return len(str(path).split(" AND "))

    def last_condition(path):
        if path == "ROOT":
            return "root"
        parts = str(path).split(" AND ")
        return parts[-1]

    # One row per internal node, indexed by its parent node path
    rows_by_path = {row["path_pretty"]: row for _, row in df.iterrows()}

    def build_line(label, ae, y, depth):
        prefix = "  " * depth
        if depth == 0:
            return f"{prefix}{label}: ae={fmt(ae)}   y={fmt(y)}"
        return f"{prefix}- {label}: ae={fmt(ae)}   y={fmt(y)}"

    lines = []

    # Root summary comes from the row whose path_pretty == ROOT
    root_row = rows_by_path.get("ROOT")
    if root_row is None:
        # Fallback: shallowest row
        root_row = df.sort_values("path_pretty", key=lambda s: s.map(path_depth)).iloc[0]

    lines.append(build_line("root", root_row.get("ae_parent"), root_row.get("sum_y_parent"), 0))

    visited = set()

    def walk(parent_path, depth):
        if parent_path in visited:
            return
        visited.add(parent_path)

        row = rows_by_path.get(parent_path)
        if row is None:
            return

        children = [
            {
                "path": row["left_path_pretty"],
                "label": last_condition(row["left_path_pretty"]),
                "ae": row.get("ae_left"),
                "y": row.get("sum_y_left"),
            },
            {
                "path": row["right_path_pretty"],
                "label": last_condition(row["right_path_pretty"]),
                "ae": row.get("ae_right"),
                "y": row.get("sum_y_right"),
            },
        ]

        if sort_children:
            children = sorted(children, key=lambda x: (path_depth(x["path"]), x["label"]))

        for child in children:
            lines.append(build_line(child["label"], child["ae"], child["y"], depth))
            # Recurse only if this child path is itself a parent node somewhere
            if child["path"] in rows_by_path:
                walk(child["path"], depth + 1)

    walk("ROOT", 1)
    return "\n".join(lines)
