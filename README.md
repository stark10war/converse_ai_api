import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from scipy.stats import pointbiserialr, chi2_contingency
import statsmodels.api as sm

# ----------------------
# Helper: WoE & IV
# ----------------------
def calc_woe_iv(X, y, bins=10):
    try:
        # Bin continuous variable
        if np.issubdtype(X.dtype, np.number):
            est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
            X_binned = est.fit_transform(X.values.reshape(-1, 1)).astype(int).flatten()
        else:
            X_binned = X.astype(str)

        df = pd.DataFrame({"x": X_binned, "y": y})
        grouped = df.groupby("x")

        dist_good = grouped["y"].apply(lambda x: (x == 0).sum()) / (y == 0).sum()
        dist_bad = grouped["y"].apply(lambda x: (x == 1).sum()) / (y == 1).sum()

        woe = np.log((dist_bad + 1e-6) / (dist_good + 1e-6))
        iv = ((dist_bad - dist_good) * woe).sum()
        return iv
    except Exception:
        return np.nan

# ----------------------
# Main function
# ----------------------
def feature_metrics(df, target, features):
    y = df[target].values
    X = df[features]

    results = []

    # 1. Train RF for importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_

    for feat in features:
        x = df[feat]
        
        # --- IV ---
        iv = calc_woe_iv(x, y)

        # --- Correlation ---
        try:
            if np.issubdtype(x.dtype, np.number):
                corr, _ = pointbiserialr(y, x.fillna(0))
            else:
                contingency = pd.crosstab(x.fillna("NA"), y)
                chi2, _, _, _ = chi2_contingency(contingency)
                corr = chi2
        except Exception:
            corr = np.nan

        # --- Logistic p-value ---
        try:
            x_const = sm.add_constant(x.fillna(0))
            logit = sm.Logit(y, x_const).fit(disp=0)
            pval = logit.pvalues[1]
        except Exception:
            pval = np.nan

        results.append({
            "feature": feat,
            "IV": iv,
            "Importance_RF": rf_importance[list(features).index(feat)],
            "Correlation": corr,
            "Logit_pval": pval
        })

    return pd.DataFrame(results).sort_values(by="Importance_RF", ascending=False).reset_index(drop=True)

# ----------------------
# Example usage
# ----------------------
# df = pd.read_csv("your_data.csv")
# metrics_df = feature_metrics(df, target="target_col", features=["f1","f2","f3"])
# print(metrics_df)
