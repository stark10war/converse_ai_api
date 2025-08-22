
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import pointbiserialr, chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ----------------------
# Helper: WoE & IV
# ----------------------
def calc_woe_iv(X, y, bins=10):
    try:
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
# Helper: VIF
# ----------------------
def calc_vif(df_features):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_features.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df_features.values, i)
        for i in range(df_features.shape[1])
    ]
    return dict(zip(vif_data["feature"], vif_data["VIF"]))

# ----------------------
# Main function
# ----------------------
def feature_metrics(df, target, features, top_n=50):
    y = df[target].values
    X = df[features]

    results = []

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_

    numeric_feats = [f for f in features if np.issubdtype(df[f].dtype, np.number)]
    vif_dict = calc_vif(df[numeric_feats].fillna(0)) if numeric_feats else {}

    for feat in features:
        x = df[feat]
        iv = calc_woe_iv(x, y)

        try:
            if np.issubdtype(x.dtype, np.number):
                corr, _ = pointbiserialr(y, x.fillna(0))
            else:
                contingency = pd.crosstab(x.fillna("NA"), y)
                chi2, _, _, _ = chi2_contingency(contingency)
                corr = chi2
        except Exception:
            corr = np.nan

        try:
            x_const = sm.add_constant(x.fillna(0))
            logit = sm.Logit(y, x_const).fit(disp=0)
            pval = logit.pvalues[1]
        except Exception:
            pval = np.nan

        vif_val = vif_dict.get(feat, np.nan)

        results.append({
            "feature": feat,
            "IV": iv,
            "Importance_RF": rf_importance[list(features).index(feat)],
            "Correlation": corr,
            "Logit_pval": pval,
            "VIF": vif_val
        })

    df_metrics = pd.DataFrame(results)

    # Rank each metric (lower p-value = better, so we invert rank)
    df_metrics["Rank_IV"] = df_metrics["IV"].rank(ascending=False)
    df_metrics["Rank_Importance"] = df_metrics["Importance_RF"].rank(ascending=False)
    df_metrics["Rank_Corr"] = df_metrics["Correlation"].abs().rank(ascending=False)
    df_metrics["Rank_Pval"] = df_metrics["Logit_pval"].rank(ascending=True)

    # Average rank
    df_metrics["Avg_Rank"] = df_metrics[["Rank_IV", "Rank_Importance", "Rank_Corr", "Rank_Pval"]].mean(axis=1)

    # Select top N
    top_features = df_metrics.sort_values("Avg_Rank").head(top_n)["feature"].tolist()

    # Drop duplicates if any
    top_features = list(dict.fromkeys(top_features))

    # Recalculate VIF for final set
    final_numeric = [f for f in top_features if np.issubdtype(df[f].dtype, np.number)]
    final_vif = calc_vif(df[final_numeric].fillna(0)) if final_numeric else {}

    return df_metrics, top_features, final_vif

# ----------------------
# Example usage
# ----------------------
# df = pd.read_csv("your_data.csv")
# metrics_df, final_feats, final_vif = feature_metrics(df, target="target_col", features=["f1","f2","f3"], top_n=50)
# print(metrics_df.head())
# print("Final selected:", final_feats)
# print("VIFs:", final_vif)
