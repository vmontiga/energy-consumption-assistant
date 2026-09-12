from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_ID_COL = "customer_id"


def feature_columns(df: pd.DataFrame, id_col: str = DEFAULT_ID_COL, exclude=None) -> list[str]:
    """Every numeric column except the id column and anything explicitly
    excluded (e.g. informational columns that aren't meant as ML features)."""
    exclude = set(exclude or []) | {id_col}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def evaluate_k_range(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    k_range=range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """Impute + scale ONCE, then fit a plain KMeans for each k in k_range,
    recording inertia and silhouette score. Use this to choose k before
    calling build_and_fit with the chosen value."""
    X = features_df[feature_cols].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 and k < len(X_scaled) else np.nan
        rows.append({"k": k, "inertia": km.inertia_, "silhouette_score": sil})
    return pd.DataFrame(rows)


def build_and_fit(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int,
    random_state: int = 42,
) -> tuple[Pipeline, np.ndarray]:
    """Fit the final imputer+scaler+KMeans pipeline as one object. Returns
    (pipeline, cluster_labels) - persist `pipeline` (e.g. via joblib) to
    reuse it later without retraining."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)),
    ])
    X = features_df[feature_cols].to_numpy(dtype=float)
    labels = pipeline.fit_predict(X)
    return pipeline, labels


def predict_cluster(pipeline: Pipeline, features_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Predict cluster(s) for new customer(s) using an already-fitted
    pipeline - e.g. for a freshly-uploaded customer in the Streamlit app.
    `features_df` must have the same feature_cols as the training data
    (missing values are fine - the persisted imputer handles them)."""
    X = features_df[feature_cols].to_numpy(dtype=float)
    return pipeline.predict(X)
