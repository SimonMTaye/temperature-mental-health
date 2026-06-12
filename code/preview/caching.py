import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyfixest as pf

CACHE_VERSION = "v2"
CACHE_DIR = Path(".cache") / "regressions" / CACHE_VERSION
DEFAULT_MODEL_STATS = ["N", "r2"]


@dataclass(frozen=True)
class CachedRegressionResult:
    coef_table: pd.DataFrame
    vcov: pd.DataFrame
    depvar: str
    fixef: str | None
    stats: dict[str, Any]
    vcov_info: dict[str, Any]
    var_labels: dict[str, str] | None = None
    default_stat_keys: list[str] | None = None

    @property
    def __maketables_coef_table__(self) -> pd.DataFrame:
        return self.coef_table

    @property
    def __maketables_depvar__(self) -> str:
        return self.depvar

    @property
    def __maketables_fixef_string__(self) -> str | None:
        return self.fixef

    @property
    def __maketables_vcov_info__(self) -> dict[str, Any]:
        return self.vcov_info

    @property
    def __maketables_var_labels__(self) -> dict[str, str] | None:
        return self.var_labels

    @property
    def __maketables_default_stat_keys__(self) -> list[str] | None:
        return self.default_stat_keys

    def __maketables_stat__(self, key: str) -> Any:
        return self.stats.get(key)


def dataframe_digest(df: pd.DataFrame) -> str:
    h = hashlib.blake2b(digest_size=16)
    row_hashes = pd.util.hash_pandas_object(df, index=True)
    h.update(row_hashes.to_numpy().tobytes())
    h.update(repr(tuple(df.columns)).encode())
    h.update(repr(tuple(str(dtype) for dtype in df.dtypes)).encode())
    return h.hexdigest()


def cache_key(formula: str, df: pd.DataFrame) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(formula.encode())
    h.update(dataframe_digest(df).encode())
    return h.hexdigest()


def run_regression_with_caching(
    spec: Any,
    cache_dir: Path = CACHE_DIR,
) -> CachedRegressionResult:
    df = spec.df if isinstance(spec.df, pd.DataFrame) else pd.read_parquet(spec.df)
    cache_path = cache_dir / f"{cache_key(spec.formula, df)}.pkl"

    if cache_path.exists():
        cached_result = load_cached_result(cache_path)
        if cached_result is not None:
            return cached_result

    result = cached_result_from_model(pf.feols(spec.formula, data=df))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(result, f)
    tmp_path.replace(cache_path)
    return result


def load_cached_result(cache_path: Path) -> CachedRegressionResult | None:
    try:
        with cache_path.open("rb") as f:
            cached_result = pickle.load(f)
    except (EOFError, pickle.UnpicklingError, AttributeError, ModuleNotFoundError):
        return None
    if isinstance(cached_result, CachedRegressionResult) and hasattr(
        cached_result, "vcov"
    ):
        return cached_result
    return None


def cached_result_from_model(model: Any) -> CachedRegressionResult:
    coefficients = model.coef()
    return CachedRegressionResult(
        coef_table=coef_table_from_model(model),
        vcov=pd.DataFrame(model._vcov, index=coefficients.index, columns=coefficients.index),
        depvar=getattr(model, "_depvar", "y"),
        fixef=getattr(model, "_fixef", None),
        stats=stats_from_model(model),
        vcov_info={
            "vcov_type": getattr(model, "_vcov_type", None),
            "clustervar": getattr(model, "_clustervar", None),
        },
        default_stat_keys=DEFAULT_MODEL_STATS,
    )


def coef_table_from_model(model: Any) -> pd.DataFrame:
    coef_table = model.tidy()
    rename_map = {
        "Estimate": "b",
        "Std. Error": "se",
        "Pr(>|t|)": "p",
        "t value": "t",
        "2.5%": "ci95l",
        "97.5%": "ci95u",
    }
    coef_table = coef_table.rename(
        columns={old: new for old, new in rename_map.items() if old in coef_table}
    )
    coef_table.index.name = "Coefficient"

    front_cols = [col for col in ["b", "se", "t", "p"] if col in coef_table]
    other_cols = [col for col in coef_table.columns if col not in front_cols]
    return coef_table[front_cols + other_cols]


def stats_from_model(model: Any) -> dict[str, Any]:
    return {
        "N": int(model._N) if getattr(model, "_N", None) is not None else None,
        "n_clusters": first_or_none(getattr(model, "_G", None)),
        "se_type": se_type_from_model(model),
        "r2": getattr(model, "_r2", None),
        "adj_r2": getattr(model, "_r2_adj", None),
        "r2_within": getattr(model, "_r2_within", None),
        "adj_r2_within": getattr(model, "_adj_r2_within", None),
        "rmse": getattr(model, "_rmse", None),
        "fvalue": getattr(model, "_F_stat", None),
        "f_statistic": getattr(model, "_f_stat_1st_stage", None),
        "deviance": first_or_self(getattr(model, "deviance", None)),
    }


def se_type_from_model(model: Any) -> str | None:
    vcov_type = getattr(model, "_vcov_type", None)
    clustervar = getattr(model, "_clustervar", None)
    if vcov_type == "CRV" and clustervar:
        return "by: " + "+".join(clustervar)
    return vcov_type


def first_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return value[0] if len(value) else None
    return value


def first_or_self(value: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return value[0] if len(value) else None
    return value
