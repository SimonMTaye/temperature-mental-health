from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import re

import maketables as mt
import numpy as np
import pandas as pd

from caching import run_regression_with_caching
from dictionary import VARIABLE_LABELS

FORMULA_SPLIT_PATTERN = re.compile(r"(\s+|[~+*:/|()])")
SHOCK_TABLE_TERMS = [
    "group:post:heat",
    "group:heat",
    "heat",
    "group:post",
    "group",
    "post:heat",
    "post",
]
MARGINAL_HEAT_EFFECT_TERMS = ["heat", "group:heat", "group:post:heat"]


@dataclass(frozen=True)
class RegressionSpec:
    """Specification for a regression test case."""

    title: str
    formula: str
    df: pd.DataFrame | str = field(compare=False, hash=False)
    tags: frozenset[str]
    show_terms: frozenset[str] | None


def search_replace_term(text: str, old_term: str, new_term: str) -> tuple[str, bool]:
    """Replace whole formula tokens in text, preserving separators."""
    parts = FORMULA_SPLIT_PATTERN.split(text)
    found = False
    for index, part in enumerate(parts):
        if part == old_term:
            parts[index] = new_term
            found = True
    return "".join(parts), found


def make_regression_table(
    specs: list[RegressionSpec],
    *,
    titles: list[str] | None = None,
    rename: dict[str, str] | list[dict[str, str]] | None = None,
    keep: list[str] | None = None,
    order: list[str] | None = None,
    **etable_kwargs: object,
) -> mt.ETable:
    """Run a set of regression specifications and combine results into a table."""

    renames = _renames_by_spec(rename, specs)
    models = [
        rename_terms(run_regression_with_caching(spec), rename_map)
        for spec, rename_map in zip(specs, renames, strict=True)
    ]

    return _make_etable(
        models,
        specs,
        renames,
        titles=titles,
        keep=keep,
        order=order,
        **etable_kwargs,
    )


def _make_etable(
    models: list[object],
    specs: list[RegressionSpec],
    renames: list[dict[str, str]],
    *,
    titles: list[str] | None = None,
    keep: list[str] | None = None,
    order: list[str] | None = None,
    **etable_kwargs: object,
) -> mt.ETable:
    """Combine fitted regression results into a table."""

    if titles and (len(titles) != len(specs)):
        raise ValueError("Length of titles must match length of specs")
    model_heads = titles if titles else [spec.title for spec in specs]

    if keep is None:
        keep = list(
            set(
                _replace_terms(term, rename_map)
                for spec, rename_map in zip(specs, renames, strict=True)
                if spec.show_terms is not None
                for term in spec.show_terms
            )
        )
    default_etable_kwargs = {
        "model_heads": model_heads,
        "keep": keep,
        "order": order,
        "exact_match": False,
        "labels": VARIABLE_LABELS,
        "felabels": VARIABLE_LABELS,
    }
    return mt.ETable(models, **{**default_etable_kwargs, **etable_kwargs})


def make_shock_regression_table(
    specs: list[RegressionSpec],
    *,
    group: str | Sequence[str],
    post: str | Sequence[str],
    temperature: str | Sequence[str],
    **etable_kwargs: object,
) -> mt.ETable:
    """Run shock regressions and align group/post/heat terms across models."""

    def _rename_to(
        terms: str | Sequence[str],
        new_term: str,
    ) -> dict[str, str] | list[dict[str, str]]:
        if isinstance(terms, str):
            return {terms: new_term}
        return [{term: new_term} for term in terms]

    group_renames = _renames_by_spec(_rename_to(group, "group"), specs)
    post_renames = _renames_by_spec(_rename_to(post, "post"), specs)
    temperature_renames = _renames_by_spec(_rename_to(temperature, "heat"), specs)
    renames = [
        {**group_rename, **post_rename, **temperature_rename}
        for group_rename, post_rename, temperature_rename in zip(
            group_renames, post_renames, temperature_renames, strict=True
        )
    ]
    models = [
        rename_terms(run_regression_with_caching(spec), rename_map)
        for spec, rename_map in zip(specs, renames, strict=True)
    ]

    effects: list[float] = []
    standard_errors: list[float] = []
    for model in models:
        coefs = model.coef_table["b"]
        if any(term not in coefs.index for term in MARGINAL_HEAT_EFFECT_TERMS):
            effects.append(np.nan)
            standard_errors.append(np.nan)
            continue

        vcov = model.vcov.reindex(index=coefs.index, columns=coefs.index)
        contrast = pd.Series(0.0, index=coefs.index)
        contrast.loc[MARGINAL_HEAT_EFFECT_TERMS] = 1.0
        variance = float(contrast @ vcov @ contrast)

        effects.append(float(contrast @ coefs))
        standard_errors.append(float(np.sqrt(variance)) if variance >= 0 else np.nan)

    custom_model_stats = {
        "Heat effect on treated": effects,
        "S.E.": standard_errors,
        **etable_kwargs.pop("custom_model_stats", {}),
    }

    return _make_etable(
        models,
        specs,
        renames,
        keep=SHOCK_TABLE_TERMS,
        order=SHOCK_TABLE_TERMS,
        custom_model_stats=custom_model_stats,
        **etable_kwargs,
    )


def _renames_by_spec(
    rename: dict[str, str] | list[dict[str, str]] | None,
    specs: list[RegressionSpec],
) -> list[dict[str, str]]:
    """Return one term-renaming dictionary per spec."""
    if rename is None:
        return [{}] * len(specs)
    if isinstance(rename, dict):
        return [rename] * len(specs)
    if len(rename) != len(specs):
        raise ValueError("rename must be a dict or have one dict per spec")
    return rename


def rename_terms(model, replacements: dict[str, str]):
    """Return model with coefficient terms renamed by whole formula tokens."""
    coef_table = model.coef_table.copy()
    coef_table.index = pd.Index(
        [
            _replace_terms(coefficient, replacements)
            for coefficient in coef_table.index.astype(str)
        ],
        name=coef_table.index.name,
    )

    vcov = model.vcov.copy()
    vcov.index = pd.Index(
        [_replace_terms(term, replacements) for term in vcov.index.astype(str)],
        name=vcov.index.name,
    )
    vcov.columns = pd.Index(
        [_replace_terms(term, replacements) for term in vcov.columns.astype(str)],
        name=vcov.columns.name,
    )
    return replace(model, coef_table=coef_table, vcov=vcov)


def _replace_terms(text: str, replacements: dict[str, str]) -> str:
    for old_term, new_term in replacements.items():
        text = search_replace_term(text, old_term, new_term)[0]
    return text
