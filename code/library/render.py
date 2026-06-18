from pathlib import Path
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from math import erfc, sqrt
import re

import maketables as mt
import numpy as np
import pandas as pd

from library.caching import run_regression_with_caching
from library.dictionary import VARIABLE_LABELS

mt.MTable.DEFAULT_TEX_STYLE.update(
    {
        "arraystretch": 0.5,
        "data_addlinespace": "0.25ex",
    }
)

FORMULA_SPLIT_PATTERN = re.compile(r"(\s+|[~+*:/|()])")
SHOCK_TABLE_TERMS = [
    "group:post:heat",
    "heat",
]
MARGINAL_HEAT_EFFECT_TERMS = ["heat", "group:heat", "group:post:heat"]
DIFFERENTIAL_HEAT_EFFECT_TERM = "differential_impact_heat"


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


def render_table_to_latex(table: mt.ETable, file_path: str | Path):
    def wrap_tabular_in_resizebox(latex: str, width: str = r"\linewidth") -> str:
        begin = r"\begin{tabular}"
        end = r"\end{tabular}"
        start = latex.find(begin)
        stop = latex.find(end, start)
        if start == -1 or stop == -1:
            raise ValueError("Expected maketables to emit a tabular environment")

        stop += len(end)
        return (
            latex[:start]
            + rf"\resizebox{{{width}}}{{!}}{{%"
            + "\n"
            + latex[start:stop]
            + "%"
            + "\n}"
            + latex[stop:]
        )

    latex = table.make(type="tex", tex_style={"tab_width": None})
    latex = wrap_tabular_in_resizebox(latex)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(latex)


def make_regression_table(
    specs: list[RegressionSpec],
    *,
    titles: list[str] | None = None,
    rename: dict[str, str] | list[dict[str, str]] | None = None,
    keep: list[str] | None = None,
    order: list[str] | None = None,
    **etable_kwargs,
) -> mt.ETable:
    """Run a set of regression specifications and combine results into a table."""

    renames = _renames_by_spec(rename, specs)
    models = [
        rename_terms(run_regression_with_caching(spec), rename_map)
        for spec, rename_map in zip(specs, renames, strict=True)
    ]

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
    return mt.ETable(models, **{**default_etable_kwargs, **etable_kwargs})  # ty:ignore[invalid-argument-type]


def make_shock_regression_table(
    specs: list[RegressionSpec],
    *,
    group: str | Sequence[str],
    post: str | Sequence[str | None],
    temperature: str | Sequence[str],
    **etable_kwargs,
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
    models = [run_regression_with_caching(spec) for spec in specs]

    effects: list[object] = []
    standard_errors: list[object] = []
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

        effect = float(contrast @ coefs)
        standard_error = float(np.sqrt(variance)) if variance >= 0 else np.nan
        t_stat = effect / standard_error if standard_error > 0 else np.nan
        p_value = erfc(abs(t_stat) / sqrt(2)) if not pd.isna(t_stat) else np.nan
        stars = ""
        if p_value < 0.01:
            stars = "***"
        elif p_value < 0.05:
            stars = "**"
        elif p_value < 0.10:
            stars = "*"

        effects.append(f"{effect:.3f}{stars}")
        standard_errors.append(
            f"{standard_error:.3f}" if not pd.isna(standard_error) else np.nan
        )

    custom_model_stats = {
        "Heat effect on treated": effects,
        "S.E.": standard_errors,
        **etable_kwargs.pop("custom_model_stats", {}),
    }
    return make_regression_table(
        specs,
        rename=renames,
        custom_model_stats=custom_model_stats,
        keep=SHOCK_TABLE_TERMS,
        **etable_kwargs,
    )


def make_shock_regression_table_trimmed(
    specs: list[RegressionSpec],
    *,
    group: str | Sequence[str],
    post: str | Sequence[str | None],
    temperature: str | Sequence[str],
    **etable_kwargs,
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
        {
            **group_rename,
            **post_rename,
            **temperature_rename,
            "group:post:heat": DIFFERENTIAL_HEAT_EFFECT_TERM,
        }
        for group_rename, post_rename, temperature_rename in zip(
            group_renames, post_renames, temperature_renames, strict=True
        )
    ]

    return make_regression_table(
        specs,
        rename=renames,
        keep=[DIFFERENTIAL_HEAT_EFFECT_TERM, "heat"],
        exact_match=True,
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
    if text in replacements:
        return replacements[text]
    for old_term, new_term in replacements.items():
        text = search_replace_term(text, old_term, new_term)[0]
    if text in replacements:
        return replacements[text]
    return text
