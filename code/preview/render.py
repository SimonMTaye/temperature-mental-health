from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import re

import pandas as pd
import maketables as mt

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
    specs: list[RegressionSpec], *, titles: list[str] | None
) -> mt.ETable:
    """Run a set of regression specifications and combine results into a table."""

    models = [run_regression_with_caching(spec) for spec in specs]

    terms = list(
        set(
            term
            for spec in specs
            if spec.show_terms is not None
            for term in spec.show_terms
        )
    )

    if not titles or (len(titles) != len(specs)):
        raise ValueError("titles must be provided and have one value per spec")
    model_heads = titles if titles else [spec.title for spec in specs]

    return mt.ETable(
        models,
        model_heads=model_heads,
        keep=terms,
        labels=VARIABLE_LABELS,
        felabels=VARIABLE_LABELS,
    )


def make_shock_regression_table(
    specs: list[RegressionSpec],
    *,
    group: str | Sequence[str],
    post: str | Sequence[str],
    temperature: str | Sequence[str],
) -> mt.ETable:
    """Run shock regressions and align group/post/heat terms across models."""

    groups = _terms_by_spec(group, specs)
    posts = _terms_by_spec(post, specs)
    temperatures = _terms_by_spec(temperature, specs)

    models = [
        _canonicalize_shock_model(
            run_regression_with_caching(spec),
            group=group_term,
            post=post_term,
            temperature=temperature_term,
        )
        for spec, group_term, post_term, temperature_term in zip(
            specs, groups, posts, temperatures, strict=True
        )
    ]

    return mt.ETable(
        models,
        model_heads=[spec.title for spec in specs],
        keep=SHOCK_TABLE_TERMS,
        order=SHOCK_TABLE_TERMS,
        exact_match=True,
        labels={**VARIABLE_LABELS},
        felabels=VARIABLE_LABELS,
    )


def _terms_by_spec(
    terms: str | Sequence[str],
    specs: list[RegressionSpec],
) -> list[str]:
    """Return one canonicalization term per spec.

    A single string is broadcast to every spec. A sequence must contain exactly
    one value per spec, because otherwise per-spec term replacement would be
    ambiguous.
    """
    if isinstance(terms, str):
        return [terms] * len(specs)
    if len(terms) != len(specs):
        raise ValueError(f"{terms} must be a string or have one value per spec")
    return list(terms)


def _canonicalize_shock_model(
    model,
    *,
    group: str,
    post: str,
    temperature: str,
):
    coef_table = model.coef_table.copy()
    replacements = [
        (group, "group"),
        (post, "post"),
        (temperature, "heat"),
    ]
    coef_table.index = pd.Index(
        [
            _replace_terms(coefficient, replacements)
            for coefficient in coef_table.index.astype(str)
        ],
        name=coef_table.index.name,
    )
    return replace(model, coef_table=coef_table)


def _replace_terms(text: str, replacements: list[tuple[str, str]]) -> str:
    for old_term, new_term in replacements:
        text = search_replace_term(text, old_term, new_term)[0]
    return text
