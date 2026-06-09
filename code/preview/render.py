from dataclasses import dataclass, field

import pandas as pd
import maketables as mt

from caching import run_regression_with_caching
from dictionary import VARIABLE_LABELS


@dataclass(frozen=True)
class RegressionSpec:
    """Specification for a regression test case."""

    title: str
    formula: str
    df: pd.DataFrame | str = field(compare=False, hash=False)
    tags: frozenset[str]
    show_terms: frozenset[str] | None


def make_regression_table(specs: list[RegressionSpec]) -> mt.ETable:
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
    return mt.ETable(
        models,
        model_heads=[spec.title for spec in specs],
        keep=terms,
        labels=VARIABLE_LABELS,
        felabels=VARIABLE_LABELS,
    )
