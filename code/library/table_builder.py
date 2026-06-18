def make_row(label: str, coefficients: list[str] | list[int]) -> str:
    # handle string case
    if isinstance(coefficients[0], str):
        return label + " & " + " & ".join(coefficients) + r" \\"
    if isinstance(coefficients[0], int):
        return label + " & " + " & ".join(f"{coef:,}" for coef in coefficients) + r" \\"


def stars(p_value: float) -> str:
    # check if p_value is nan
    if not p_value:
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def coefficient_rows(
    label: str,
    stats: list[tuple[float, float, float]],
    *,
    show_p_value: bool = False,
    formatting: str = ".3f",
    coefficient_format: str | None = None,
    standard_error_format: str | None = None,
    p_value_format: str | None = None,
) -> tuple:
    coefficient_format = coefficient_format or formatting
    standard_error_format = standard_error_format or formatting
    p_value_format = p_value_format or formatting
    coefficient_cells = [
        f"{coefficient:{coefficient_format}}{stars(p_value)}"
        for coefficient, _, p_value in stats
    ]
    standard_error_cells = [
        f"({standard_error:{standard_error_format}})" for _, standard_error, _ in stats
    ]
    if not show_p_value:
        return make_row(label, coefficient_cells), make_row("", standard_error_cells)
    else:
        p_value_cells = [f"[{p_value:{p_value_format}}]" for _, _, p_value in stats]
        return (
            make_row(label, coefficient_cells),
            make_row("", standard_error_cells),
            make_row("", p_value_cells),
        )
