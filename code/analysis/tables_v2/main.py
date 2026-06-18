from analysis.tables_v2.table_a_temperature_effects import make_table as table_a
from analysis.tables_v2.table_b_temperature_and_shock_effects import (
    make_table as table_b,
)
from analysis.tables_v2.table_c_shock_balances import make_table as table_c
from analysis.tables_v2.table_d_palm_farmer_alt_definitions import make_table as table_d
from analysis.tables_v2.table_e_job_loss_window_robustness import make_table as table_e
from analysis.tables_v2.table_f_job_loss_controls import make_table as table_f
from analysis.tables_v2.table_g_fuel_shock_alt_definitions import make_table as table_g
from analysis.tables_v2.table_h_sumstats import make_table as table_h
from library.log import log


def main() -> None:
    table_a()
    log("Table A completed")
    table_b()
    log("Table B completed")
    table_c()
    log("Table C completed")
    table_d()
    log("Table D completed")
    table_e()
    log("Table E completed")
    table_f()
    log("Table F completed")
    table_g()
    log("Table G completed")
    table_h()
    log("Table H completed")


if __name__ == "__main__":
    main()
