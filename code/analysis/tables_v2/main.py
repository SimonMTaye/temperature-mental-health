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
from analysis.tables_v2.table_i_economics import make_table as table_i
from analysis.tables_v2.table_j_palm_farmer_selfemp import make_table as table_j
from analysis.tables_v2.table_k_jobloss_voluntary import make_table as table_k
from analysis.tables_v2.table_l_temperature_shock_cesd_breakdown import (
    make_table as table_l,
)
from analysis.tables_v2.table_m_fuel_share_het import make_table as table_m

from library.log import log


def main() -> None:
    log("Table A Started")
    table_a()
    log("Table A completed")
    log("Table B Started")
    table_b()
    log("Table B completed")
    log("Table C Started")
    table_c()
    log("Table C completed")
    log("Table D Started")
    table_d()
    log("Table D completed")
    log("Table E Started")
    table_e()
    log("Table E completed")
    log("Table E Started")
    table_f()
    log("Table F completed")
    log("Table G Started")
    table_g()
    log("Table G completed")
    log("Table H Started")
    table_h()
    log("Table H completed")
    log("Table I Started")
    table_i()
    log("Table I completed")
    log("Table J Started")
    table_j()
    log("Table J completed")
    log("Table K Started")
    table_k()
    log("Table K completed")
    log("Table L Started")
    table_l()
    log("Table L completed")
    log("Table M Started")
    table_m()
    log("Table M completed")


if __name__ == "__main__":
    main()
