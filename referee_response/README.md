# Code for the referee-response tables

This code is only for the response to referees. It is not part of the main paper's
replication package. In the repository it lives in its own folder, `referee_response/`,
separate from `code/`.

## built_from_repo_data/

Run from the root of the GitHub repository (main e7d07d5). Copy the script into
`code/analysis/updates/` and run, for example:

    uv run python code/analysis/updates/palm_identification_controls.py

| Response table | File |
|---|---|
| R3.1a, R3.1b | palm_identification_controls.py |
| R3.1 ageing check | palm_age_group_slopes.py |
| R3.2a | palm_four_group_table.py |
| R3.9 | temperature_match_level_table.py |
| R3.10 | palm_clustering_table.py |
| R2 minor 1 | economics_winsorizing_table.py |
| R2 minor 2 | temperature_quadratic_table.py |
| R3.4 (mean temperature row) | R3_4_table_f_job_loss_controls_fixed.py |

The R3.4 file is the repository's `tables_v2/table_f_job_loss_controls.py` with all
sectors entered in the sector x heat column (on main only the last sector enters).

## other_tables/

These use raw IFLS variables that the repository's data build does not produce, so
they read files from the Dropbox project folder and have their own paths.

| Response table | Folder |
|---|---|
| R2.1 placebo outcomes | R2_1_placebo |
| R2.3 / R3.3 cutoff figure | R2_3_cutoff_sweep |
| R3.1c non-movers and other R3.1 checks | R3_1c_and_R3_1_checks |
| R3.1d housing and assets | R3_1d_housing_assets |
| R3.2c palm and rubber with crop controls | R3_2c_palm_rubber |
| R3.3 both sides of the cut | R3_3_fuel_both_sides |
| Joint test of the three interactions | joint_null_test |
| R2.3 / R3.3 event-time figure, 15-day windows | R2_3_event_study_15day |
| R3.1e panel ageing | R3_1e_panel_ageing |
| R2.4 equality of the three heat interactions | R2_4_equality_test |

The last three are the exception in this folder: each is one script that needs only
`data/generated/30_analysis_table_input.parquet` and runs from the repository root.

## Already on GitHub (Simon's scripts in code/analysis/updates/)

R2.2d CES-D cutoffs (`cesd_threshold_table_b.py`), R3.6 magnitudes
(`residual_temperature_magnitudes.py`), fuel cutoff placebo, both-sides table, map.
