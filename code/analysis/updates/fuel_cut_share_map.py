"""Map each kecamatan's share of IFLS5 interviews after the fuel subsidy cut.

Run from the repo root:

    uv run python code/analysis/updates/fuel_cut_share_map.py

Writes output/figures/fuel_cut_share_after_map.png.

Purpose: show where the pre/post comparison behind the fuel-shock columns can
come from. With kecamatan fixed effects, only kecamatans interviewed on both
sides of the Nov 18, 2014 cut contribute to it. Those are colored by their
share of interviews after the cut; kecamatans interviewed entirely before or
entirely after are gray. The caption reports how many kecamatans are gray and
the correlation between the share after and the kecamatan's median 7-day mean
temperature among the colored ones (a check that fieldwork timing around the
cut does not line up with hot or cool places).

Geography is the GADM level-3 (kecamatan) polygon matched in step 02. The few
codes that only matched at kabupaten or province level are dropped so large
polygons do not cover their neighbours. Kecamatans with fewer than 5 IFLS5
respondents (mostly movers; ~7% of respondents) are dropped because their
shares are noise. The extent is cropped to the 0.5-99.5th percentile of
kecamatan centroids.
"""

import geopandas as gpd
import pandas as pd
from data.config import GENERATED_DATA
from library.config import FIGURE_OUTPUT
from library.specs import MAIN_TEMP_MEASURE
from mizani.labels import label_percent
from plotnine import (
    aes,
    coord_fixed,
    element_blank,
    geom_map,
    ggplot,
    labs,
    scale_fill_cmap,
    theme,
    theme_void,
)
from shapely import wkt

MIN_RESPONDENTS = 5
ONE_SIDE_GRAY = "#D9D9D9"


def kecamatan_polygons() -> gpd.GeoDataFrame:
    polygons = pd.read_parquet(GENERATED_DATA / "02_kabupaten_polygons.parquet")
    polygons = polygons.loc[polygons["match_level"].eq("kecamatan")]
    return gpd.GeoDataFrame(
        polygons[["gadm_fullcode"]],
        geometry=polygons["geometry_wkt"].map(wkt.loads),
        crs=4326,
    )


def kecamatan_summaries() -> pd.DataFrame:
    interviews = pd.read_parquet(
        GENERATED_DATA / "30_analysis_table_input.parquet",
        columns=["wave", "gadm_fullcode", "post_subsidy", MAIN_TEMP_MEASURE],
    )
    interviews = interviews.loc[interviews["wave"].eq("IFLS5")]
    by_kecamatan = interviews.groupby("gadm_fullcode")
    summaries = pd.DataFrame(
        {
            "respondents": by_kecamatan.size(),
            "share_after": by_kecamatan["post_subsidy"].mean().astype(float),
            "median_temperature": by_kecamatan[MAIN_TEMP_MEASURE].median(),
        }
    ).reset_index()
    return summaries.loc[summaries["respondents"].ge(MIN_RESPONDENTS)]


def caption(kecamatans: gpd.GeoDataFrame) -> str:
    both_sides = kecamatans["share_after"].between(0, 1, inclusive="neither")
    correlation = kecamatans.loc[both_sides, "share_after"].corr(
        kecamatans.loc[both_sides, "median_temperature"]
    )
    return (
        f"Gray: interviews on one side of the cut only "
        f"({kecamatans['share_after'].eq(1).mean():.0%} of kecamatans all after, "
        f"{kecamatans['share_after'].eq(0).mean():.0%} all before).\n"
        f"Colored kecamatans ({both_sides.mean():.0%}): correlation of share after "
        f"with median 7-day mean temperature = {correlation:.2f}."
    )


def share_after_map(kecamatans: gpd.GeoDataFrame):
    both_sides = kecamatans["share_after"].between(0, 1, inclusive="neither")
    centroids = kecamatans.geometry.representative_point()
    xlim = tuple(centroids.x.quantile([0.005, 0.995]) + [-0.5, 0.5])
    ylim = tuple(centroids.y.quantile([0.005, 0.995]) + [-0.5, 0.5])
    return (
        ggplot(kecamatans)
        + geom_map(data=kecamatans.loc[~both_sides], fill=ONE_SIDE_GRAY, color=None)
        + geom_map(aes(fill="share_after"), data=kecamatans.loc[both_sides], color=None)
        + scale_fill_cmap(
            "viridis", limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1],
            labels=label_percent(),
        )
        + coord_fixed(xlim=xlim, ylim=ylim)
        + labs(
            fill="IFLS5 share of interviews after the Nov 18, 2014 fuel subsidy cut",
            caption=caption(kecamatans),
        )
        + theme_void(base_size=11)
        + theme(
            legend_position="bottom",
            legend_title_position="top",
            legend_key_width=400,
            legend_key_height=8,
            panel_grid=element_blank(),
        )
    )


kecamatans = kecamatan_polygons().merge(kecamatan_summaries(), on="gadm_fullcode")
FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
share_after_map(kecamatans).save(
    FIGURE_OUTPUT / "fuel_cut_share_after_map.png",
    width=11, height=6, dpi=300, verbose=False,
)
