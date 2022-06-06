from typing import Literal
import pandas as pd
import seaborn as sns
from utils import read

sources = read("sources")
demand = read("demand")

print(sources)
print(sources.describe())
# print(demand.describe())


def supply_demand():
    supply_sample = sources.resample("1H").mean().sum(axis=1, numeric_only=True)
    demand_sample = demand.resample("1H").mean()

    df = pd.DataFrame()
    df.index = supply_sample.index

    df["Supply"] = supply_sample.rolling("30D", min_periods=24 * 7).mean()
    df["Demand"] = (
        demand_sample["Current demand"].rolling("30D", min_periods=24 * 7).mean()
    )
    df = (
        df.melt(ignore_index=False)
        .reset_index()
        .rename(columns={"variable": "Variable", "value": "Value"})
    )

    sns.lineplot(data=df, x="Datetime", y="Value", hue="Variable").set_title(
        "Supply and demand across time (30-day rolling average)"
    )


def percent_renewable(rolling_window: Literal['30D', '180D']):
    """Graph the share of renewable sources across time."""
    sample = sources.resample("1H").mean()
    df = pd.DataFrame()
    df.index = sample.index
    renewable = (
        sample["Solar"]
        + sample["Wind"]
        + sample["Geothermal"]
        + sample["Small hydro"]
        + sample["Large hydro"]
    )

    total = sample.sum(axis=1, numeric_only=True)
    total_no_imports = sample.drop("Imports", axis=1).sum(axis=1, numeric_only=True)

    percent = 100 * renewable / total
    percent_no_imports = 100 * renewable / total_no_imports

    df["w/ imports"] = percent.rolling(rolling_window, min_periods=24).mean()
    df["no imports"] = percent_no_imports.rolling(rolling_window, min_periods=24).mean()
    df = (
        df.melt(ignore_index=False)
        .reset_index()
        .rename(columns={"variable": "Imports included", "value": "Percentage"})
    )
    sns.lineplot(
        data=df, x="Datetime", y="Percentage", hue="Imports included"
    ).set_title(f"Share of energy production from renewable sources ({rolling_window} rolling average)")


def source_mix():
    sample = sources.resample("1H").mean().fillna(0)
    means = sample.mean()
    mix = (
        means.sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Source", 0: "Percentage"})
    )
    mix["Percentage"] /= means.sum() / 100
    sns.barplot(data=mix, x="Percentage", y="Source").set_title("Energy production mix")


sns.set_theme()
