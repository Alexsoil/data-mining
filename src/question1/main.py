from typing import Literal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dbscan import outlier_detector
import utils

sns.set_theme()

temp = utils.load_data()
sources = utils.fill_nan(temp[0])
demand = utils.fill_nan(temp[1])
# print(sources)
# print(sources.describe())
# print(demand)
# print(demand.describe())

# outlier_detector(sources, utils.source_names)
# outlier_detector(demand, utils.demand_dtype)


def all_sources():
    sample = sources.resample("1H").mean().fillna(0)
    sample = sample.apply(lambda x: x.rolling("60D", min_periods=24 * 30).mean())

    sample['Date'] = sample.index
    df = (
        sample.melt(['Date'])
        .rename(columns={"variable": "Source", "value": "Megawatts"})
    )
    sns.lineplot(x='Date', y='Megawatts', hue="Source", data=df)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def duck_curve():
    start = pd.to_datetime('2021-08-15 00:00:00')
    end = pd.to_datetime('2021-08-15 23:55:00')
    day_sources = sources.loc[start:end]
    day_demand = demand.loc[start:end]['Current demand']
    day_dispatchable = day_sources.drop(['Solar', 'Wind'], axis=1)

    df = pd.DataFrame()
    df.index = day_sources.index
    df['Demand'] = day_demand
    df['Solar & wind'] = day_sources['Solar'] + day_sources['Wind']
    df['Dispatchable'] = day_dispatchable.sum(axis=1)

    df.index = df.index.hour

    df = df.melt(ignore_index=False).reset_index()
    df = df.rename(columns={
        "variable": "Variable",
        "value": "Megawatts",
        "Datetime": "Hour"
    })

    sns.lineplot(x=df['Hour'], y='Megawatts', hue='Variable', data=df).set_title(
        "Hourly demand vs. dispatchable sources vs. solar & wind (August 15, 2021)"
    )

    plt.xticks(range(0, 24, 2))
    plt.show()


all_sources()


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



