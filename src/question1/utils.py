from typing import Literal
import os
import pandas as pd

int32 = pd.Int32Dtype()  # Nullable integer type
source_names = ['Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas',
                'Small hydro', 'Coal', 'Nuclear', 'Natural gas', 'Large hydro',
                'Batteries', 'Imports', 'Other']
sources_dtype = dict.fromkeys(source_names, int32)
demand_dtype = dict.fromkeys(
    ['Day ahead forecast', 'Hour ahead forecast', 'Current demand'], int32
)


def read(kind: Literal['sources', 'demand']):
    sep = os.path.sep
    df = pd.read_csv(
        f'dataset{sep}{kind}.csv',
        dtype=sources_dtype if kind == 'sources' else demand_dtype,
        index_col='Datetime'
    )
    df.index = pd.to_datetime(df.index)
    return df


def read_day(filename: str, kind: Literal['sources', 'demand']):
    if kind not in ['sources', 'demand']:
        raise ValueError("Kind must be either 'sources' or 'demand'")

    # Parse file
    try:
        df = pd.read_csv(
            f'../../dataset/{kind}/{filename}',
            dtype=sources_dtype if kind == 'sources' else demand_dtype,
            header=0,
            names=['Time'] + source_names if kind == 'sources' else None
        )[:-1]  # Ignore last row
    except pd.errors.EmptyDataError:
        return

    df.rename(columns={'Time': 'Datetime'}, inplace=True)
    # Use filename to add date to data
    df['Datetime'] = filename.replace('.csv', '') + ' ' + df['Datetime']
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df


def merge_days():
    for kind in ['sources', 'demand']:
        filenames = os.listdir(f'../../dataset/{kind}')
        # Parse every .csv and merge them all into one DataFrame
        df = pd.concat([read_day(filename, kind) for filename in filenames])
        # Reindex to get rid of duplicates
        df.reset_index()

        print(df.info())

        df.to_csv(f'../../dataset/{kind}.csv', index=False)

# If already pickled, load the data from pickled files for speed. Otherwise read .csv files and pickle them for future use. Return tuple with DataFrames (sources, demand).
def load_data() -> tuple:
    try:
        sources = pd.read_pickle('dataset' + os.path.sep + 'pickleJar' + os.path.sep + 'sources.pkl')
        demand = pd.read_pickle('dataset' + os.path.sep + 'pickleJar' + os.path.sep + 'demand.pkl')
        print('Pickles retrieved')
    except:
        sources = read("sources")
        demand = read("demand")
        sources.to_pickle('dataset' + os.path.sep + 'pickleJar' + os.path.sep + 'sources.pkl')
        demand.to_pickle('dataset' + os.path.sep + 'pickleJar' + os.path.sep + 'demand.pkl')
        print('Data pickled')
    finally:
        return (sources, demand)

# Proprocessing
def fill_nan(raw: pd.DataFrame) -> pd.DataFrame:
    for name, values in raw.iteritems():
        raw[name].fillna(round(raw[name].median(skipna = True)), inplace = True)
    print("Preprocessing complete")
    return raw


if __name__ == '__main__':
    merge_days()
