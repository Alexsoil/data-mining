import pandas as pd
import numpy as np
import utils
import os
import datetime
import keras

print("H portra kollhse")
try:
    renewable = pd.read_pickle('dataset' + os.path.sep + 'pickleJar' + os.path.sep + 'renewables.pkl')
    print('Pickles retrieved')
except:
    temp = utils.load_data()
    sources = utils.fill_nan(temp[0])
    demand = utils.fill_nan(temp[1])
    # print(sources)
    # print(demand)
    # According to the U.S Energy Information Administration
    renewable = sources[['Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Large hydro']].copy()
    renewable['Total'] = renewable.sum(axis = 1)
    renewable['Datetime'] = renewable.index
    renewable[['DayofYear', 'Hour', 'Minute']] = np.nan
    # Convert datetime format to day of year (1-365), hours and minutes
    print("Formating data, this may take a while...")
    for idx, row in renewable.iterrows():
        renewable.loc[idx, 'DayofYear'] = int(row['Datetime'].timetuple().tm_yday)
        renewable.loc[idx, 'Hour'] = int(row['Datetime'].time().hour)
        renewable.loc[idx, 'Minute'] = int(row['Datetime'].time().minute)
    renewable.to_pickle('dataset' + os.path.sep + 'pickleJar' + os.path.sep + 'renewables.pkl')
    print("Data Pickled")
print(renewable)
