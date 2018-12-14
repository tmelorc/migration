

    


**Required folders and files**

- migration_matrices
- dendrograms
- ultra_metric
- `Net_Gross_US.txt` 

Raw data file downloaded from [census.gov](https://www2.census.gov/programs-surveys/demo/tables/geographic-mobility/2015/county-to-county-migration-2011-2015/county-to-county-migration-flows/Net_Gross_US.txt)
- prepare_data_fields.csv
- prepare_data_states_num_counties.csv

**Execution order**

- `prepare_data.py`
- `compute_ultrametric.py`
- `plot_dendrogram.py`
- `plot_VR.py` (not working yet)
