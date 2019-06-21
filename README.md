# Migration networks

(...)

## USA migration

(...)

#### Required folders and files

##### Folders

You should have the following (possibly empty) folders in your main folder:

- `migration_matrices`
- `dendrograms`
- `ultra_metric`

##### Files

You should have the following files in your main folder:

-  `my_shortest_path.so`
- `Net_Gross_US.txt` from [census.gov](https://www2.census.gov/programs-surveys/demo/tables/geographic-mobility/2015/county-to-county-migration-2011-2015/county-to-county-migration-flows/Net_Gross_US.txt) (~100Mb)
- `prepare_data_fields.csv`
- `prepare_data_states_num_counties.csv`

#### Execution order

Run the following files with `python <file.py>`

- `prepare_data.py`
- `compute_ultrametric_v2.py`
- `plot_dendrogram.py`
- `plot_VR.py` (not working yet)

## [Global migration](https://github.com/tmelorc/migration/tree/master/global)
