## Pipeline modules

- download_datasets
    - download_obs4mips
    - download_cmip
- load_datasets
    - load_cmip
    - load_obs4mips
- preprocess_datasets
    - preprocess_cmip
    - preprocess_obs4mips
- model_cmip_data
    - bin_cmip
    - interpolate_cmip
    - add_vertical_to_cmip
- combine_datasets
    - ak_to_cmip
- clean_up

## Flow for downloading CMIP data

+ finalize flow for downloading all data sets
+ create optional argument for deleting zip files after flow ended
+ combine data sets
+ add additional information about site code in AGAGE and GAGE datasets
