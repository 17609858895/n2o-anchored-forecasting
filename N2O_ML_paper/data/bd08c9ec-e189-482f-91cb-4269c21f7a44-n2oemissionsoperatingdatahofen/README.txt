
This is the readme-file for the raw data supplied alongside the article:

Andreas Froemelt, Leon Zueger, Luzia von Kaenel, Daniel Braun, Wenzel Gruber: "Pattern recognition of operational states leading to N2O-emissions in full-scale biological wastewater treatment", Water Research X, 2025
https://doi.org/10.1016/j.wroa.2025.100336


The dataset encompasses operating data and N2O emissions of lane 2 of ARA Hofen (ARA = Water Resource Recovery Facility) for three full years (January 2020 to December 2022).

4 files are supplied in two folders:

- folder meta_data:

    - qgl_hof_meta.csv--> this is a meta-file explaining the columns/abbreviations in the raw-data-file including units

    - columns_translator_lane21.json --> helper-file to translate the columns to the names used in the article (this json is for lane 2, street 1)

    - columns_translator_lane22.json --> helper-file to translate the columns to the names used in the article (this json is for lane 2, street 2)

- folder raw_data:

    - qgl_hof_raw.csv--> this is the raw-data before removing outliers and imputing data

        