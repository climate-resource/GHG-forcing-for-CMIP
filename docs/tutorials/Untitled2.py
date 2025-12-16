# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
from itertools import product
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

# %%
gas = "ch4"
df_ch4_eo = pd.read_csv(f"../../data/downloads/{gas}/{gas}_eo_raw.csv")
df_ch4 = pd.read_csv(f"../../data/downloads/{gas}/{gas}_gb_raw.csv")

# %%
# target variables (used for grouping, selection, etc.)
sel_cols = ["year", "month", "lat", "lon"]

# template dataframe with complete information for every 
# year x month x lat x lon combination
df_template = pd.DataFrame(
    product(df_ch4.year.unique(), np.arange(1,13,1), 
            df_ch4.lat.unique(), df_ch4.lon.unique()),
    columns = sel_cols
)

# combine ground-based and satellite data into one dataset
df_combined = (df_ch4[sel_cols+["value"]]
 .drop_duplicates()
 .merge(
     df_ch4_eo[sel_cols+["value"]],
     on=sel_cols,
     how="outer",
     suffixes=("_gb", "_eo")
 )
)

# combine dataset with template data to get also all combinations
# where both, GB and EO are NaN
df_total = df_combined.merge(df_template, on=sel_cols, how="outer")

df_total["day"] = 1
df_total["date"] = pd.to_datetime(df_total[["year", "month", "day"]])
df_total = df_total.drop(columns=["year", "month", "day"]).set_index("date")
df_total.head()

# %%
# collocated dataset (GB != NaN & EO != NaN)
df_collocated = (
    df_combined[(~df_combined.value_gb.isna()) & (~df_combined.value_eo.isna())]
        .drop_duplicates()
        .reset_index(drop=True)
)

# add variable indicating hemispheric level (Northern, Southern, Tropics)
conditions = [
    (df_collocated['lat'] > 45),
    (df_collocated['lat'] < -45)
]
choices = ['N', 'S']
df_collocated['hemisphere'] = np.select(conditions, choices, default='Tropics')

# add indicator variable dividing latitudinal direction into
# Northern and Southern hemisphere (no tropics)
df_collocated["X_N"] = np.where(df_collocated["lat"] > 0, 1, 0)
df_collocated["time"] = df_collocated["year"] + (df_collocated["month"] - 1) / 12.0
df_collocated.head()

# %%
ARIMA(df_collocated.value_gb, df_collocated[["time", "value_eo"]], order=(1,1,1))
