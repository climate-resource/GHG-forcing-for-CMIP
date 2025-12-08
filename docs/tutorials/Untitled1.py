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
import numpy as np
import pandas as pd
import pymc as pm

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# %%
df_ch4_eo = pd.read_csv("../../data/downloads/co2/co2_eo_raw.csv")
df_ch4 = pd.read_csv("../../data/downloads/co2/co2_gb_raw.csv")

# %%
df_gb_agg = (
    df_ch4.groupby(["year", "month"])
    .agg({"value":"mean"})
    .reset_index()
)
df_eo_agg = (
    df_ch4_eo.groupby(["year", "month"])
    .agg({"value":"mean"})
    .reset_index()
)
df_all = (df_gb_agg.merge(df_eo_agg, on=["year", "month"], 
                          how="outer", suffixes=("_gb", "_eo"))
)
df_all.head()

# %%
df_coll = (
    df_all[(~df_all.value_gb.isna()) & (~df_all.value_eo.isna())]
        .drop_duplicates()
        .reset_index(drop=True)
)
df_coll["day"] = 15
df_coll["date"] = pd.to_datetime(df_coll[["year", "month", "day"]])
df_coll = df_coll.drop(columns=["year", "month", "day"]).drop_duplicates().set_index("date")
df_coll

# %%
