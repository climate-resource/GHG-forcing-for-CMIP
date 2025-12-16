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
import verde as vd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import geopandas
from geodatasets import get_path
from sklearn.model_selection import TimeSeriesSplit
from ghg_forcing_for_cmip import CONFIG
from tqdm import tqdm

# %%
SEED = 1211
GAS = "ch4" # select gas
UNIT = "ppm"
DAY = 15 # assign day for date
MAX_LAT = 90
MAX_LON = 180
GRID_CELL_SIZE = CONFIG.GRID_CELL_SIZE

# %%
df_eo = pd.read_csv(f"../../data/downloads/{GAS}/{GAS}_eo_raw.csv")
df_gb = pd.read_csv(f"../../data/downloads/{GAS}/{GAS}_gb_raw.csv")

# %%
# target variables (used for grouping, selection, etc.)
sel_cols = ["year", "month", "lat", "lon"]

# template dataframe with complete information for every 
# year x month x lat x lon combination
df_template = pd.DataFrame(
    product(df_gb.year.unique(), np.arange(1,13,1), 
            np.arange(-(MAX_LAT-(GRID_CELL_SIZE/2)), MAX_LAT, GRID_CELL_SIZE), 
            np.arange(-(MAX_LON-(GRID_CELL_SIZE/2)), MAX_LON, GRID_CELL_SIZE)
           ),
    columns = sel_cols
)

# combine ground-based and satellite data into one dataset
df_combined = (df_gb[sel_cols+["value"]]
 .drop_duplicates()
 .merge(
     df_eo[sel_cols+["value"]],
     on=sel_cols,
     how="outer",
     suffixes=("_gb", "_eo")
 )
)

# combine dataset with template data to get also all combinations
# where both, GB and EO are NaN
df_total = df_combined.merge(df_template, on=sel_cols, how="outer")
df_total["date"] = pd.to_datetime(df_total[['year', 'month']].assign(day=DAY))
df_total = df_total.set_index("date")


# %%
def prepare_dataset(df_combined, condition):
    if condition == "collocated":
        df = df_combined[(~df_combined.value_gb.isna()) & (~df_combined.value_eo.isna())]
    elif condition == "eo-only":
        df = df_combined[(df_combined.value_gb.isna()) & (~df_combined.value_eo.isna())]
    elif condition == "gb-only":
        df = df_combined[(~df_combined.value_gb.isna()) & (df_combined.value_eo.isna())]
    elif condition == "both-none":
        df = df_combined[(df_combined.value_gb.isna()) & (df_combined.value_eo.isna())]
    else:
        raise ValueError(condition, "does not exist")
        
    df_clean = df.drop_duplicates().reset_index(drop=True)

    df_clean["date"] = pd.to_datetime(df_clean[['year', 'month']].assign(day=DAY))

    df_clean['season_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12)
    df_clean['season_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12)

    return df_clean


# %%
features = ['year', 'lat', 'lon', 'value_eo', 'season_sin', 'season_cos']
target = "value_gb"

# get dataset and sort it according to time
df_collocated = prepare_dataset(df_combined, condition="collocated")
df_collocated = df_collocated.sort_values(by=['year', 'month'])

# Separate features and target variable
X = df_collocated[features]
y = df_collocated[target]

# prepare split in train and test data
tscv = TimeSeriesSplit(n_splits=5)

# the model
model_reg = xgb.XGBRegressor(
    n_estimators=1000,       # Number of trees
    learning_rate=0.05,      # Step size
    max_depth=6,             # Depth of trees (catches interactions like Lat vs Season)
    early_stopping_rounds=50,
    n_jobs=-1
)

score = []
for train_index, test_index in tscv.split(X):
        
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    score.append(model_reg.score(X_test, y_test))

print(f"Test Score: {np.mean(score):.2f} (SD: {np.std(score):.2f})")

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=SEED)

df_test = df_collocated.iloc[X_test.index].copy()
model_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
df_test["value_gb_pred"] = model_reg.predict(X_test)


# %% jupyter={"source_hidden": true}
def plot_predictions_by_component(df, lat=True, lon=True):
    df_year = df.groupby(["year"]).agg({"value_gb":"mean", "value_gb_pred":"mean"}).reset_index()
    df_month = df.groupby(["month"]).agg({"value_gb":"mean", "value_gb_pred":"mean"}).reset_index()
    
    fig, axs = plt.subplots(2,2, constrained_layout=True, figsize=(8,4))
    sns.lineplot(data=df_year, x="year", y="value_gb", label="observed", ax=axs[0,0])
    sns.lineplot(data=df_year, x="year", y="value_gb_pred", label="predicted", ax=axs[0,0], linestyle="--")
    axs[0,0].legend(frameon=False)
    
    sns.lineplot(data=df_month, x="month", y="value_gb", ax=axs[0,1])
    sns.lineplot(data=df_month, x="month", y="value_gb_pred", ax=axs[0,1], linestyle="--")

    if lat:
        df_lat = df.groupby(["lat"]).agg({"value_gb":"mean", "value_gb_pred":"mean"}).reset_index()
        sns.lineplot(data=df_lat, x="lat", y="value_gb", ax=axs[1,0])
        sns.lineplot(data=df_lat, x="lat", y="value_gb_pred", ax=axs[1,0], linestyle="--")
        
    if lon:
        df_lon = df.groupby(["lon"]).agg({"value_gb":"mean", "value_gb_pred":"mean"}).reset_index()
        sns.lineplot(data=df_lon, x="lon", y="value_gb", ax=axs[1,1])
        sns.lineplot(data=df_lon, x="lon", y="value_gb_pred", ax=axs[1,1], linestyle="--")

    return fig, axs


# %%
plot_predictions_by_component(df_test, lat=True, lon=True);


# %% jupyter={"source_hidden": true}
def plot_predictions_by_hemisphere(df_collocated, gas, unit, eo=False):
    conditions = [
        (df_collocated['lat'] > 45),
        (df_collocated['lat'] < -45)
    ]
    choices = ['N', 'S']
    df_collocated['hemisphere'] = np.select(conditions, choices, default='Tropics')
    
    df_coll_agg = (
        df_collocated
            .reset_index(drop=True)
            .groupby(["date", "hemisphere"])
            .agg({"value_gb": "mean", "value_eo": "mean", "value_gb_pred": "mean"})
            .reset_index()
    )
    
    fig, axs = plt.subplots(3,1, figsize=(7,7), constrained_layout=True, sharex=True)
    for i, hemi in enumerate(df_coll_agg.hemisphere.unique()):
        if i == 0:
            label_gb = "GB - observed"
            label_pred = "GB - predicted"
            label_eo = "EO (for reference)"
        else:
            label_gb = None
            label_pred = None
            label_eo = None
            
        sns.lineplot(data=df_coll_agg[df_coll_agg.hemisphere == hemi], x="date", y="value_gb", 
                     lw=2, ax=axs[i], label=label_gb, alpha=0.3, color="grey")
        sns.scatterplot(data=df_coll_agg[df_coll_agg.hemisphere == hemi], x="date", y="value_gb", 
                     ax=axs[i], color="black", s=7)
        sns.lineplot(data=df_coll_agg[df_coll_agg.hemisphere == hemi], x="date", y="value_gb_pred", 
                     lw=2, ax=axs[i], alpha=0.6, label=label_pred)
        if eo:
            sns.lineplot(data=df_coll_agg[df_coll_agg.hemisphere == hemi], x="date", y="value_eo", 
                     lw=2, ax=axs[i], alpha=0.6, label=label_eo)
        axs[i].set_title(f"hemisphere: {hemi}")
        axs[i].set_ylabel(f"{gas} in {unit}")
    axs[2].set_xlabel("year/month")
    axs[0].legend(frameon=False)

    return fig, axs


# %%
plot_predictions_by_hemisphere(df_test, GAS, UNIT, eo=False);

# %%
df_collocated["obs_gb"] = True

df_gb_only = prepare_dataset(df_combined, condition="gb-only")
df_gb_only["obs_gb"] = True

df_eo_only = prepare_dataset(df_combined, condition="eo-only")
df_eo_only["value_gb"] = model_reg.predict(df_eo_only[features])
df_eo_only["obs_gb"] = False

df_gb_pred = pd.concat([df_collocated, df_eo_only, df_gb_only]).reset_index(drop=True)


# %% jupyter={"source_hidden": true}
def plot_coverage(df: pd.DataFrame, year, grid_size: int, gas, unit, subset=False, ms=10, lw=0.5, vmin=None, vmax=None):

    # compute grid-cell average
    df_gb_binned = df[df.year == year].groupby(["lat", "lon"]).agg({"value_gb":"mean", "obs_gb":"max"}).reset_index()

    gdf_gb = geopandas.GeoDataFrame(
        df_gb_binned,
        geometry=geopandas.points_from_xy(df_gb_binned.lon, df_gb_binned.lat),
        crs="EPSG:4326",
    )

    world = geopandas.read_file(get_path("naturalearth.land"))

    legend_kwds = {
        "shrink": 0.8,
        "label": f"{gas.upper()} [{unit}]",
        "orientation": "vertical",
    }

    fig, axs = plt.subplots(1, 1, figsize=(12,4))
    world.plot(ax=axs, color="lightgrey", edgecolor=None)

    gdf_gb.plot(
        ax=axs,
        column=df_gb_binned.value_gb,
        marker="s",
        #edgecolor="black",
        markersize=ms,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
        legend=True,
        legend_kwds=legend_kwds,
    )
    gdf_highlight = gdf_gb[gdf_gb["obs_gb"] == True]
    if not gdf_highlight.empty:
            gdf_highlight.plot(
                ax=axs,
                facecolor="none",    # Transparent center
                edgecolor="red",     # Red border
                linewidth=lw,       # Thickness of the border
                marker="s",          # Same shape as above
                markersize=ms,       # Same size as above
                zorder=3             # Draw on top of the base layer
            )
    
    for hl in np.arange(0, 90 + grid_size, grid_size):
        axs.axhline(float(hl), color="lightgrey", lw=0.5)
        axs.axhline(-float(hl), color="lightgrey", lw=0.5)
    for vl in np.arange(0, 180 + grid_size, grid_size):
        axs.axvline(float(vl), color="lightgrey", lw=0.5)
        axs.axvline(-float(vl), color="lightgrey", lw=0.5)

    if subset:
        axs.set_xlim(-10, 40)
        axs.set_ylim(30, 70)

    axs.set_title(f"{grid_size} x {grid_size} grid size (red squares: observed GB)")
    axs.set_xlabel("longitude")
    axs.set_ylabel("latitude")

    return fig, axs


# %%
plot_coverage(df_gb_pred, year=2022, grid_size=GRID_CELL_SIZE, gas=GAS, unit=UNIT, lw=1);

# %%
df_gb_pred = df_gb_pred[(df_gb_pred.year >= df_eo.year.unique().min()) & (df_gb_pred.year <= df_eo.year.unique().max())]
df_missing = prepare_dataset(df_total, condition="both-none")
df_missing = df_missing[(df_missing.year >= df_eo.year.unique().min()) & (df_missing.year <= df_eo.year.unique().max())]

value_gb_predicted = []
for year_month in tqdm(df_missing[["year", "month"]].drop_duplicates().values):
    df_pred = df_gb_pred[(df_gb_pred.year == year_month[0]) & (df_gb_pred.month==year_month[1])]
    df_gaps = df_missing[(df_missing.year == year_month[0]) & (df_missing.month==year_month[1])]
    
    # 'damping' smooths the data (regularization). 
    spline = vd.Spline(damping=1e-3)
    spline.fit((df_pred.lon, df_pred.lat), df_pred.value_gb)
    value_gb_predicted.append(spline.predict(coordinates=(df_gaps.lon, df_gaps.lat)))

df_missing["value_gb"] = np.concat(value_gb_predicted)

# %%
df_entire = pd.concat([df_missing, df_gb_pred]).reset_index(drop=True)

# %%
plot_coverage(df_gb_pred, year=2022, grid_size=CONFIG.GRID_CELL_SIZE, gas=GAS, unit=UNIT, lw=1, vmin=1730);

# %%
plot_coverage(df_entire, year=2022, grid_size=CONFIG.GRID_CELL_SIZE, gas=GAS, unit=UNIT, lw=1, vmin=1730);

# %%
conditions = [
    (df_entire['lat'] > 45),
    (df_entire['lat'] < -45)
]
choices = ['N', 'S']
df_entire['hemisphere'] = np.select(conditions, choices, default='Tropics')

df_agg = df_entire.groupby(["date", "hemisphere"]).agg({"value_gb":"mean"}).reset_index()

fig, axs = plt.subplots(1,1)
sns.lineplot(data=df_agg, x="date", y="value_gb", hue="hemisphere", ax=axs)
axs.set_ylabel(GAS.upper())
axs.spines[["right", "top"]].set_visible(False)
axs.legend(frameon=False, handlelength=0.5, ncol=3);

# %%
