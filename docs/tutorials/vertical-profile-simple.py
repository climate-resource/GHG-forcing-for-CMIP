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

# %% [markdown]
# # Correct for vertical dimension

# %%
import pmdarima as pma
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas
from geodatasets import get_path
import pymc as pm
import statsmodels

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate
from sklearn.metrics import mean_squared_error, r2_score

from ghg_forcing_for_cmip import CONFIG, plotting, utils
from itertools import product

# %%
from skforecast.recursive import ForecasterRecursive


# %% [markdown]
# ## EO and GB data
#
# Which data do I use? How do I preprocess the data? How to run the prefect flow to get final data?
#

# %%
df_ch4_eo = pd.read_csv("../../data/downloads/co2/co2_eo_raw.csv")
df_ch4 = pd.read_csv("../../data/downloads/co2/co2_gb_raw.csv")

# %%
agg_cols = ["year", "month", "lat_bnd", "lon_bnd", "bnd", "lat", "lon"]
d_full = df_ch4.merge(df_ch4_eo[agg_cols+["value"]], how="outer", on=agg_cols, suffixes=("_gb", "_eo"))
d_full = d_full[agg_cols + ["value_gb", "value_eo"]]
d_full = d_full.drop_duplicates().reset_index()
d_sel = d_full[(d_full.year==2014) & (d_full.month==2)]

# %%
# filling satellite data
d_gb = d_sel[~d_sel.value_gb.isna()].reset_index()

grid_lon = d_gb.lon.unique()
grid_lat = d_gb.lat.unique()
LON, LAT = np.meshgrid(grid_lon, grid_lat)

sat_grid1 = d_gb.pivot_table(
    index='lat', 
    columns='lon', 
    values='value_eo',
    aggfunc='mean' # Use mean just in case there are duplicate entries for one cell
).reset_index(drop=True)
sat_grid1 = np.array(sat_grid1)

valid_sat_rows, valid_sat_cols = np.where(~np.isnan(sat_grid1))
sat_train_z = sat_grid1[valid_sat_rows, valid_sat_cols]
sat_train_x = LON[valid_sat_rows, valid_sat_cols]
sat_train_y = LAT[valid_sat_rows, valid_sat_cols]

ok_sat = OrdinaryKriging(
    sat_train_x, sat_train_y, sat_train_z, 
    variogram_model='spherical'
)

sat_filled, _ = ok_sat.execute('grid', grid_lon, grid_lat)

# %%
ground_vals = d_gb.pivot_table(
    index='lat', 
    columns='lon', 
    values='value_gb',
    aggfunc='mean' # Use mean just in case there are duplicate entries for one cell
).reset_index(drop=True)
ground_vals = np.array(ground_vals)

g_rows, g_cols = np.where(~np.isnan(ground_vals))

# Training Arrays
train_ground = ground_vals[g_rows, g_cols]
train_x = LON[g_rows, g_cols]
train_y = LAT[g_rows, g_cols]

# Extract the drift (satellite) values for these specific points
# We pull from 'sat_filled' so we are guaranteed to have a value
train_drift_values = sat_filled[g_rows, g_cols]

# %%
uk = UniversalKriging(
    train_x, 
    train_y, 
    train_ground,
    variogram_model='spherical',
    drift_terms=['specified'],
    specified_drift=[train_drift_values] # The satellite values at ground stations
)

# Execute on the full grid
# We use the FULL filled satellite map as the drift
z_pred, z_var = uk.execute(
    'grid',  
    d_full.lon.unique(), 
    d_full.lat.unique(),
    specified_drift_arrays=[np.ones((len(d_full.lon.unique()), len(d_full.lat.unique())))] # The satellite map for the whole area
)

# %%
plt.figure(figsize=(12, 4))

#plt.subplot(1, 3, 1)
#plt.title("1. Raw Satellite (with Gaps)")
#plt.imshow(sat_observed, origin='lower')
#plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("2. Filled Satellite (Stage 1)")
plt.imshow(sat_filled, origin='lower')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("3. Final Ground Predictions")
plt.imshow(z_pred, origin='lower')
plt.colorbar()

plt.show()

# %%
ground_matrix_full = d_full.pivot_table(
    index='lat', 
    columns='lon', 
    values='value_eo',
    aggfunc='mean' # Use mean just in case there are duplicate entries for one cell
).reset_index(drop=True)
ground_matrix_full = np.array(ground_matrix_full)

# %%
ground_matrix = d_gb.pivot_table(
    index='lat', 
    columns='lon', 
    values='value_gb',
    aggfunc='mean' # Use mean just in case there are duplicate entries for one cell
).reset_index(drop=True)
ground_matrix = np.array(ground_matrix)

# %%
satellite_grid = d_sel.pivot_table(
    index="lat", 
    columns='lon', 
    values='value_eo',
    aggfunc='mean' # Use mean just in case there are duplicate entries for one cell
).reset_index(drop=True)
satellite_grid = np.array(satellite_grid)

# %%
#d_gb = np.array(ground_matrix[(ground_matrix.year == 2014) & (ground_matrix.month == 2)])
#d_eo = np.array(satellite_grid[(satellite_grid.year == 2014) & (satellite_grid.month == 2)])



# %%
rows, cols = np.where(~np.isnan(ground_matrix))
z_data = ground_matrix[rows,cols]
x_coords = LON[rows, cols]
y_coords = LAT[rows, cols]

drift_at_ground_locs = satellite_grid[rows, cols]

# %%
uk = UniversalKriging(
    x_coords, 
    y_coords, 
    z_data,
    variogram_model='spherical', # You can also try 'linear' or 'gaussian'
    drift_terms=['specified'],
    specified_drift=[drift_at_ground_locs]
)

z_pred, z_var = uk.execute(
    'grid', 
    d_full.lon.unique(), 
    d_full.lat.unique(),
    specified_drift_arrays=[np.zeros(ground_matrix_full.shape)] # The satellite map for the whole area
)

# %%
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Final Fused GHG Map")
plt.imshow(z_pred, origin='lower')
plt.colorbar(label="GHG Level")

plt.subplot(1, 2, 2)
plt.title("Uncertainty (Variance)")
plt.imshow(z_var, origin='lower')
plt.colorbar(label="Variance")
plt.show()

# %%
d_agg = df_ch4_eo.groupby(["year", "month", "lat", "pre"]).agg(
    {"vmr_profile_apriori": "mean"}
).reset_index()

ghg_surf = d_agg[d_agg.pre == 0.85].reset_index(drop=True)
ghg_surf.rename(columns={"vmr_profile_apriori":"surface_pressure"}, inplace=True)

d_full = df_ch4_eo.merge(ghg_surf.drop(columns="pre"), on=["year", "month", "lat"])
d_full["value_adj"] = d_full["value"] * (1/d_full["value"].divide(d_full["surface_pressure"]))

# %%
year = 2004
cols = ["year", "month", "lat", "lon"]
d_collocated = d_full[cols + ["value_adj", "value"]].merge(df_ch4[cols + ["value"]], on=cols, suffixes=("_eo", "_gb"))

combi = d_collocated[d_collocated.year == year].groupby(["lat", "lon"]).count()["month"].reset_index().sort_values("month", ascending=False).reset_index(drop=True)

# %%

# %%
_, axs = plt.subplots(2,5, constrained_layout=True, figsize=(10,4))
for i in range(5):
    lat = combi.lat.iloc[i]
    lon = combi.lon.iloc[i]

    d_plot = d_collocated[(d_collocated.lat==lat) & (d_collocated.lon==lon)].reset_index()
    d_plot["time"] = d_plot.year + d_plot.month/12
    sns.lineplot(d_plot, x="time", y="value_gb", label="GB", ax=axs[0,i])
    sns.lineplot(d_plot, x="time", y="value_eo",label="EO", ax=axs[0,i])
    sns.lineplot(d_plot, x="time", y="value_adj",label="EO-adj", ax=axs[0,i])
    axs[0,i].legend(frameon=False, fontsize="x-small")
    
for i,j in enumerate(range(5,10)):
    lat = combi.lat.iloc[j]
    lon = combi.lon.iloc[j]

    d_plot = d_collocated[(d_collocated.lat==lat) & (d_collocated.lon==lon)].reset_index()
    d_plot["time"] = d_plot.year + d_plot.month/12
    sns.lineplot(d_plot, x="time", y="value_gb", ax=axs[1,i])
    sns.lineplot(d_plot, x="time", y="value_eo", ax=axs[1,i])
    sns.lineplot(d_plot, x="time", y="value_adj", ax=axs[1,i])

# %% [markdown]
# Create a master dataframe indexed by $t$ (time), $x$ (longitude), and $y$ (latitude).

# %%
# template dataframe with complete information for every 
# year x month x lat x lon combination
df_template = pd.DataFrame(
    product(df_ch4.year.unique(), np.arange(1,13,1), 
            df_ch4.lat.unique(), df_ch4.lon.unique()),
    columns = ["year", "month", "lat", "lon"]
)

# %%
# combine ground-based and satellite data into one dataset
sel_cols = ["year", "month", "lat", "lon", "value"]

df = (df_ch4[sel_cols]
 .drop_duplicates()
 .merge(
     df_ch4_eo[sel_cols],
     on=["year", "month", "lat", "lon"],
     how="outer",
     suffixes=("_gb", "_eo")
 )
)

df = df.merge(df_template, on=["year", "month", "lat", "lon"], how="outer")

df['decimal_year'] = df['year'] + (df['month'] - 1) / 12
df

# %%
no_total = len(df)
no_missing_both = len(df[(df.value_gb.isna()) & (df.value_eo.isna())])
no_missing_gb = len(df[(df.value_gb.isna()) & (~df.value_eo.isna())])
no_missing_eo = len(df[(~df.value_gb.isna()) & (df.value_eo.isna())])

print("Both GB and EO are missing: ", 
      no_missing_both, f"({no_missing_both/no_total:.2f}%)")
print("Only GB is missing: ", 
      no_missing_gb, f"({no_missing_gb/no_total:.2f}%)")
print("Only EO is missing: ", 
      no_missing_eo, f"({no_missing_eo/no_total:.2f}%)")


# %% [markdown]
# ## Workflow idea
#
# 1. fill missing GB by predictions based on EO
#     + select collocated observations (i.e., rows for which both GB and EO products exist)
#     + learn function $f$ that predicts GB based on EO values
#     + validate prediction model using cross validation
#     + select observations for which only EO values exist and predict with $f$ corresponding GB values
#
# The first step is to learn the relationship between Surface and Column CO2 so you can convert your $D_2$ data into "Synthetic Surface Data". We need a transfer function $f$. A simple linear regression is insufficient because the relationship changes based on latitude (vegetation belts) and season (mixing height).Recommended Model: Generalized Additive Model (GAM)A GAM allows for non-linear relationships and interactions.

# %%
def evaluate(model, X, y, cv):
    cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
        )
    mae = -cv_results["test_neg_mean_absolute_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    r2 = cv_results["test_r2"]
    print(
        f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}\n"
        f"R2: {r2.mean():.3f} +/- {r2.std():.3f}"
    )


# %%
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
# collocated dataset
df_coll = df[(~df.value_gb.isna()) & (~df.value_eo.isna())]

# calibration model
features = sel_cols[:-1] + ["value_eo"]
y = df_coll.value_gb
X = df_coll[features]

# Split for validation (to check if our filling is accurate)
tscv = TimeSeriesSplit(
    n_splits=5, 
    gap=48,
    max_train_size=10000,
    test_size=1000
)

gbr = GradientBoostingRegressor(random_state=42)
dt_reg = DecisionTreeRegressor(random_state=42)
mod = AutoReg(endog=y, lags=3, exog=X)
res = mod.fit()
res
# Evaluate
#evaluate(mod, X, y, tscv)

# %%
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, MinMaxScaler
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-6, 6, 25)

def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )

cyclic_spline_transformer = ColumnTransformer(
    transformers=[
        ("cyclic_year", periodic_spline_transformer(1, n_splines=1), ["year"]),
        ("cyclic_month", periodic_spline_transformer(12, n_splines=6), ["month"]),
    ],
    remainder=MinMaxScaler(),
)

cyclic_spline_poly_pipeline = make_pipeline(
    cyclic_spline_transformer,
    Nystroem(kernel="poly", degree=2, n_components=300, random_state=0),
    RidgeCV(alphas=alphas),
)

evaluate(cyclic_spline_poly_pipeline, X, y, tscv)

# %%
# Identify rows where Surface (D1) is missing, but Column (D2) exists.
fill_mask = df.value_gb.isna() & df.value_eo.notna()

# Predict the surface values for these rows
X_fill = df.loc[fill_mask, features]
predicted_surface = rf.predict(X_fill)

# Create a new column 'D1_Combined'
df['value_gb_combined'] = df.value_gb.copy()
df.loc[fill_mask, 'value_gb_combined'] = predicted_surface

# %%
# Check improvement
print(f"\nMissing values before filling: {len(df[df['value_gb'].isna()])/len(df):.2f}")
print(f"Missing values after filling with satellite data: {len(df[df['value_gb_combined'].isna()])/len(df):.2f}")
print("(Remaining missing values are where BOTH datasets had gaps)")


# %%
def plot_coverage(target_value, gas, df_sel, ms, vmin, vmax, grid_size=5):
    gdf = geopandas.GeoDataFrame(
            df_sel,
            geometry=geopandas.points_from_xy(df_sel.lon, df_sel.lat),
            crs="EPSG:4326",
        )
    world = geopandas.read_file(get_path("naturalearth.land"))
    
    legend_kwds = {
        "shrink": 0.8,
        "label": "CH4",
        "orientation": "vertical",
    }
    
    fig, axs = plt.subplots(1, 1)
    world.plot(ax=axs, color="lightgrey", edgecolor="grey")
    
    gdf.plot(
        ax=axs,
        column=df_sel[target_value],
        marker="s",
       # edgecolor="black",
        markersize=ms,
        zorder=2,
        vmin=vmin,
        vmax=vmax,
        label="ground-based",
        legend=True,
        legend_kwds=legend_kwds,
    )

    for hl in np.arange(0, 90 + grid_size, grid_size):
        axs.axhline(float(hl), color="black", lw=0.5)
        axs.axhline(-float(hl), color="black", lw=0.5)
    for vl in np.arange(0, 180 + grid_size, grid_size):
        axs.axvline(float(vl), color="black", lw=0.5)
        axs.axvline(-float(vl), color="black", lw=0.5)
        
    axs.set_xlim(-10, 40)
    axs.set_ylim(30, 70)
    axs.set_xlabel("longitude")
    axs.set_ylabel("latitude")

    return fig, axs


# %%
df_sel = df[df.year==2004]
gas="ch4"

plot_coverage("value_gb", gas, df_sel, ms=20, vmin=None, vmax=None);

# %%
plot_coverage("value_gb_combined", gas, df_sel, ms=20, vmin=None, vmax=None);


# %%
def plot_hemisphere(df):
    conditions = [
        df["lat"] >= 45,                        
        (df["lat"] < 45) & (df["lat"] > -45),   
    ]
    choices = [
        "N",
        "Tropics",
    ]
    df.loc[df.index, "hemisphere"] = np.select(conditions, choices, default="S")
    
    df_agg = (df
        .groupby(["decimal_year", "hemisphere"])
        .agg({"value_gb": "mean", "value_gb_combined": "mean", "value_eo": "mean"})
        .reset_index()
    ) 
    
    fig, axs = plt.subplots(1,2, constrained_layout=True, figsize=(10,3), sharex=True, sharey=True)
    sns.lineplot(data=df_agg, x="decimal_year", y="value_gb", hue="hemisphere", alpha=0.2, legend=False, ax=axs[0])
    sns.lineplot(data=df_agg, x="decimal_year", y="value_gb_combined", hue="hemisphere", ax=axs[0])
    sns.lineplot(data=df_agg, x="decimal_year", y="value_eo", hue="hemisphere", ax=axs[1])

plot_hemisphere(df);

# %% [markdown]
# ## Gaussian Process model

# %%
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

long_term_trend_kernel = 50.0**2 * RBF(length_scale=50)

seasonal_kernel = (
    2.0**2
    * RBF(length_scale=100)
    * ExpSineSquared(length_scale=1., periodicity=1.0, periodicity_bounds="fixed")
)

irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)

noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
    noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
)

ch4_kernel = (
    long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
)
ch4_kernel

# %%
df_loc = (df[(~df.value_gb_combined.isna()) & (df.year > 2003)]
    .groupby(["decimal_year"])
    .agg({"value_gb_combined":"mean"})
    .reset_index())

X = df_loc.decimal_year.to_numpy()[:,None]
y = df_loc.value_gb_combined.to_numpy()
y_mean = df_loc.value_gb_combined.mean()

gaussian_process = GaussianProcessRegressor(kernel=ch4_kernel, normalize_y=False)
gaussian_process.fit(X, y - y_mean)


X_test = np.linspace(start=df_loc.decimal_year.min(), stop=2040, num=1_000).reshape(-1, 1)
mean_y_pred, std_y_pred = gaussian_process.predict(X_test, return_std=True)
mean_y_pred += y_mean

plt.plot(X, y, color="black", linestyle="dashed", label="Measurements")
plt.plot(X_test, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
plt.fill_between(
    X_test.ravel(),
    mean_y_pred - std_y_pred,
    mean_y_pred + std_y_pred,
    color="tab:blue",
    alpha=0.2,
)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Monthly average of CH$_4$ concentration (ppb)");

# %%
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

# create a copy for modeling to avoid warnings
model_df = df.copy()

# 1. Trend Features (Quadratic to capture accelerating CO2)
model_df['t'] = model_df['decimal_year'] - df.year.min()
model_df['t_sq'] = model_df['t'] ** 2

# 2. Seasonal Features (Harmonics)
# We use sin/cos to model the wave
model_df['sin_yr'] = np.sin(2 * np.pi * model_df['decimal_year'])
model_df['cos_yr'] = np.cos(2 * np.pi * model_df['decimal_year'])

# 3. Interaction Features (Crucial!)
# Seasonality amplitude depends on Latitude.
# (High amplitude at poles, low at equator).
# We multiply Lat by Sin/Cos to allow the model to learn this "fading" effect.
model_df['sin_x_lat'] = model_df['sin_yr'] * model_df['lat']
model_df['cos_x_lat'] = model_df['cos_yr'] * model_df['lat']

# %%
# We train on the "Combined" data from Step 2 (which has D1 + D2-inferred values)
train_mask = model_df['value_gb_combined'].notna()
train_data = model_df[train_mask]

# Predictors for the physical trend
trend_features = ['t', 't_sq', 'lat', 'lat_sq', 'sin_x_lat', 'cos_x_lat']
# Note: We include lat_sq because CO2 distribution isn't linear N-S
train_data = train_data.assign(lat_sq = train_data['lat']**2)

# Fit the Trend Model (Linear Regression is great for extrapolation)
trend_model = Ridge(alpha=1.0)
trend_model.fit(train_data[trend_features], train_data['value_gb_combined'])

# Apply this model to the WHOLE dataset (filling holes with the smooth trend)
model_df['lat_sq'] = model_df['lat']**2
model_df['global_trend'] = trend_model.predict(model_df[trend_features])

### (Gap Filling)
# The trend model is too smooth. Real data has local bumps.
# Let's calculate the "Residual" (Actual - Trend)
model_df['residual'] = model_df['value_gb_combined'] - model_df['global_trend']

# We use KNN (Spatial interpolation) to fill missing residuals
# effectively "smearing" local anomalies into empty neighboring cells.
valid_res_mask = model_df['residual'].notna()
X_res = model_df.loc[valid_res_mask, ['lat', 'lon', 'decimal_year']]
y_res = model_df.loc[valid_res_mask, 'residual']

# KNN Regressor looks for the 5 nearest neighbors in Space-Time
knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(X_res, y_res)

# Predict residuals for ALL rows (even those that were missing)
# Note: For very large datasets, you might only predict on missing rows to save time
filled_residuals = knn.predict(model_df[['lat', 'lon', 'decimal_year']])

# Final Reconstruction: Trend + Spatially Interpolated Residual
model_df['value_gb_final'] = model_df['global_trend'] + filled_residuals

# %%
months= np.arange(1,13,1)
lats = df.lat.unique()
lons = df.lon.unique()

# Create future grid (e.g., for 2022)
future_years = [2026, 2027] #np.arange(2022, 2032, 1)
future_grid = list(product(future_years, months, lats, lons))
df_future = pd.DataFrame(future_grid, columns=['year', 'month', 'lat', 'lon'])

# Re-create features for future
df_future['decimal_year'] = df_future['year'] + (df_future['month'] - 1) / 12
df_future['t'] = df_future['decimal_year'] - df.year.min()
df_future['t_sq'] = df_future['t'] ** 2
df_future['sin_yr'] = np.sin(2 * np.pi * df_future['decimal_year'])
df_future['cos_yr'] = np.cos(2 * np.pi * df_future['decimal_year'])
df_future['sin_x_lat'] = df_future['sin_yr'] * df_future['lat']
df_future['cos_x_lat'] = df_future['cos_yr'] * df_future['lat']
df_future['lat_sq'] = df_future['lat']**2

# Predict Trend
df_future['global_trend'] = trend_model.predict(df_future[trend_features])

# For future residuals, we usually assume 0 (mean reversion) 
# or we could decay the last known residual. For now, we assume 0.
df_future['value_gb_final'] = df_future['global_trend']

# %%

# %%
# Append future to history
final_dataset = pd.concat([model_df, df_future], ignore_index=True)

plt.figure(figsize=(12, 6))

# Plot for a specific location
final_dataset = final_dataset.groupby(["decimal_year"]).agg({"value_gb_final":"mean"}).reset_index()

subset = final_dataset.sort_values('decimal_year')

# Plot History
plt.plot(subset['decimal_year'], subset['value_gb_final'], 
         label='Final Model (Filled + Forecast)', color='red', linestyle='--')
plt.scatter(subset['decimal_year'], subset['value_gb_final'], 
            label='Available Data (Observed + Step 2)', color='black', s=10, alpha=0.5)

# Highlight Future
plt.axvline(future_years[0])
#future_subset = subset[subset['year'] >= future_years]
#plt.axvspan(future_years.min(), future_years.max(), color='yellow', alpha=0.2, label='Future Forecast')

plt.title('Final Result: Gap Filling & Forecasting (Lat: 60, Lon: 0)')
plt.ylabel('Surface CO2 (ppm)')
plt.xlabel('Year')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Misc

# %%
fetch_openml(name="mauna-loa-atmospheric-co2", as_frame=True)

# %%
from sklearn.datasets import fetch_openml
import polars as pl


co2 = fetch_openml(data_id=41187, as_frame=True)
co2_data = pl.DataFrame(co2.frame[["year", "month", "day", "co2"]]).select(
    pl.date("year", "month", "day"), "co2"
)
co2_data = (
    co2_data.sort(by="date")
    .group_by_dynamic("date", every="1mo")
    .agg(pl.col("co2").mean())
    .drop_nulls()
)
X = co2_data.select(
    pl.col("date").dt.year() + pl.col("date").dt.month() / 12
).to_numpy()
y = co2_data["co2"].to_numpy()
