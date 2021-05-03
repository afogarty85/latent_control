from h3 import h3
import geopandas as gpd
import geopandas.tools
from shapely import geometry, ops
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
import statsmodels.api as sm
from numba import njit
import matplotlib.colors as mcolors
from bokeh.io import show
from bokeh.plotting import figure
import bokeh.models as bm
from bokeh.models import LinearColorMapper, FixedTicker
from bokeh.palettes import RdBu


def load_and_prepare_districts(filepath):
    """Loads a geojson files of polygon geometries and features,
    swaps the latitude and longitude and stores geojson.
    The geoJSON must be exported such that it is a polygon,
    not a multipolygon. In QGIS, vector > geometry tools > multi to single parts
    geom_json: (lon, lat)
    geometry: (lon, lat)
    geom_swap: (lat, lon)
    geom_swap_geojson: (lat, lon)
    """
    gdf_districts = gpd.read_file(filepath, driver="GeoJSON")

    gdf_districts["geom_geojson"] = gdf_districts["geometry"].apply(
                                              lambda x: geometry.mapping(x))

    gdf_districts["geom_swap"] = gdf_districts["geometry"].map(
                                              lambda polygon: ops.transform(
                                                  lambda x, y: (y, x), polygon))

    gdf_districts["geom_swap_geojson"] = gdf_districts["geom_swap"].apply(
                                              lambda x: geometry.mapping(x))

    return gdf_districts

# load polygon geojson
input_file_districts = "C:\\Users\\Andrew\Desktop\\af_shape\\af_whole_json.geojson"
# create geopandas
gdf_districts = load_and_prepare_districts(filepath=input_file_districts)
# reproject to WGS84 Latitude/Longitude
gdf_districts.crs = "EPSG:4326"


def fill_hexagons(geom_geojson, res):
    """Fills a geometry given in geojson format with H3 hexagons at specified
    resolution. The flag_reverse_geojson allows to specify whether the geometry
    is lon/lat or swapped"""

    set_hexagons = h3.polyfill(geojson=geom_geojson,
                               res=res,
                               geo_json_conformant=False)
    return set_hexagons

# create h3 hexs to fill the shapefile
# resolution 6 = 36.1 km2 area
# resolution 7 = 5.16 km2 area
# resolution 8 = 0.73 km2 area
gdf_districts["h3_hexs"] = gdf_districts["geom_swap_geojson"].apply(lambda x: list(fill_hexagons(geom_geojson=x, res=6)))

# remove areas with empty hexs
gdf_districts = gdf_districts[gdf_districts['h3_hexs'].str.len() > 0].reset_index(drop=True)

# add matching variable for district name
gdf_districts['level_0'] = list(range(0, gdf_districts.shape[0]))

# create subset for matching
matching_districts = gdf_districts[['level_0', 'DIST_34_NA', 'PROV_34_NA']]

# explode the list of hexs
df = pd.DataFrame(gdf_districts["h3_hexs"])
unnested_lst = []
for col in df.columns:
    unnested_lst.append(df[col].apply(pd.Series).stack())
df = pd.concat(unnested_lst, axis=1, keys=df.columns)
df = df.reset_index()
df = df.drop('level_1', axis=1)

# add in district/prov names
df = pd.merge(df, matching_districts,  how='left', left_on=['level_0'], right_on=['level_0'])

# add lat & lng of center of hex
df['centroid_lat'] = df['h3_hexs'].apply(lambda x: h3.h3_to_geo(x)[0])
df['centroid_long'] = df['h3_hexs'].apply(lambda x: h3.h3_to_geo(x)[1])

# turn h3 hexs into geo. boundary
df['geometry'] = df["h3_hexs"].apply(lambda x: h3.h3_to_geo_boundary(h=x, geo_json=True))
# turn to Point
df['geometry'] = df['geometry'].apply(lambda x: [Point(x, y) for [x, y] in x])
# turn to Polygon
df['geometry'] = df['geometry'].apply(lambda x: Polygon([[poly.x, poly.y] for poly in x]))

# turn to geoDF
df_geo = gpd.GeoDataFrame(df, geometry="geometry")

# plot to see
#df_geo.plot()

# turn to time series; repeat each geo row, 1 for each month
df_geo = df_geo.loc[df_geo.index.repeat(12)]
# create a timeindex, 1 for each month
df_geo['time_index'] = np.tile(list(range(1,13)), len(df_geo) // len(list(range(1,13))))

# size of df
df_geo.shape

# reset index
df_geo = df_geo.reset_index(drop=True)

###########################
####### EVENT DATA -- Observable Emissions
###########################

# load event data
af = pd.read_csv('C:\\Users\\Andrew\\Desktop\\2019_af.csv', encoding='Latin-1')

# fill na
af = af.fillna(0)
af['total_cas'] = af['KIA Report Host Nation Security|Military'] + af['WIA Report Host Nation Security|Military']
af['total_cas'].describe()
threshold = af['total_cas'].quantile(q=0.90)
af['intense'] = af['total_cas'].apply(lambda x: 1 if x >= threshold else 0)
af['intense'].value_counts()

# strip white spaces
af['Month'] = af['Month'].str.strip()

# uppercase first letter
af['Month'] = af['Month'].str.capitalize()

# check months
af['Month'].unique()

# check year
af['Year'].unique()

# strip typo from year
#af['Year'] = af['Year'].str.strip('`')

# create mapping for month
d = dict((v,k) for k,v in zip(range(1, 13), af.Month.unique()))

# overwrite month
af['Month'] = af['Month'].map(d)

# create datetime; just year month
af['dt'] = pd.to_datetime(af[['Year', 'Month', 'Day']]).dt.to_period('M')

# create time series index for month
d = dict((v,k) for k,v in zip(range(1, 13), af.dt.unique()))

# create month index for time series
af['month_index'] = af['dt'].map(d)

# create separate dfs for types of ops
events_intense = af.loc[af['intense'] == 1].reset_index(drop=True)
events_not_intense = af.loc[af['intense'] == 0].reset_index(drop=True)

# spatial distance
def coords_to_vectors(df1, df2):
    ''' for broadcasting, df1 should be the larger df '''
    lon1 = np.array(df1['centroid_long'].tolist()).astype(np.float32)
    lat1 = np.array(df1['centroid_lat'].tolist()).astype(np.float32)
    lon2 = np.array(df2['Longitude'].tolist()).astype(np.float32)
    lat2 = np.array(df2['Latitude'].tolist()).astype(np.float32)
    return lon1, lat1, lon2, lat2

def vect_haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1[:, None]
    dlat = lat2 - lat1[:, None]
    a = np.sin(dlat/2.0)**2 + np.cos(lat1[:, None]) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

# create a vector of spatial distances from each event to each centroid
intense_distance_results = vect_haversine(*coords_to_vectors(df_geo, events_intense))
intense_distance_results.dtype

# create a vector of spatial distances from each event to each centroid
# takes several minutes and a lot of memory
not_intense_distance_results = vect_haversine(*coords_to_vectors(df_geo, events_not_intense))

# need to do the same with time now
def time_to_vectors(df1, df2):
    d1 = np.array(df1['time_index'].tolist()).astype(np.float32)
    d2 = np.array(df2['month_index'].tolist()).astype(np.float32)
    return d1, d2

def vec_time(data1, data2):
    age_centroid = data1
    age_event = data2
    age = np.where(age_event-1 < age_centroid[:, None], age_centroid[:, None] - age_event, np.nan)
    return age

# create a vector of age time from each event to each centroid
intense_time_results = vec_time(*time_to_vectors(df_geo, events_intense))

# create a vector of age time from each event to each centroid
not_intense_time_results = vec_time(*time_to_vectors(df_geo, events_not_intense))

################################
### logistic decay function
###############################

# spatial decay
def logistic_spatial_decay(vec, k=7, gamma=-0.35):
    ''' logistic spatial decay function, params:
    k: slope (float)
    gamma: inflection point (float)
    numerical overflows will occur with large distances,
    returning zero; likely bad coordinates '''
    w = (1 / (1 + np.exp(-(k + gamma*vec))))
    return w


# view function
distance = np.arange(1, 41)
weights_dist = logistic_spatial_decay(distance)
plt.plot(distance, weights_dist)
plt.xlabel('distance in km')
plt.ylabel('weight')

# apply function
intense_spatial_decay = logistic_spatial_decay(intense_distance_results)
not_intense_spatial_decay = logistic_spatial_decay(not_intense_distance_results)

intense_spatial_decay2 = logistic_spatial_decay(intense_distance_results)
not_intense_spatial_decay2 = logistic_spatial_decay(not_intense_distance_results)

# temporal decay
def logistic_time_decay(vec, k=8, gamma=-2.5):
    ''' logistic temporal decay function, params:
    k: slope (float)
    gamma: inflection point (float)
    '''
    w = (1 / (1 + np.exp(-(k + gamma*vec))))
    return w


# view function
time = np.linspace(1, 12)  # continuous look
weights_time = logistic_time_decay(time)
plt.plot(time, weights_time)
plt.xlabel('months')
plt.ylabel('weight')

# apply function
intense_time_decay = logistic_time_decay(intense_time_results)
not_intense_time_decay = logistic_time_decay(not_intense_time_results)

# truncate small values to zero
intense_spatial_decay = np.where(intense_spatial_decay <= 0.05, 0, intense_spatial_decay)
not_intense_spatial_decay = np.where(not_intense_spatial_decay <= 0.05, 0, not_intense_spatial_decay)

intense_time_decay = np.where(intense_time_decay <= 0.05, 0, intense_time_decay)
not_intense_time_decay = np.where(not_intense_time_decay <= 0.05, 0, not_intense_time_decay)

################################
### Exposure - weighted sum of time/space
###############################
exposure_intense = np.nanprod(np.dstack((intense_spatial_decay, intense_time_decay)), 2)
exposure_intense = np.sum(exposure_intense, axis=1)
exposure_intense.shape


exposure_not_intense = np.nanprod(np.dstack((not_intense_spatial_decay, not_intense_time_decay)), 2)
exposure_not_intense = np.sum(exposure_not_intense, axis=1)
exposure_not_intense.shape

################################
### Place them into DF
###############################
df = pd.DataFrame({'exposure_intense': exposure_intense, 'exposure_not_intense': exposure_not_intense})
df['time_index'] = np.tile(list(range(1, 13)), len(df) // len(list(range(1, 13))))

df_agg = df.groupby('time_index').agg(intense_mean=('exposure_intense', 'mean'),
                                      intense_var=('exposure_intense', 'var'),
                                      not_intense_mean=('exposure_not_intense', 'mean'),
                                      not_intense_var=('exposure_not_intense', 'var'))
df_agg = df_agg.reset_index()

# join df_agg to df
df = df.merge(df_agg, how='left', on=['time_index'])

# drop time_inde
df = df.drop(['time_index'], axis=1)

# join grid hex data to df
df = pd.concat([df, df_geo], axis=1)


# cdf function
def negbin_cdf(series):
    '''
    This function takes a np.array and returns a negative
    binomial CDF for overdispersed count data.
    # prob. that x is less than or equal to val.
    '''
    series = series.tolist()
    y = np.array([series])
    y = y.flatten()
    # create intercept to fit a model with intercept
    intercept = np.ones(len(y))
    # fit negative binomial
    m1 = sm.NegativeBinomial(y, intercept, loglike_method='nb2').fit()
    # retrieve mu
    mu = np.exp(m1.params[0])
    # retrieve alpha
    alpha = m1.params[1]
    # set Q to zero for nb2 method, Q to 1 for nb1 method
    Q = 0
    # derive size
    size = 1. / alpha * mu ** Q
    # derive prob
    prob = size / (size + mu)
    return nbinom.cdf(y, n=size, p=prob)

# apply fun.
df['exposure_intense'] = negbin_cdf(df['exposure_intense'])
df['exposure_not_intense'] = negbin_cdf(df['exposure_not_intense'])


# emission thresholds
mar = 0.025
ixs = 0.1


def preprocess_series(s):
    ''' this function generates categorical emissions '''
    t = None
    c = None
    if s['exposure_intense'] <= ixs:
        t = 0
    if s['exposure_not_intense'] <= ixs:
        c = 0
    if (t == 0) & (c == 0):
        return 0  # O1; zones of rebel or gov total control
    elif ((s['exposure_intense'] > s['exposure_not_intense'])) and (abs(s['exposure_intense'] - s['exposure_not_intense'])) > mar:
        return 1  # O2 - closer to rebel control, more aggressive
    elif ((abs(s['exposure_intense'] - s['exposure_not_intense']) <= mar)):
        return 2  # O3 - highly disputed
    elif (s['exposure_intense'] < s['exposure_not_intense']) and (abs(s['exposure_intense'] - s['exposure_not_intense']) > mar):
        return 3  # O4 - closer to gov control, less aggressive


################################
### HMM
###############################
# Emission Matrix
s1 = np.array([0.6, 0.175, 0.175, 0.05])  # rows sum to 1
s2 = np.array([0.05, 0.6, 0.175, 0.175])
s3 = np.array([0.05, 0.175, 0.6, 0.175])
s4 = np.array([0.05, 0.175, 0.175, 0.6])
s5 = np.array([0.6, 0.05, 0.175, 0.175])
emissions_matrix = np.vstack([s1, s2, s3, s4, s5])

# Initialize Markov Chain
# initial state probability vector
# probs of starting the sequence at a given state
# no prior of who controls what territory.
p_init = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Transition matrix
# probability of transitioning from one state to another
# rows: state at time t
# cols: state at time t+1
# row probs sum to 1
# off-diagonal mass: transition out of states
# diagonal mass: stay in state entered
t1 = np.array([0.25, 0.5, 0.025, 0.2, 0.025])
t2 = np.array([0.25, 0.15, 0.075, 0.5, 0.025])
t3 = np.array([0.05, 0.025, 0.050, 0.850, 0.025])
t4 = np.array([0.025, 0.075, 0.15, 0.125, 0.625])
t5 = np.array([0.05, 0.075, 0.475, 0.025, 0.375])
transition_matrix = np.vstack([t1, t2, t3, t4, t5])
assert transition_matrix[2, :].sum() == 1

# States: 5 possible states
states = np.array([0, 1, 2, 3, 4])


# viterbi decoding
@njit
def viterbi(transProbs, initProbs, emissionProbs, states, observations):
    '''
    port for R library (HMM) viterbi decoding
    yields most likely sequence
    ideal if we care about getting the right sequence
    '''
    assert np.isnan(transProbs).flatten().any() == False, 'nan exist'
    assert np.isnan(emissionProbs).flatten().any() == False, 'nan exist'
    nObservations = len(observations)
    nStates = len(states)
    v = np.full(shape=(nStates, nObservations), fill_value=np.nan)
    for state in states:
        v[state, 0] = np.log(initProbs[state] * emissionProbs[state, observations[0]])
        # iteration
        for k in range(1, nObservations):
            for state in states:
                maxi = -np.inf
                for previousState in states:
                    temp = v[previousState, k-1] + np.log(transProbs[previousState, state])
                    maxi = max(maxi, temp)
                v[state, k] = np.log(emissionProbs[state, observations[k]]) + maxi
        viterbiPath = np.repeat(np.nan, nObservations)
        for state in states:
            if max(v[:, nObservations-1]) == v[state, nObservations-1]:
                viterbiPath[nObservations-1] = state
                break
        for k in range(nObservations-2, -1, -1):
            for state in states:
                if (max(v[:, k] + np.log(transProbs[:, int(viterbiPath[k+1])]))) == v[state, k] + np.log(transProbs[state, int(viterbiPath[k+1])]):
                    viterbiPath[k] = state
                    break
    return viterbiPath


def forward_hmm(transProbs, initProbs, emissionProbs, states, observations):
    '''
    port for R library (HMM) forward decoding
    '''
    assert np.isnan(transProbs).flatten().any() == False, 'nan exist'
    assert np.isnan(emissionProbs).flatten().any() == False, 'nan exist'
    nObservations = len(observations)
    nStates = len(states)
    f = np.full(shape=(nStates, nObservations), fill_value=np.nan)
    for state in states:
        f[state, 0] = np.log(initProbs[state] * emissionProbs[state, observations[0]])
    for k in range(1, nObservations):
        for state in states:
            logsum = -np.inf
            for previousState in states:
                temp = f[previousState, k-1] + np.log(transProbs[previousState, state])
                if temp > -np.inf:
                    logsum = temp + np.log(1 + np.exp(logsum - temp))
            f[state, k] = np.log(emissionProbs[state, observations[k]]) + logsum
    return f

def backward_hmm(transProbs, initProbs, emissionProbs, states, observations):
    '''
    port for R library (HMM) backward function
    '''
    assert np.isnan(transProbs).flatten().any() == False, 'nan exist'
    assert np.isnan(emissionProbs).flatten().any() == False, 'nan exist'
    nObservations = len(observations)
    nStates = len(states)
    b = np.full(shape=(nStates, nObservations), fill_value=np.nan)
    for state in states:
        b[state, nObservations-1] = np.log(1)
    for k in range(nObservations-2, -1, -1):
        for state in states:
            logsum = -np.inf
            for nextState in states:
                temp = b[nextState, k+1] + np.log(transProbs[state, nextState]
                                                  * emissionProbs[nextState, observations[k+1]])
                if temp > -np.inf:
                    logsum = temp + np.log(1 + np.exp(logsum - temp))
            b[state, k] = logsum
    return b


def posterior_hmm(transProbs, initProbs, emissionProbs, states, observations):
    '''
    port for R library (HMM) posterior function
    yields most likely state at each time step
    ideal if we care about individual state errors
    '''
    assert np.isnan(transProbs).flatten().any() == False, 'nan exist'
    assert np.isnan(emissionProbs).flatten().any() == False, 'nan exist'
    f = forward_hmm(transProbs, initProbs, emissionProbs, states, observations)
    b = backward_hmm(transProbs, initProbs, emissionProbs, states, observations)
    probObservations = f[0, len(observations)-1]
    for i in range(1, len(states)):
        j = f[i, len(observations)-1]
        if j > -np.inf:
            probObservations = j + np.log(1 + np.exp(probObservations - j))
    posteriorProb = np.exp((f+b) - probObservations)
    return posteriorProb


# results
preds = []
probas = []
time_steps = 12  # months
for batch_number, batch_df in df.groupby(np.arange(len(df)) // time_steps):
    ev_obs = np.array(list(batch_df.apply(preprocess_series, axis=1).values))
    viterbi_out = viterbi(transition_matrix, p_init, emissions_matrix, states, ev_obs)
    state_probas = np.max(posterior_hmm(transition_matrix, p_init, emissions_matrix, states, ev_obs), axis=0)
    preds.append(viterbi_out)
    probas.append(state_probas)

# check outs
len(viterbi_out)
len(ev_obs)
len(preds) * time_steps
len(df)
len(probas) * 12

# join preds
out_viterbi = np.concatenate(preds).ravel()
out_probas = np.concatenate(probas).ravel()

# apply preds to df
df['pred_labels'] = out_viterbi
df['label_probas'] = out_probas

#########################
### Plot Results
###############################
# custom color map
cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "tan", "blue"])

# q1
df_q1 = df[(df.time_index <= 4)].reset_index(drop=True)
df_q1['label_mean'] = df_q1.groupby('h3_hexs')['pred_labels'].transform('mean')
df_q1['label_mean'] = df_q1['label_mean'].round()
df_q1['prob_mean'] = df_q1.groupby('h3_hexs')['label_probas'].transform('mean')
df_q1 = df_q1[(df_q1.time_index == 4)].reset_index(drop=True)
df_q1 = df_q1[['label_mean', 'geometry', 'prob_mean']]
# turn to geoDF
df_q1 = gpd.GeoDataFrame(df_q1, geometry="geometry")
df_q1.plot(figsize=(16, 16), alpha=0.5, categorical=True, column='label_mean', legend=True, cmap=cmap)


# q2
df_q2 = df[(df.time_index > 4) & (df.time_index <= 8)].reset_index(drop=True)
df_q2['label_mean'] = df_q2.groupby('h3_hexs')['pred_labels'].transform('mean')
df_q2['label_mean'] = df_q2['label_mean'].round()
df_q2['prob_mean'] = df_q2.groupby('h3_hexs')['label_probas'].transform('mean')
df_q2 = df_q2[(df_q2.time_index == 8)].reset_index(drop=True)
df_q2 = df_q2[['label_mean', 'geometry', 'prob_mean']]

# turn to geoDF
df_q2 = gpd.GeoDataFrame(df_q2, geometry="geometry")
df_q2.plot(figsize=(16, 16), alpha=0.5, categorical=True, column='label_mean', legend=True, cmap=cmap)


# q3
df_q3 = df[(df.time_index > 8) & (df.time_index <= 12)].reset_index(drop=True)
df_q3['label_mean'] = df_q3.groupby('h3_hexs')['pred_labels'].transform('mean')
df_q3['label_mean'] = df_q3['label_mean'].round()
df_q3['prob_mean'] = df_q3.groupby('h3_hexs')['label_probas'].transform('mean')
df_q3 = df_q3[(df_q3.time_index == 12)].reset_index(drop=True)
df_q3 = df_q3[['label_mean', 'geometry', 'prob_mean']]

# turn to geoDF
df_q3 = gpd.GeoDataFrame(df_q3, geometry="geometry")
df_q3.plot(figsize=(16, 16), alpha=0.5, categorical=True, column='label_mean', legend=True, cmap=cmap)




##########
def plot_bokeh(gdf, str_title):
    # geoJSON gdf
    geo_src = bm.GeoJSONDataSource(geojson=gdf.to_json())
    # colormap
    cmap = LinearColorMapper(palette=RdBu[5][::-1], low=0, high=4)  # reverse
    # define web tools
    TOOLS = "pan, wheel_zoom, box_zoom, reset, hover, save"
    # set up bokeh figure
    p = figure(
        title=str_title,
        tools=TOOLS,
        toolbar_location="below",
        x_axis_location=None,
        y_axis_location=None,
        width=900,
        height=800
    )
    # remove the grid
    p.grid.grid_line_color = None
    # add a patch for each polygon in the gdf
    p.patches(
        'xs', 'ys',
        fill_alpha=0.7,
        fill_color={'field': 'label_mean', 'transform': cmap},
        line_color='black',
        line_width=0.5,
        source=geo_src
    )
    # set up mouse hover informations
    hover = p.select_one(bm.HoverTool)
    hover.point_policy = 'follow_mouse'
    hover.tooltips = [
        ("Control Label:", "@label_mean"),
        ("Label Predicted Probability:", "@prob_mean"),
    ]
    ticker = FixedTicker(ticks=[0, 1, 2, 3, 4])
    fixed_labels = dict({0: 'rebel control', 1: 'leaning rebel control',
                         2: 'contested', 3: 'leaning government',
                         4: 'government'})
    # add a color bar
    color_bar = bm.ColorBar(
        color_mapper=cmap,
        ticker=ticker,
        label_standoff=25,
        major_label_overrides=fixed_labels,
        location=(10, 0))
    p.add_layout(color_bar, 'right')
    return show(p)

plot_bokeh(gdf=df_q1, str_title='Measuring Latent Territorial Control: JAN-APR 19')
plot_bokeh(gdf=df_q2, str_title='Measuring Latent Territorial Control: MAY-JUL 19')
plot_bokeh(gdf=df_q3, str_title='Measuring Latent Territorial Control: AUG-DEC 19')







##########
# Emission Matrix
s1 = np.array([0.4, 0.175, 0.175, 0.25])  # rows sum to 1
s2 = np.array([0.05, 0.6, 0.175, 0.175])
s3 = np.array([0.05, 0.175, 0.6, 0.175])
s4 = np.array([0.05, 0.175, 0.175, 0.6])
s5 = np.array([0.6, 0.05, 0.175, 0.175])
emissions_matrix = np.vstack([s1, s2, s3, s4, s5])

# Initialize Markov Chain
# initial state probability vector
# probs of starting the sequence at a given state
# no prior of who controls what territory.
p_init = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Transition matrix
# probability of transitioning from one state to another
# rows: state at time t
# cols: state at time t+1
# row probs sum to 1
# off-diagonal mass: transition out of states
# diagonal mass: stay in state entered
t1 = np.array([0.5, 0.25, 0.025, 0.2, 0.025])
t2 = np.array([0.25, 0.15, 0.075, 0.5, 0.025])
t3 = np.array([0.05, 0.025, 0.050, 0.850, 0.025])
t4 = np.array([0.025, 0.075, 0.15, 0.125, 0.625])
t5 = np.array([0.05, 0.075, 0.475, 0.025, 0.375])
transition_matrix = np.vstack([t1, t2, t3, t4, t5])

forward_probs = np.exp(forward_hmm(transition_matrix, p_init, emissions_matrix, states, ev_obs))
np.sum(forward_probs[:, -1])



#
