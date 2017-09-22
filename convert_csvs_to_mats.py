import math
import numpy as np
import pandas as pd
import os
import tempfile
from itertools import product

from scipy.spatial.distance import pdist, squareform
import scipy.io as sio


# params
stepsize = pd.Timedelta('5 min')
horizon = pd.Timedelta('2 hour')
start = pd.to_datetime('2016-01-21 00:00:00')
end = pd.to_datetime('2016-01-22 00:00:00')

def load_df():
	# 1 distribution by source & time

	# load data
	orderdf = pd.read_csv('ignored_assets/orders.csv', parse_dates = [6])

	# Create map of driver IDs
	driver_ids=orderdf['driver_id'].unique()

	# create map of core stations
	core_stations = orderdf['start_district_hash'].unique()
	core_stations.sort()

	#Index stuff
	orderdf_f=orderdf[(~pd.isnull(orderdf['driver_id'])) & orderdf['dest_district_hash'].isin(core_stations)]
	myindex=pd.MultiIndex.from_arrays([orderdf_f['driver_id'],orderdf_f['timestamp']],names=('driver_id','timestamp'))

	orderdf2=orderdf_f.set_index(myindex)
	orderdf_s=orderdf2.sort_index()

	orderdf_s['time_bucket'] = orderdf_s['timestamp'].dt.round('5min')
	orderdf_s['pax_arrival_time'] = orderdf_s['timestamp'] + orderdf_s['expected_travel_time'] * pd.Timedelta('1 minute')
	orderdf_s['time_bucket_arrival'] = orderdf_s['pax_arrival_time'].dt.round('5min')

	return orderdf_s, core_stations

def get_posterior(orderdf):

	# create map of core stations
	core_stations = orderdf['start_district_hash'].unique()
	core_stations.sort()
	dates = [(start + j*stepsize).time() for j in range(288)]
	iterables = [dates, core_stations]
	mindex = pd.MultiIndex.from_product(iterables, names=['time_bucket', 'start_district_hash'])
	mcolumns = pd.MultiIndex.from_arrays([core_stations], names=['dest_district_hash'])

	base = pd.DataFrame(index=mindex, columns=mcolumns).fillna(0)

	orderdf['time_bucket_s'] = orderdf['time_bucket'].map(lambda t: t.time())
	pt = pd.pivot_table(orderdf[orderdf['time_bucket'] < start], index=['time_bucket_s','start_district_hash'], 
               columns=['dest_district_hash'], aggfunc=len, 
               values='order_id', fill_value=0, dropna=False)
	pt.index.names = ['time_bucket','start_district_hash']
	pt = (base + pt).fillna(0)

	prior = pd.pivot_table(orderdf[orderdf['time_bucket'] < start], 
		index=['start_district_hash'], 
		columns=['dest_district_hash'], aggfunc=len, 
		values='order_id', fill_value=0) + 1/66.
	#weight of the prior (based on hours and days of samples)
	ndays = 20.
	nhours = 24.
	w = 1e-6

	# sum both
	count = pt + w*prior

	# get posterior
	posterior = count.div( count.sum(axis=1), axis = 0).fillna(0)
	return posterior

def get_predictor(look_forward, posterior, predictions, core_stations):
	predictor = np.zeros((288,), dtype=np.object)

	for i in range(288):
	    begin = start + i*stepsize
	    index = [begin + (j+1)*stepsize for j in range(look_forward)]
	    predictor[i] = np.zeros((len(index)+1,), dtype=np.object)
	    base = pd.DataFrame(
	            np.zeros((len(core_stations),len(core_stations)),dtype=np.float64),
	            index=core_stations,
	            columns = core_stations
	        )
	    predictor[i][0] = base.as_matrix()

	    for j,t in enumerate(index):
	        interim = predictions[(predictions['predicted_at'] == begin) & (predictions['time_bucket'] == t)][core_stations].T
	        print interim
	        interim.columns = ['demand']
	        interim = posterior.loc[t.time()].multiply(interim['demand'], axis='index').fillna(0)
	        interim[interim < 0] = 0
	        predictor[i][j+1] = (base + interim).fillna(0).as_matrix().round()
	return predictor

def load_predictions(look_forward, look_back):
	# 3. load lstm predictions
	newdf = pd.read_csv('ignored_assets/to_lstm_'+str(look_back)+'b_'+str(look_forward)+'f.csv')
	newdf['predicted_at'] = pd.to_datetime(newdf['predicted_at'])
	newdf['time_bucket'] = pd.to_datetime(newdf['timestamp'])
	newdf = newdf.drop('timestamp', axis = 1)
	return newdf

def main():
	print 'Hi there'
	desired_path = '/home/rdit/development/cs341/ignored_assets/predictors'
	print 'loading df'
	orderdf, core_stations = load_df()
	print 'computing posterior'
	posterior = get_posterior(orderdf)
	horizons = [3, 6, 12, 24, 48]
	print 'begin processing the predictions'
	for look_back, look_forward in product(horizons, horizons):
		print 'look_back: ', look_back, ', look_forward: ', look_forward
		predictions = load_predictions(look_forward, look_back)
		predictor = get_predictor(look_forward, posterior, predictions, core_stations)
		sio.savemat(desired_path+'/tod_predictions_'+str(look_back)+'b_'+str(look_forward)+'f_np.mat',{'predictor': predictor})

if __name__ == '__main__':
	main()