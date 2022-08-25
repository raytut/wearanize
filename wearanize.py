# -*- coding: utf-8 -*-
"""
Copyright 2022, Frederik D. Weber

Notes:

conda create --name=hbs_wearables python=3
conda activate hbs_wearables

# install the requirements
pip install -r requirements.txt

# install them manually on the python version
# first make sure pip is up to date

python.exe -m pip install --upgrade pip
python.exe -m pip install -r requirements.txt


#or just do them step by step
python.exe -m pip install mne
python.exe -m pip install heartpy
python.exe -m pip install plotnine
...

or to save the enironment package requirements (also unused)
python.exe -m pip freeze > requirements.txt

or use project specific used packages:
pip install pipreqs
pipreqs /path/to/this/project


"""
# imports #

import rapidhrv
import mne
import matplotlib
import numpy
import warnings
import os
import glob
import csv
import datetime
import heartpy as hp
import EDFlib
import pyedflib
import shutil
import pandas
import statistics
import scipy
from sklearn.linear_model import LinearRegression
import argparse
import pathlib
import statsmodels
import pytz# Activpal
import errno
from collections import namedtuple
import pyphysio as ph
import sys
if sys.version_info >= (3, 6):
	import zipfile
	from zipfile import ZipFile
else:
	import zipfile36 as zipfile

import tempfile

from joblib import Parallel, delayed
import logging
import traceback
import subprocess

from zmax_edf_merge_converter import file_path, dir_path, dir_path_create, fileparts, zip_directory, safe_zip_dir_extract, safe_zip_dir_cleanup, raw_prolong_constant, read_edf_to_raw, edfWriteAnnotation, write_raw_to_edf, read_edf_to_raw_zipped, write_raw_to_edf_zipped, raw_zmax_data_quality
from e4_converter import read_e4_to_raw_list, read_e4_to_raw, e4_concatenate, e4_concatente_par
import apl_converter as apl

# constants #
FILE_EXTENSION_WEARABLE_ZMAX_REEXPORT = "_merged"

# classes #

class StreamToLogger(object):
	def __init__(self, logger, log_level=logging.INFO):
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ''

	def write(self, buf):
		for line in buf.rstrip().splitlines():
			self.logger.log(self.log_level, line.rstrip())



# functions #
'''
def plot_synched(filepath_csv_in, filepath_plot_export):

	df_csv_in = pandas.read_csv(filepath_csv_in)
	df_csv_in.reset_index()  # make sure indexes pair with number of rows
	plot = (plotnine.ggplot(data=df_csv_in, mapping=plotnine.aes(x=datetime,y=wearable))
			+ plotnine.geom_rect(mapping=plotnine.aes(NULL,NULL,xmin=start,xmax=end,fill=modality), ymin=0, ymax=1, colour="back", size=0.5, alpha=0.5)
			+ plotnine.scale_fill_manual(values=c("acc"="blue", "hr"="red")))


	plotnine.ggsave(filename=filepath_plot_export + ".pdf", plot=plot, height=5, width=10, units = 'cm', dpi=300)
'''

# =============================================================================
# 
# =============================================================================
def parse_wearable_filepath_info(filepath):
	split_str = '_'

	path, name, extension = fileparts(filepath)

	name_parts = name.split(split_str)
	path_parts = path.split(os.sep)
	subject_path_id = ''
	for p in path_parts:
		if 'sub-HB' in p:
			subject_path_id = p
			break

	subject_file_id = name_parts[0]
	period = name_parts[1]
	datatype = name_parts[2]
	if name_parts[2]=="app-ema":
		device_wearable = "app"
	else:
		device_wearable=name_parts[3]
		
	if name_parts.__len__() > 4:
		session = split_str.join(name_parts[4:])
	else:
		session = ''

	return {'subject_path_id': subject_path_id,'subject_file_id': subject_file_id, 'filepath':  filepath, 'period':  period, 'datatype':  datatype, 'device_wearable':  device_wearable, 'session': session}

# =============================================================================
# 
# =============================================================================

def app_to_long(filepath): 
	# Parse file info
	file_info=parse_wearable_filepath_info(filepath)
	# Check if EMA or not
	if file_info["device_wearable"]!="app":
		warnings.warn("FILE NOT EMA APP DATA")
	else:
		# get session identifier for clean-up
		prefix, session=str.split(file_info['period'], "-")
		# read data for said session, get id files
		df_ema=pandas.read_csv(file_info['filepath'])
		df_ema_ids=df_ema.iloc[:,0:4]
		df_ema_ids['session']=file_info['period']
		# Clean up headers
		df_ema=df_ema.filter(regex=(session +'$'),axis=1)
		df_ema=df_ema.rename(columns = lambda x : str(x)[:-2])
		# Rejoin with ema ids
		df_ema=df_ema_ids.join(df_ema)
		return(df_ema)
	

def full_app_to_long(filename, output=None, anon_datetime=False):
	app_wide=pandas.read_csv(filename, na_values=' ', low_memory=False)
	app_long=pandas.DataFrame()
	
	for week in range(1,4):
		# subset
		app_pre=app_wide[app_wide.SURVEY_ID==week]
		app_pre_ids=app_pre.iloc[:,0:4]
		# clean up headers
		app_pre=app_pre.filter(regex=(str(week) +'$'),axis=1)
		app_pre=app_pre.rename(columns = lambda x : str(x)[:-2])
		app_pre=app_pre_ids.join(app_pre)
		# rejoin
		app_long=pandas.concat([app_long, app_pre], axis=0)
	
	# reformat date times
	app_long.loc[app_long.iloc[:,4]==' ']='nan'
	app_long.iloc[:,4]=pandas.to_datetime(app_long.iloc[:,4], format='%m/%d/%Y %H:%M:%S')
	app_long.loc[app_long.iloc[:,71]==' ']='nan'
	app_long.iloc[:,71]=pandas.to_datetime(app_long.iloc[:,71], format='%m/%d/%Y %H:%M:%S')
	
	# metric calculations
	app_long['completion_time_beep']=app_long['EMA_timestamp_end_beep_']- app_long['EMA_timestamp__start_beep_']
	app_long['completion_time_window_yn']=app_long['completion_time_beep'].between(datetime.timedelta(seconds=(400*70/1000)), datetime.timedelta(seconds=(60*15)))
	# Per subject/session
	app_group=pandas.DataFrame()
	for name, group in app_long.groupby(by=['EMA_ID', 'SURVEY_ID']):
		group['completion_time_average']=numpy.mean(group['completion_time_beep'])
		group['compliance']=len(group.EMA_ID)/60
		app_group=pandas.concat([app_group, group])
		
	# anonymize beep times
	if anon_datetime == True:
		# round the seconds down, then keep the time
		app_long['EMA_timestamp__start_beep_'] = app_long['EMA_timestamp__start_beep_'].dt.floor('T')
		app_long['EMA_timestamp__start_beep_'] = pandas.to_datetime(app_long['EMA_timestamp__start_beep_']).dt.time
		app_long['EMA_timestamp_end_beep_'] = app_long['EMA_timestamp_end_beep_'].dt.floor('T')
		app_long['EMA_timestamp_end_beep_'] = pandas.to_datetime(app_long['EMA_timestamp_end_beep_']).dt.time

	if output!=None:
		app_group.to_csv(output)
		
	return app_group

# =============================================================================
# 
# =============================================================================
def window_selector(raw):
	"""
	Given a raw signal, generate datetime and duration to create search windows
	"""
	try:
		date=raw.info['meas_date']
		duration=(raw.last_samp+1)*(1/raw.info['sfreq'])
		return date, duration
	except:
		warnings.warn("Check signal is mne.raw and has all necessary info")
		pass
# =============================================================================
# 
# =============================================================================
def get_raw_by_date_and_time(filepath,  datetime_ts, duration_seconds,  wearable='zmx', resample_hz=None, offset_seconds=0.0): 
	"""get raw data file according to time stamps
	"""
	# parse file parts
	sub_path, wearable_file, extension = fileparts(filepath)
	
	#find the start date and end date
	start_date = datetime_ts
	end_date = start_date + datetime.timedelta(seconds=duration_seconds)
	
	# Get primary signal
	if wearable == 'zmx':
		raw = read_edf_to_raw_zipped(filepath)
	elif wearable == 'zmx-merge':
		raw = read_edf_to_raw_zipped(filepath, format='edf')
	elif wearable == 'emp':
		raw = read_e4_to_raw_list(filepath)
	elif wearable == 'apl': #TODO: Add apl files
		warnings.warn("ACTIVPAL Files too large for efficient syncing")
		pass
	 
	# convert to dateframe and subset time window    
	raw_df = raw.to_data_frame(time_format='datetime')
	raw_df = raw_df[(raw_df.time > start_date) & (raw_df.time < end_date)]   
	raw_df = raw_df.set_index('time', drop=True)
	
	#list all files and search all the relevant files that fall within these time limits
	print("Searching all wearable files in directory...")
	for wearables in ['zmx', 'apl', 'emp']: 
		# Skip if we have the same modality
		if wearables != wearable:
			
			# check all files in directory to match window
			if wearables == 'zmx':
				file_list = find_wearable_files(sub_path, wearable="zmax")
				channel_df = pandas.DataFrame()
				for file in file_list:
					try:
						raw_channel = read_edf_to_raw_zipped(filepath)
						raw_channel = raw_channel.to_data_frame(time_format='datetime')
						raw_channel = raw_channel[(raw_channel.time > start_date) & (raw_channel.time < end_date)]
						if raw_channel.size != 0: 
							raw_channel = raw_channel.set_index('time', drop=True)
							channel_df = pandas.concat([channel_df,raw_channel], axis=1)
					except:
						pass
				raw_df = pandas.concat([raw_df,channel_df], axis=1)
				
			# Empatica
			elif wearables == "emp":  
				file_list = find_wearable_files(sub_path, wearable="empatica")
				channel_df = pandas.DataFrame()
				for file in file_list:
					if (file.endswith('emp_full.zip') == False) and (file.endswith('emp.zip') == False):
						try:
							raw_channel = read_e4_to_raw(sub_path + os.sep + file)
							raw_channel = raw_channel.to_data_frame(time_format='datetime')
							raw_channel = raw_channel[(raw_channel.time > start_date) & (raw_channel.time < end_date)]
							if raw_channel.size != 0: 
								raw_channel = raw_channel.set_index('time', drop=True)
								channel_df = pandas.concat([channel_df,raw_channel], axis=1)
						except:
							pass
				raw_df = pandas.concat([raw_df,channel_df], axis=1)
				
			elif wearables == 'apl':
				apl_raw = apl_window_to_raw(filepath, wearable, buffer_seconds=offset_seconds)
				channel_df = apl_raw.to_data_frame(time_format='datetime')
				channel_df = channel_df.set_index('time', drop=True)
				raw_df = pandas.concat([raw_df, channel_df], axis=1)
				
	# recreate a raw mne file with all channels 
	mne_info=mne.create_info(ch_names=list(raw_df.columns), sfreq=raw.info['sfreq'])
	raw_full=mne.io.RawArray(raw_df.to_numpy().transpose(), mne_info) 
	raw_full.set_meas_date( raw_df.index[0])
	if resample_hz != None:
		raw_full=uneven_raw_resample(raw_full, resample_hz)

	return raw_full


def uneven_raw_resample(raw, resample_hz, interpolation_method='pad'):
	# set resampling string for use with pandas
	resample_offset = (str(1/resample_hz)+'S')
	raw_df = raw.to_data_frame(time_format='datetime')
	raw_df = raw_df.set_index('time', drop=True)
	raw_df = raw_df.resample(resample_offset).ffill()
	raw_df = raw_df.interpolate(method=interpolation_method)
	# convert back to mne.rwa
	mne_info = mne.create_info(ch_names=list(raw_df.columns), sfreq=resample_hz)
	raw = mne.io.RawArray(raw_df.to_numpy().transpose(), mne_info) 
	raw.set_meas_date(raw_df.index[0])
	return raw

# =============================================================================
# 
# =============================================================================
def raw_detect_heart_rate_PPG(raw, ppg_channel, enhance_peaks=True, device='emp'):
	"""
	Parameters
	---------------
	raw: mne.raw
		mne raw object containing blood volume pulse channel
	channel_name: str
		name of channel containing bvp data
	enhance_peaks: bool
		Determines whether to include peak enhancements, default False (recommended for empatica)
	windowsize: int
		Window size for use with peak detection algorithm. Recommended 1 for Empatica, 2 for ZMAX
	"""

	# first take care of the nans 
	raw_series = pandas.Series(raw.get_data(ppg_channel)[0])
	raw_series = raw_series.fillna(raw_series.mean())
	raw_series = raw_series.array
	
	# peak enhancement
	if enhance_peaks == True:
		raw_series = hp.enhance_peaks(raw_series, iterations=3)

	# set filtering parameters
	if device == 'emp':
		fp = 0.6
		fs = 8
	else:
		fp = 0.75
		fs = 8
		
	# filtering based on lian et al 2018
	filtered = hp.filter_signal(raw_series, cutoff=[fp, fs], sample_rate=raw.info['sfreq'], order=4,  filtertype='bandpass')
	# run peak detection
	hr, measures = hp.process(filtered, int(raw.info['sfreq']), windowsize=2)
	return hr, measures
	
	
# =============================================================================
# 
# =============================================================================
def interpolate_heart_rate(raw, ppg_channel):
	"""
	"""

# =============================================================================
# 
# =============================================================================
def check_zmax_integrity():
	"""
	#Check if all the files are in order of the files
	#Are there any files missing in the dates
	#Check if a default date is present after a series of recordings or already starting with the first
	#Exlude files that are just very short recordings if the number of recordings is XXX
	#Optionally look for low Voltage in Battery (some lower threshold that was crossed to mark some forced shuttoff
	"""

def parseDevice(device):
	if device == 'all':
		device = 'zmx|emp|app|apl'
	return device

# =============================================================================
# 
# =============================================================================
def find_wearable_files(parentdirpath, wearable):
	"""
	finds all the wearable data from different wearables + app data in the HB file structure given the parent path to the subject files
	:param wearable:
	:return:
	"""
	filepath_list = []
	if wearable == 'zmax':
		wearable = 'zmx'
	elif wearable == 'empatica':
		wearable = 'emp'
	elif wearable == 'apl':
		wearable = 'apl'
	else:
		wearable = ''
	filepath_list = glob.glob(parentdirpath + os.sep + "**" + os.sep + "sub-HB" + "*" + "_wrb_" + wearable + "*.*",recursive=True)
	filepath_list = filepath_list + (glob.glob(parentdirpath + os.sep + "**" + os.sep + "pre*" + os.sep + "app" + os.sep + "*"))
	# compatible with python versions < 3.10 remove the root_dir
	for i, filepath in enumerate(filepath_list):
		filepath_list[i] = filepath.replace(parentdirpath + os.sep,"")

	return filepath_list

# =============================================================================
# 
# =============================================================================
def parse_wearable_data_write_csv(parentdirpath, filepath_csv_out, device='all'):
	device = parseDevice(device)
	filepath_list = find_wearable_files(parentdirpath, device)

	with open(filepath_csv_out, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(['subject_path_id', 'subject_file_id', 'filepath', 'period', 'datatype', 'device_wearable', 'session'])
		for filepath in filepath_list:
			info = parse_wearable_filepath_info(filepath)
			writer.writerow([info['subject_path_id'],info['subject_file_id'], info['filepath'], info['period'], info['datatype'], info['device_wearable'], info['session']])

# =============================================================================
# Parser: adds info
# =============================================================================
def parse_wearable_data_with_csv_annotate_datetimes(parentdirpath, filepath_csv_in, filepath_csv_out, device='all'):
	device = parseDevice(device)
	df_csv_in = pandas.read_csv(filepath_csv_in, quoting=csv.QUOTE_NONNUMERIC)
	df_csv_in.reset_index()  # make sure indexes pair with number of rows

	with open(filepath_csv_out, 'w', newline='') as csvfile2:
		writer = csv.writer(csvfile2, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')

		header_new = numpy.append(df_csv_in.columns.values, ['signal', 'rec_start_datetime', 'rec_stop_datetime', 'rec_duration_datetime', 'sampling_rate_max_Hz', 'rec_quality'])
		writer.writerow(header_new)
		for i, row in df_csv_in.iterrows():
			filepath = row['filepath']
			filepath_full = parentdirpath + os.sep + filepath
			device_wearable = row['device_wearable']
			session = row['session']
			
			signal='unretrieved'
			rec_start_datetime = 'unretrieved'
			rec_stop_datetime = 'unretrieved'
			rec_duration_datetime = 'unretrieved'
			sampling_rate_max_Hz = 'unretrieved'
			rec_quality = 'unretrieved'

			print("attempt read info from file: " + filepath_full)
			try:
				if device_wearable == 'zmx' and 'zmx' in device:
					if session in ["1", "2", "3", "4", "5", "6", "7", "8"]:
						raw = read_edf_to_raw_zipped(filepath_full, format="zmax_edf", zmax_ppgparser=zmax_ppgparser, zmax_ppgparser_exe_path=zmax_ppgparser_exe_path, zmax_ppgparser_timeout=zmax_ppgparser_timeout)
						rec_start_datetime = raw.info['meas_date']
						rec_stop_datetime = rec_start_datetime + datetime.timedelta(seconds=(raw._last_time - raw._first_time))
						rec_duration_datetime = datetime.timedelta(seconds=(raw._last_time - raw._first_time))
						sampling_rate_max_Hz = raw.info['sfreq']
						rec_quality = raw_zmax_data_quality(raw)
						signal="zmx_all"
						row_new = numpy.append(row.values, [signal, rec_start_datetime, rec_stop_datetime, rec_duration_datetime, sampling_rate_max_Hz, rec_quality])
						writer.writerow(row_new)
						
				elif device_wearable == 'emp' and 'emp' in device:
				# Make this a try, to avoid the improper files and concatenated ones
					try: 
						# If we cant turn into integer, its probably right	
						session=int(session)
						emp_zip=zipfile.ZipFile(filepath_full)
						tzinfo=datetime.timezone(datetime.timedelta(0))
						# Estimate different parameters per signal
						for signal_types in ['IBI.csv','BVP.csv', 'HR.csv','EDA.csv','TEMP.csv', 'ACC.csv']:
							raw = pandas.read_csv(emp_zip.open(signal_types))
							if signal_types == "IBI.csv":
								signal = signal_types.removesuffix('.csv')
								rec_start_datetime = datetime.datetime.fromtimestamp(int(float(raw.columns[0])), tz=tzinfo)
								rec_stop_datetime = rec_start_datetime + datetime.timedelta(seconds=(raw.iloc[-1,0]))
								rec_duration_datetime=(rec_stop_datetime - rec_start_datetime)
								sampling_rate_max_Hz = "custom"
								rec_quality= raw[" IBI"].sum()/raw.iloc[-1,0]
								row_new = numpy.append(row.values, [signal, rec_start_datetime, rec_stop_datetime, rec_duration_datetime, sampling_rate_max_Hz, rec_quality])
								writer.writerow(row_new)
							else:
								signal = signal_types.removesuffix('.csv')
								rec_start_datetime=datetime.datetime.fromtimestamp(int(float(raw.columns[0])), tz=tzinfo)
								rec_stop_datetime = rec_start_datetime + datetime.timedelta(((((len(raw.index)-1)*(1/raw.iloc[0,0]))/60)/60)/24)
								rec_duration_datetime=(rec_stop_datetime - rec_start_datetime)
								sampling_rate_max_Hz=str(raw.iloc[0,0])
								if signal=='EDA':
									rec_quality=100-(100*(raw[0] < 0.02).sum()/(len(raw.index)-1))
								row_new = numpy.append(row.values, [signal, rec_start_datetime, rec_stop_datetime, rec_duration_datetime, sampling_rate_max_Hz, rec_quality])
								writer.writerow(row_new)
					except:
						pass
					
				elif device_wearable == 'apl' and 'apl' in device:
					meta, raw = load_activpal_data(filepath_full)
					tz = pytz.timezone('Europe/Amsterdam')
					# Get info
					signal = 'apl'
					rec_start_datetime = tz.localize(meta[5])
					rec_stop_datetime = tz.localize(meta[6])
					rec_duration_datetime = (rec_stop_datetime - rec_start_datetime)
					sampling_rate_max_Hz = meta[3]
					rec_quality = 'unretrieved'
					row_new = numpy.append(row.values, [signal, rec_start_datetime, rec_stop_datetime, rec_duration_datetime, sampling_rate_max_Hz, rec_quality])
					writer.writerow(row_new)
				
				elif device_wearable == 'app' and 'app' in device:
					# read in full file
					df_ema = app_to_long(filepath_full)
					# Clean up date times
					df_ema['EMA_timestamp__start_beep_'] = pandas.to_datetime(df_ema['EMA_timestamp__start_beep_'], utc=True)
					df_ema['EMA_timestamp__start_beep_'] = df_ema.set_index(df_ema['EMA_timestamp__start_beep_']).tz_convert("Europe/Amsterdam").index
					df_ema['EMA_timestamp_end_beep_'] = pandas.to_datetime(df_ema['EMA_timestamp_end_beep_'], utc=True)
					df_ema['EMA_timestamp_end_beep_'] = df_ema.set_index(df_ema['EMA_timestamp_end_beep_']).tz_convert("Europe/Amsterdam").index
					# Get info
					signal = 'app'
					rec_start_datetime = df_ema['EMA_timestamp__start_beep_'][0]
					rec_stop_datetime = df_ema['EMA_timestamp_end_beep_'].iloc[-1]
					rec_duration_datetime = (rec_stop_datetime - rec_start_datetime)
					sampling_rate_max_Hz = "custom"
					rec_quality = (df_ema['EMA_timestamp_end_beep_'] - df_ema['EMA_timestamp__start_beep_']).astype('timedelta64[ms]')
					rec_quality = len(rec_quality[ rec_quality > (66*500)])/60
					row_new = numpy.append(row.values, [signal, rec_start_datetime, rec_stop_datetime, rec_duration_datetime, sampling_rate_max_Hz, rec_quality])
					writer.writerow(row_new)
					
			except:
				print("cannot read info from file: " + filepath_full)
				rec_start_datetime = 'retrieval_failed'
				rec_stop_datetime = 'retrieval_failed'
				rec_duration_datetime = 'retrieval_failed'
				sampling_rate_max_Hz = 'retrieval_failed'
				rec_quality = 'retrieval_failed'
				row_new = numpy.append(row.values, [signal, rec_start_datetime, rec_stop_datetime, rec_duration_datetime, sampling_rate_max_Hz, rec_quality])
				writer.writerow(row_new)

# =============================================================================
# 
# =============================================================================
def chunks(a, n, nstep):
	for i in range(0, len(a), nstep):
		yield a[i:i + n]

# =============================================================================
# 
# =============================================================================
def find_consecutive_consistent_lags_index(lags, max_merge_lag_difference, lag_forward_pair_steps=1):
	consecutive_consistent_lags = numpy.zeros(len(lags), dtype=bool)
	for ilag in range(0, len(lags)-lag_forward_pair_steps):
		for iforward in range(1,lag_forward_pair_steps+1):
			lag = lags[ilag]
			next_lag = lags[ilag+iforward]
			if abs(lag-next_lag) <= max_merge_lag_difference:
				consecutive_consistent_lags[ilag] = True
				consecutive_consistent_lags[ilag+iforward] = True
	return numpy.where(consecutive_consistent_lags)[0]

# =============================================================================
# 
# =============================================================================
def linear_regression(x, y):
	X = x.reshape(-1,1)
	model = LinearRegression().fit(X, y)
	r_sq = model.score(X, y)
	intercept = model.intercept_
	slope = model.coef_
	return r_sq, intercept, slope

# =============================================================================
# 
# =============================================================================
def datetime_overlap(list_datetime_start_stop, list2_datetime_start_stop):
	latest_start = max(list_datetime_start_stop[0], list2_datetime_start_stop[0])
	earliest_stop = min(list_datetime_start_stop[1], list2_datetime_start_stop[1])
	delta = (earliest_stop - latest_start)
	#overlap = max(datetime.timedelta(seconds=0), delta)
	overlap = max(datetime.timedelta(seconds=0), delta)
	return overlap


# =============================================================================
# 
# =============================================================================
def sync_reach(list_datetimes_paired, min_reach_duration_seconds=3600):
	"""from a list of lists with paired start and stop (of events) datetimes
		find all the overlaps in time considering and additional reach buffer.
		this assumes the start and stop datetime pairs are already ascending in time, i.e. start <= stop
		The list can be unordered with respect to the datetime pairs.
		The list only contains the matching event pairs in reach as with their respective index in the original list
		together with additional offset and real overlap info
		A list is returned and contains further lists with each having the items:
			index of the original start_stop pair in the list with the earliest start
			index of the original start_stop pair in reach with the later start
			the offset in datetime.timedelta to add to the first pair to match the second pair start time
			the overlap in datetime.timedelta that the events share
		The returned list is ordered ascending by the earliest start time of the matching pairs (i.e. the onese in reach)
		Note that if overlap smaller than 0 is returned they might be in reach but actually do not overlap in time but are that far apart

	Parameters
	----------
	%(list_datetimes_paired)s
	%(min_reach_duration_seconds)s, by default is 3600 seconds (= 1 hour)

	Returns
	-------
	list :
			list[0]: index of the original start_stop pair in the list with the earliest start
			list[1]: index of the original start_stop pair in reach with the later start
			list[2]: the offset in datetime.timedelta to add to the first pair to match the second pair start time
			list[3]: the overlap in datetime.timedelta that the events share
	"""
	list_indices_paired_offsetTo2nd_overlap = []
	min_reach_duration_datetime = datetime.timedelta(seconds=min_reach_duration_seconds)
	if len(list_datetimes_paired) < 2:
		return []
	else:
		for i, datetimes_paired_test in enumerate(list_datetimes_paired):
			for i2, datetimes_paired_match in enumerate(list_datetimes_paired[(i+1):]):
				i2nd = i2+i+1
				if datetimes_paired_test[0] <= datetimes_paired_match[1]: # test starts before or with match
					if (datetimes_paired_test[1] + min_reach_duration_datetime) < datetimes_paired_match[0]: # test does not reach match
						continue
					else: # test overlaps with match to its right
						overlap_datetime = datetime_overlap(datetimes_paired_test,datetimes_paired_match)
						offset_datetime = datetimes_paired_match[0]-datetimes_paired_test[0]
						list_indices_paired_offsetTo2nd_overlap.append([i, i2nd, offset_datetime, overlap_datetime])
				else: # test starts after match
					if (datetimes_paired_match[1] + min_reach_duration_datetime) < datetimes_paired_test[0]: #match does not reach test
						continue
					else:
						overlap_datetime = datetime_overlap(datetimes_paired_match,datetimes_paired_test)
						offset_datetime = datetimes_paired_test[0]-datetimes_paired_match[0]
						list_indices_paired_offsetTo2nd_overlap.append([i2nd, i, offset_datetime, overlap_datetime])
		#return ordered ascending in start datetimes (and endtimes if equal) of the listed reached items
		return sorted(list_indices_paired_offsetTo2nd_overlap, key=lambda i_i2_off_over: (list_datetimes_paired[i_i2_off_over[0]][0], list_datetimes_paired[i_i2_off_over[0]][1]), reverse=False)

# =============================================================================
#
# =============================================================================
def sync_signals(signal_ref, signal_sync, fsample, chunk_size_seconds=60*10, chunk_step_seconds=60*5, lag_merge_window_seconds=60*20, max_merge_lag_difference_seconds=0.5, threshold_chunk_min_match=2, allow_anti_correlation=False):

	chunk_size=max(1,round(fsample*chunk_size_seconds))
	chunk_step=max(1,round(fsample*chunk_step_seconds))
	lag_merge_window=max(1,round(fsample*lag_merge_window_seconds))
	max_merge_lag_difference=max(1,round(fsample*max_merge_lag_difference_seconds))

	sigchunks = chunks(signal_sync, chunk_size, chunk_step)
	correlation_chunks_lags_and_max_val = numpy.array([[],[]])

	lag_forward_pair_steps = lag_merge_window//chunk_step

	for chunk in sigchunks:
		correlation = scipy.signal.correlate(chunk, signal_ref, mode='full', method='auto')
		lags = scipy.signal.correlation_lags(chunk.size, signal_ref.size, mode="full")
		if allow_anti_correlation:
			correlation = abs(correlation)
		ind_max_correlation = numpy.argmax(correlation)

		correlation_max_value = correlation[ind_max_correlation]
		lag_add_to_chunk = -lags[ind_max_correlation]

		correlation_chunks_lags_and_max_val = numpy.append(correlation_chunks_lags_and_max_val, numpy.array([[lag_add_to_chunk],[correlation_max_value]]),axis=1)

	correlation_chunks_lags_and_max_val_relative = correlation_chunks_lags_and_max_val[0,] - range(0,len(correlation_chunks_lags_and_max_val[0,])*chunk_step,chunk_step)
	consecutive_consistent_lags_index = find_consecutive_consistent_lags_index(lags=correlation_chunks_lags_and_max_val_relative, max_merge_lag_difference=max_merge_lag_difference, lag_forward_pair_steps=lag_forward_pair_steps)
	n_matching_chunks = len(consecutive_consistent_lags_index)

	lag = None
	dilation = None
	lag_after_dilation = None
	sample_rate_adaptation_factor = None
	if  n_matching_chunks >= threshold_chunk_min_match:
		r_sq, intercept, slope = linear_regression(consecutive_consistent_lags_index, correlation_chunks_lags_and_max_val[0,consecutive_consistent_lags_index])
		#y = slope*x + intercept
		first_matching_chunk_index = numpy.min(consecutive_consistent_lags_index)
		#first_matching_chunk_lag = correlation_chunks_lags_and_max_val[0,first_matching_chunk_index]
		first_matching_chunk_lag = slope*first_matching_chunk_index + intercept
		lag = first_matching_chunk_lag + first_matching_chunk_index*chunk_step
		dilation = slope/chunk_step
		lag_after_dilation = lag*dilation
		sample_rate_adaptation_factor = 1/dilation
	lag_seconds = lag/fsample
	lag_after_dilation_seconds = lag_after_dilation/fsample
	return lag_seconds, dilation, lag_after_dilation_seconds, sample_rate_adaptation_factor


# =============================================================================
# 
# =============================================================================

def raw_append_signal(raw, signal, ch_name):
	# Create MNE Raw object and add to a list of objects
	mne_info = mne.create_info(ch_names=[ch_name], sfreq=raw.info.get("sfreq"), ch_types="misc")
	mne_raw_signal = mne.io.RawArray([signal], mne_info)
	raw.add_channels([mne_raw_signal])
	return raw


# =============================================================================
#
# =============================================================================

def raw_append_integrate_acc(raw, ch_name_acc_x, ch_name_acc_y, ch_name_acc_z, integrated_ch_name='integrated_acc', window=None):
	"""
	    Parameters
	----------
	raw: mne.raw 
		File containing 3-axis accelerometer data
	ch_name_acc*: str
		names of channels in mne file containing the x, y, and z directions from accelerometer.
	integrated_ch_name: str
		name to give new channel containing integrated displacement factor
	window: int
		Moving average window in seconds
	Returns
	-------
	mne.raw: a new mne raw file with an added channel for the integrated acc
	""" 
	# Get the data from channels
	acc_x = raw.get_data(ch_name_acc_x)
	acc_y = raw.get_data(ch_name_acc_y)
	acc_z = raw.get_data(ch_name_acc_z)
	
	# Caclulate differences between consequetive samples
	acc_x_dis = numpy.array(abs(numpy.diff(acc_x, prepend = acc_x[0][0])))
	acc_y_dis = numpy.array(abs(numpy.diff(acc_y, prepend = acc_y[0][0])))
	acc_z_dis = numpy.array(abs(numpy.diff(acc_z, prepend = acc_z[0][0])))
	
	# Calculate mean displacement
	net_displacement = numpy.sqrt(acc_x_dis**2+acc_y_dis**2+acc_z_dis**2)[0]
	
	# apply moving average
	if window != None: 
		window =int( window*raw.info['sfreq'])
		net_displacement = pandas.Series(net_displacement).rolling(window).mean() 
		net_displacement = net_displacement.to_numpy()
		
	return raw_append_signal(raw, net_displacement, integrated_ch_name)


# =============================================================================
#
# =============================================================================

def raw_get_integrated_acc_signal(raw, ch_name_acc_x, ch_name_acc_y, ch_name_acc_z, integrated_ch_name="integrated_acc", resample_Hz=None):

	if resample_Hz is not None:
		raw = raw.resample(resample_Hz)

	return raw_append_integrate_acc(raw, ch_name_acc_x, ch_name_acc_y, ch_name_acc_z).get_data(picks=[integrated_ch_name])


# =============================================================================
# 
# =============================================================================

def get_signal(filepath, wearable, type='acc', resample_Hz=None):
	path, name, extension = fileparts(filepath)
	raw = None
	if type not in ['acc', 'hr']:
		TypeError('type not known')
	if type == 'acc':
		ch_name_signal = 'integrated_acc'
	elif type == 'hr':
		ch_name_signal = 'hr' #TODO change to a matching one
	if wearable == 'zmx':
		if FILE_EXTENSION_WEARABLE_ZMAX_REEXPORT not in name:
			raw = read_edf_to_raw_zipped(filepath, format="zmax_edf")
		else:
			raw = read_edf_to_raw_zipped(filepath, format="edf")
		if type == 'acc':
			raw_get_integrated_acc_signal(raw, ch_name_acc_x='dX', ch_name_acc_y='dY', ch_name_acc_z='dZ', integrated_ch_name=ch_name_signal, resample_Hz=resample_Hz)
		elif type == 'hr':
			pass
	elif wearable == 'emp':
		raw = read_e4_to_raw_list(filepath)
		if type == 'acc':
			raw = raw[4]
			raw_get_integrated_acc_signal(raw, ch_name_acc_x="acc_x", ch_name_acc_y="acc_y", ch_name_acc_z="acc_z", integrated_ch_name=ch_name_signal, resample_Hz=resample_Hz)
		elif type == 'hr':
			raw = raw[0]
	elif wearable == 'apl':
		if type == 'acc':
			pass
		elif type == 'hr':
			TypeError('activPAL does not have heart rate signal')
	if raw is not None:
		raw.pick_channels([ch_name_signal])
		if resample_Hz is not None:
			raw = raw.resample(resample_Hz)
	if raw is not None:
		return raw.get_data(picks=[ch_name_signal])[0]
	else:
		return None



# =============================================================================
# 
# =============================================================================
def sync_wearables(parentdirpath, filepath_csv_in, filepath_csv_out, device='all'):
	df_csv_in = pandas.read_csv(filepath_csv_in, quoting=csv.QUOTE_NONNUMERIC, parse_dates = ['rec_start_datetime', 'rec_stop_datetime'])
	df_csv_in['rec_duration_datetime'] = pandas.to_timedelta(df_csv_in['rec_duration_datetime'])
	df_csv_in.reset_index()  # make sure indexes pair with number of rows

	grouped_by_subject = df_csv_in.groupby('subject_path_id', sort=False)
	signal_types_valid = ["zmx_all", 'ACC.csv']
	first_rows_written = False
	for subject_path_id, df_by_subject_path_id in grouped_by_subject:
		df_by_subject_path_id_filtered = df_by_subject_path_id.loc[df_by_subject_path_id['signal'].isin(signal_types_valid)]


		# create empty columns to fill potentially with sync infos
		df_by_subject_path_id_filtered['rec_start_datetime_reference'] = numpy.datetime64("NaT")
		df_by_subject_path_id_filtered['rec_stop_datetime_reference'] = numpy.datetime64("NaT")
		df_by_subject_path_id_filtered['rec_duration_datetime_reference'] = numpy.datetime64("NaT")
		df_by_subject_path_id_filtered['sampling_rate_max_Hz_reference_adaption'] = numpy.NAN


		#['rec_start_datetime_reference', 'rec_stop_datetime_reference', 'rec_duration_datetime_reference', 'sampling_rate_max_Hz_reference_adaption']
		list_datetimes_paired = []
		for row in df_by_subject_path_id_filtered.iterrows():
			list_datetimes_paired.append([row[1]['rec_start_datetime'], row[1]['rec_stop_datetime']])

		list_indices_paired_offsetTo2nd_overlap = sync_reach(list_datetimes_paired)
		for i_i2_p_o_o in list_indices_paired_offsetTo2nd_overlap:
			i = i_i2_p_o_o[0]
			i2 = i_i2_p_o_o[1]
			device_wearable_i = df_by_subject_path_id_filtered.device_wearable.iloc[i]
			device_wearable_i2 = df_by_subject_path_id_filtered.device_wearable.iloc[i2]
			i_sync = None
			i_ref = None

			if device_wearable_i == device_wearable_i2:
				continue
			elif device_wearable_i in ['emp']:
				i_sync = i
				i_ref = i2
			elif device_wearable_i in ['zmax']:
				i_sync = i2
				i_ref = i

			device_wearable_sync = df_by_subject_path_id_filtered.device_wearable.iloc[i_sync]
			device_wearable_ref = df_by_subject_path_id_filtered.device_wearable.iloc[i_ref]
			fsample = 2
			signal_sync = get_signal(filepath=parentdirpath + os.sep + df_by_subject_path_id_filtered.filepath.iloc[i_sync], wearable=device_wearable_sync, type='acc', resample_Hz=fsample)
			signal_ref = get_signal(filepath=parentdirpath + os.sep + df_by_subject_path_id_filtered.filepath.iloc[i_ref], wearable=device_wearable_ref, type='acc', resample_Hz=fsample)
			lag_seconds, dilation, lag_after_dilation_seconds, sample_rate_adaptation_factor = sync_signals(signal_ref, signal_sync, fsample)

			rec_start_datetime_reference_i_sync = df_by_subject_path_id_filtered.rec_start_datetime.iloc[i_sync] + datetime.timedelta(seconds=lag_seconds)
			rec_stop_datetime_reference_i_sync = df_by_subject_path_id_filtered.rec_stop_datetime.iloc[i_sync] + datetime.timedelta(seconds=lag_seconds)
			sampling_rate_max_Hz_reference_adaption_i_sync = df_by_subject_path_id_filtered.sampling_rate_max_Hz_reference_adaption.iloc[i_ref] * sample_rate_adaptation_factor

			rec_start_datetime_reference_i_ref = df_by_subject_path_id_filtered.rec_start_datetime.iloc[i_ref]
			rec_stop_datetime_reference_i_ref = df_by_subject_path_id_filtered.rec_stop_datetime.iloc[i_ref]
			sampling_rate_max_Hz_reference_adaption_i_ref = df_by_subject_path_id_filtered.sampling_rate_max_Hz_reference_adaption.iloc[i_ref]

			df_by_subject_path_id_filtered.loc[i_sync, 'rec_start_datetime_reference'] = rec_start_datetime_reference_i_sync
			df_by_subject_path_id_filtered.loc[i_sync, 'rec_stop_datetime_reference'] = rec_stop_datetime_reference_i_sync
			df_by_subject_path_id_filtered.loc[i_sync, 'rec_duration_datetime_reference'] = rec_stop_datetime_reference_i_sync - rec_start_datetime_reference_i_sync
			df_by_subject_path_id_filtered.loc[i_sync, 'sampling_rate_max_Hz_reference_adaption'] = sampling_rate_max_Hz_reference_adaption_i_sync

			df_by_subject_path_id_filtered.loc[i_ref, 'rec_start_datetime_reference'] = rec_start_datetime_reference_i_ref
			df_by_subject_path_id_filtered.loc[i_ref, 'rec_stop_datetime_reference'] = rec_stop_datetime_reference_i_ref
			df_by_subject_path_id_filtered.loc[i_ref, 'rec_duration_datetime_reference'] = rec_stop_datetime_reference_i_ref - rec_start_datetime_reference_i_ref
			df_by_subject_path_id_filtered.loc[i_ref, 'sampling_rate_max_Hz_reference_adaption'] = sampling_rate_max_Hz_reference_adaption_i_ref

		#write out in chunks of rows
		if first_rows_written:
			df_by_subject_path_id_filtered.to_csv(filepath_csv_out, mode='a', index=False, header=False, quoting=csv.QUOTE_NONNUMERIC)
		else:
			df_by_subject_path_id_filtered.to_csv(filepath_csv_out, mode='a', index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)

# =============================================================================
#  Feature extraction
# =============================================================================
def chunk_signal(signal, sfreq, window):
	# window size in samples
	n_chunks = round(len(signal) / (window * 60 * sfreq))

	# look into chunks
	signal_list = list()
	j = 0
	for i in range(0, n_chunks):
		signal_chunk = signal[j:int(j + int(window * 60 * sfreq))]
		j = j + int(window * 60 * sfreq)
		# if its the last window and we rounded down
		if j < len(signal) and i == n_chunks - 1:
			signal_chunk = signal[j:]
		signal_list.append(signal_chunk)
	return signal_list


def features_eda_from_raw(raw, channel_name, features=['tonic', 'phasic'], window=10, relative_to_app=False,
						  app_file=None, app_datetime=None, app_window='before'):
	"""
	Parameters
	-----------
	raw: mne.raw
		MNE formatted data
	channel_name: str
		Name of channel containing EDA/GSR data
	features: arr
		Array with requested EDA features to extract
	window: int
		Window to chunk signal into, in minutes.
	relative_to_app: bool
		If feature extraction is to be carried out in windows around app data
	app_window: str
		Window to select around app, can be 'before' 'after' or 'around'
	"""

	# convert to data frame with time index
	try:
		eda = raw.to_data_frame(index='time', time_format='datetime')
	except:
		eda = raw.to_data_frame(index='time')
	# get sampling frequency
	sfreq = raw.info['sfreq']

	# set time in case its present in file
	if 'timestamp_ux' in eda:
		eda.timestamp_ux = pandas.to_datetime(eda['timestamp_ux'], exact=True, utc=True)
		eda = eda.set_index(eda.timestamp_ux, drop=True)

	# decide on the window
	if not relative_to_app:

		# generate chunks and lists to append to
		eda_chunks_list = chunk_signal(eda[channel_name], sfreq, window)
		time_chunks_list = chunk_signal(eda.index, sfreq, window)
		eda_features = list()
		eda_features_labs = list()
		eda_features_times = list()

		for i, eda_chunk in enumerate(eda_chunks_list):

			# convert to pyphysio evenly signal
			eda_signal = ph.EvenlySignal(eda_chunk, sampling_freq=sfreq, signal_type='EDA')
			chunk_start = time_chunks_list[i][0]

			# resample and denoise
			eda_signal = eda_signal.resample(fout=8, kind='cubic')
			eda_despiked = ph.Filters.RemoveSpikes()(eda_signal)
			eda_denoised = ph.DenoiseEDA(0.02)(eda_despiked)

			# estimate drivers and determine tonic and phasic components
			eda_driver = ph.DriverEstim()(eda_denoised)
			phasic, tonic, _ = ph.PhasicEstim(delta=0.02)(eda_driver)

			# get features
			if 'tonic' in features:
				# get features
				feat_ton_mean = ph.TimeDomain.Mean(delta=0.02)(tonic)
				feat_ton_sd = ph.TimeDomain.StDev(delta=0.02)(tonic)
				feat_ton_range = ph.TimeDomain.Range(delta=0.02)(tonic)
				# append to lists
				eda_features_labs.extend(['eda_tonic_mean', 'eda_tonic_sd', 'eda_tonic_range'])
				eda_features_times.extend([chunk_start] * 3)
				eda_features.extend([feat_ton_mean, feat_ton_sd, feat_ton_range])
			if 'phasic' in features:
				# Phasic components
				feat_pha_mean = ph.TimeDomain.Mean(delta=0.02)(phasic)
				feat_pha_sd = ph.TimeDomain.StDev(delta=0.02)(phasic)
				feat_pha_range = ph.TimeDomain.Range(delta=0.02)(phasic)
				# append to list
				eda_features_labs.extend(['eda_phasic_mean', 'eda_phasic_sd', 'eda_phasic_range'])
				eda_features_times.extend([chunk_start] * 3)
				eda_features.extend([feat_pha_mean, feat_pha_sd, feat_pha_range])
				# Phasic Peaks
				feat_pha_mag = ph.PeaksDescription.PeaksMean(delta=0.02, win_pre=1, win_post=8)(phasic)
				feat_pha_dur = ph.PeaksDescription.DurationMean(delta=0.02, win_pre=1, win_post=8)(phasic)
				feat_pha_num = ph.PeaksDescription.PeaksNum(delta=0.02)(phasic)
				feat_pha_auc = ph.TimeDomain.AUC(delta=0.02)(phasic)
				# append to list
				eda_features_labs.extend(
					['eda_phasic_magnitude', 'eda_phasic_duration', 'eda_phasic_number', 'eda_phasic_auc'])
				eda_features_times.extend([chunk_start] * 4)
				eda_features.extend([feat_pha_mag, feat_pha_dur, feat_pha_num, feat_pha_auc])

		# convert to df for output
		eda_df = {'time': eda_features_times, 'feature': eda_features_labs, '': eda_features}
		eda_df = pandas.DataFrame(eda_df)
		eda_df = eda_df.pivot(index='time', columns='feature')

	# TODO: Add Relative to app
	return eda_df
"""
	comment
	TODO:
	# delete and correct after no good duration (less than 10 min) or voltage too low at end of recording
	# create annotation like what is wrong with the recording
	# check the dates if this is a standard date and if the order needs to be adapted.
	# PPG to HR signal
	# cross correlation, on 10 min snippets with linear extrapolation.
"""
if __name__ == "__main__":

	# Instantiate the argument parser
	parser = argparse.ArgumentParser(prog='wearanize', description='this is wearanize software')

	# Required positional argument
	parser.add_argument('function', type=str,
					help="either 'reexport' to create more concise formats for the devices, or 'init' to initialize the infos, or 'request' to request data with additional arguments")


	# Optional argument
	parser.add_argument('--path_data', type=dir_path,
					help='An optional path to the parent folder where the data is stored and to be initialized (default is current directory)')

	# Optional argument
	parser.add_argument('--path_init', type=dir_path_create,
					help='An optional path to the parent folder where the data is stored and to be initialized (default is current directory)')

	# Optional argument
	parser.add_argument('--path_output', type=dir_path_create,
					help="An optional path to the folder where the requested output will be stored (default is in the init dirctory subfolder 'output')")

	# Optional argument
	parser.add_argument('--devices', type=str,
					help="An optional arguement to specify all the devices default is --devices='all' which is equivalent to --devices='zmx|emp|apl|app'")

	# Switch
	parser.add_argument('--do_file_logging', action='store_true',
					help='switch to indicate if stdout and stderr are redirected to a logging file')

	# Switch
	parser.add_argument('--zmax_ppgparser', action='store_true',
					help='Switch to indicate if ZMax PPGParser.exe is used to reparse some heart rate related channels')

	# Optional argument
	parser.add_argument('--zmax_ppgparser_exe_path', type=file_path,
					help='direct and full path to the ZMax PPGParser.exe in the Hypnodyne ZMax software folder')

	# Optional argument
	parser.add_argument('--zmax_ppgparser_timeout_seconds', type=float,
					help='An optional timeout to run the ZMax PPGParser.exe in seconds. If empty no timeout is used')


	"""
	TODO DELETE:
	# Required positional argument
	parser.add_argument('pos_arg', type=int,
					help='A required integer positional argument')

	# Optional positional argument
	parser.add_argument('opt_pos_arg', type=int, nargs='?',
					help='An optional integer positional argument')

	# Optional argument
	parser.add_argument('--opt_arg', type=int,
					help='An optional integer argument')

	# Switch
	parser.add_argument('--switch', action='store_true',
					help='A boolean switch')
	"""

	args = parser.parse_args()


	"""
	TODO DELETE
	print("Argument values:")
	print(args.pos_arg)
	print(args.opt_pos_arg)
	print(args.opt_arg)
	print(args.switch)
	"""

	path_data = os.getcwd() # the current working directory
	if args.path_data is not None:
		path_data = args.path_data

	path_init = os.getcwd() + os.sep + '.wearanize'
	if args.path_init is not None:
		path_init = args.path_init

	path_output = path_init + os.sep + 'output'
	if args.path_output is not None:
		path_output = args.path_output

	devices = 'all'
	if args.devices is not None:
		devices = args.devices

	do_file_logging = False
	if args.do_file_logging is not None:
		do_file_logging = args.do_file_logging

	zmax_ppgparser = True
	if args.zmax_ppgparser is not None:
		zmax_ppgparser = args.zmax_ppgparser

	zmax_ppgparser_exe_path = 'PPGParser.exe' # in the current working directory
	if args.zmax_ppgparser_exe_path is not None:
		zmax_ppgparser_exe_path = args.zmax_ppgparser_exe_path

	zmax_ppgparser_timeout_seconds = 1000 # in the current working directory
	if args.zmax_ppgparser_timeout_seconds is not None:
		zmax_ppgparser_timeout_seconds = args.zmax_ppgparser_timeout_seconds

	# logging redirect #
	if do_file_logging:
		stdout_logger = logging.getLogger('STDOUT')
		sl = StreamToLogger(stdout_logger, logging.INFO)
		sys.stdout = sl

		stderr_logger = logging.getLogger('STDERR')
		sl = StreamToLogger(stderr_logger, logging.ERROR)
		sys.stderr = sl

	if args.function == 'test':
		#--tests--#
		#raw_reread = read_edf_to_raw_zipped("Y:/HB/data/test_data_zmax/FW_merged.zip", format="edf")

		dt1_start = datetime.datetime(2000, 1, 1, 0,0,0)
		dt1_stop = dt1_start + datetime.timedelta(hours=6)
		dt2_start = datetime.datetime(2000, 1, 1, 6,0,0)
		dt2_stop = dt2_start + datetime.timedelta(hours=6)
		dt3_start = datetime.datetime(2000, 1, 1, 7,0,0)
		dt3_stop = dt3_start + datetime.timedelta(hours=6)
		dt4_start = datetime.datetime(2000, 1, 1, 7,0,1)
		dt4_stop = dt4_start + datetime.timedelta(hours=6)
		dt5_start = datetime.datetime(1999, 12, 31, 23, 59,  59)
		dt5_stop = dt5_start + datetime.timedelta(hours=6)
		list_datetimes_paired = [[dt1_start, dt1_stop], [dt2_start, dt2_stop], [dt3_start, dt3_stop], [dt4_start, dt4_stop], [dt5_start, dt5_stop]]
		print(sync_reach(list_datetimes_paired, min_reach_duration_seconds=3600))

		#raw = read_edf_to_raw_zipped("Y:/HB/data/test_data_zmax/FW.zip", format="zmax_edf")
		#write_raw_to_edf(raw, "Y:/HB/data/test_data_zmax/FW.merged.edf", format="zmax_edf")  # treat as a speacial zmax read EDF for export
		#write_raw_to_edf_zipped(raw, "Y:/HB/data/test_data_zmax/FW.merged.zip", format="zmax_edf") # treat as a speacial zmax read EDF for export
		#raw_reread = read_edf_to_raw_zipped("Y:/HB/data/test_data_zmax/FW_merged.zip", format="edf")
		#write_raw_to_edf_zipped(raw, "Y:/HB/data/test_data_zmax/FW.merged.reread.zip", format="zmax_edf") # treat as a speacial zmax read EDF for export

		#signal_integrated_acc = raw_append_integrate_acc(raw, 'dX', 'dX', 'dZ', integrated_ch_name="integrated_acc").get_data(picks=["integrated_acc"])

		#delay_ref = 16
		#signal_ref = raw.get_data(picks=['EEG L'],start=0+delay_ref, stop=256*60*10*3+delay_ref)[0,]
		#signal_sync = raw.get_data(picks=['EEG R'],start=0, stop=256*60*10*3)[0,]

		#lag_seconds, dilation, lag_after_dilation_seconds, sample_rate_adaptation_factor = sync_signals(signal_ref, signal_sync, 256, chunk_size_seconds=60*10, chunk_step_seconds=60*5, lag_merge_window_seconds=60*20, max_merge_lag_difference_seconds=0.5, threshold_chunk_min_match=2, allow_anti_correlation=False)
		#print(lag_seconds)


		fsample = 2
		signal_ref = get_signal(filepath='Y:\\HB\\data\\example_subjects\\data\\sub-HB0037923118974\\pre-1\\wrb\\sub-HB1EM8057291_pre-1_wrb_emp_02.zip', wearable='emp', type='acc', resample_Hz=fsample)
		signal_sync = get_signal(filepath='Y:\\HB\\data\\example_subjects\\data\\sub-HB0037923118974\\pre-1\\wrb\\sub-HB1ZM3321037_pre-1_wrb_zmx_1.zip', wearable='zmx', type='acc', resample_Hz=fsample)

		import matplotlib.pyplot as plt
		plt.rcParams["font.family"] = 'Cambria'
		plt.style.use('ggplot')
		fig, axs = plt.subplots(2, sharex=True)
		fig.suptitle('signal_sync (above) and signal_ref (below)')

		delay_samples = int(57954/fsample)
		axs[0].plot(list(range(delay_samples,delay_samples+len(signal_sync),1)),signal_sync, 'blue')
		#axs[0].grid(b=True, which='both', axis='both', color='lightgrey', markevery=5)
		#axs[0].set(xlabel=' ', ylabel='Heart Rate')
		#axs[0].set_ylim(ymin=50, ymax=160)

		axs[1].plot(list(range(0,len(signal_ref),1)),signal_ref, 'red')
		#axs[1].grid(b=True, which='both', axis='both', color='lightgrey', markevery=5)
		#axs[1].set(xlabel=' ', ylabel='IBI (ms)')
		plt.show()

		lag_seconds, dilation, lag_after_dilation_seconds, sample_rate_adaptation_factor = sync_signals(signal_ref, signal_sync, fsample, chunk_size_seconds=60*120, chunk_step_seconds=60*5, lag_merge_window_seconds=60*20, max_merge_lag_difference_seconds=60*5, threshold_chunk_min_match=1, allow_anti_correlation=False)
		print(lag_seconds)

		"""

		#raw.plot(duration=30)

		raw_ppg = raw.copy()
		raw_ppg.pick_channels(['OXY_IR_AC'], ordered=False)
		# only reserve the channel in additional memory
		raw_ppg.rename_channels({"OXY_IR_AC": "PPG"}) #in place modification
		raw_ppg = raw_ppg.resample(100)
		print("filtering:")
		raw_ppg.filter(l_freq=0.5, h_freq=4.0)

		print("hr detection:")
		ppg_signal = raw_ppg.get_data(units=raw_ppg._orig_units, picks=['PPG'])[0]
		#ppg_signal = heartpy.enhance_peaks(ppg_signal, iterations=2)
		wd, m = heartpy.process(hrdata=ppg_signal, sample_rate = raw_ppg.info['sfreq'], windowsize=0.75, report_time=False, calc_freq=False, freq_method='welch', welch_wsize=240, freq_square=False, interp_clipping=False, clipping_scale=False, interp_threshold=1020, hampel_correct=True, bpmmin=30, bpmmax=240, reject_segmentwise=False, high_precision=False, high_precision_fs=1000.0, breathing_method='welch', clean_rr=False, clean_rr_method='quotient-filter', measures=None, working_data=None)

		#set large figure
		matplotlib.pyplot.figure(figsize=(12,4))

		#call plotter
		heartpy.plotter(wd, m)

		#display measures computed
		for measure in m.keys():
			print('%s: %f' %(measure, m[measure]))


		print("writing edf data full")
		write_raw_to_edf(raw, "Y:/HB/data/test_data_zmax/FW.reexport.edf")

		print("writing edf data PPG")
		write_raw_to_edf(raw_ppg, "Y:/HB/data/test_data_zmax/FW.reexport.PPG.edf")

		#raw_ppg.plot(duration=30)

		raw2 = read_edf_to_raw("Y:/HB/data/test_data_zmax/DAVOS_1002_(1).edf")
		raw2.plot(duration=30)
		write_raw_to_edf(raw2, "Y:/HB/data/test_data_zmax/DAVOS_1002_(1).reexport.edf")
		"""

		wearable_file_structure_annotation_csv = "wearabout_0.csv"
		wearable_file_structure_annotation_datetime_csv = "wearabout_1_annotation.csv"

		parse_wearable_data_write_csv(parentdirpath=path_data,filepath_csv_out=wearable_file_structure_annotation_csv,device='zmax')
		parse_wearable_data_with_csv_annotate_datetimes(parentdirpath=path_data,filepath_csv_in=wearable_file_structure_annotation_csv,filepath_csv_out=wearable_file_structure_annotation_datetime_csv,device='zmax')

		df = pandas.read_csv(wearable_file_structure_annotation_datetime_csv)
		#df.iloc[[0]]

		print("tests finished")

	elif args.function == 'reexport':
		devices = parseDevice(devices)
		if 'zmx' in devices:
			exec_string = "\"zmax_edf_merge_converter.exe\"" + " " + "\"" + path_data + "\"" + " --no_overwrite --temp_file_postfix=\"_TEMP_\" --zipfile_match_string=\"_wrb_zmx_\" --zipfile_nonmatch_string=\"" + FILE_EXTENSION_WEARABLE_ZMAX_REEXPORT + "|_raw| - empty|_TEMP_\" --write_name_postfix=\"" + FILE_EXTENSION_WEARABLE_ZMAX_REEXPORT + "\" --exclude_empty_channels --zmax_lite --read_zip --write_zip" +  ("--zmax_ppgparser" if zmax_ppgparser else "")  + " --zmax_ppgparser_exe_path=\"" + zmax_ppgparser_exe_path +  "\" --zmax_ppgparser_timeout_seconds=" + str(zmax_ppgparser_timeout_seconds)
			try:
				subprocess.run(exec_string, shell=False, timeout=zmax_ppgparser_timeout_seconds)
			except:
				print(traceback.format_exc())
				print('FAILED TO REEXPORT with command: ' + exec_string)


	elif args.function == 'init':

		wearable_file_structure_annotation_csv = "wearabout_0.csv"
		wearable_file_structure_annotation_datetime_csv = "wearabout_1_annotation.csv"
		wearable_file_structure_annotation_datetime_sync_csv = "wearabout_2_annotation_sync.csv"

		#parse_wearable_data_write_csv(parentdirpath=path_data,filepath_csv_out=wearable_file_structure_annotation_csv,device=devices)
		#parse_wearable_data_with_csv_annotate_datetimes(parentdirpath=path_data,filepath_csv_in=wearable_file_structure_annotation_csv,filepath_csv_out=wearable_file_structure_annotation_datetime_csv, device=devices)
		sync_wearables(parentdirpath=path_data, filepath_csv_in=wearable_file_structure_annotation_datetime_csv, filepath_csv_out=wearable_file_structure_annotation_datetime_sync_csv, device=devices)

		print("init finished")
	elif args.function == 'request':

		print("request finished")
	else:
		parser.error("function unknown")

