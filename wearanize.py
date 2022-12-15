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
import statsmodels.api as sm
import pytz
import errno
import pyphysio as ph
import heartpy as hp
import rapidhrv as rhv
import sys

import apl_converter

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
import xlrd 
from math import log, e
import gc

# Import from embeded functions
from zmax_edf_merge_converter import file_path, dir_path, dir_path_create, fileparts, zip_directory, safe_zip_dir_extract, safe_zip_dir_cleanup, raw_prolong_constant, read_edf_to_raw, edfWriteAnnotation, write_raw_to_edf, read_edf_to_raw_zipped, write_raw_to_edf_zipped, raw_zmax_data_quality
from e4_converter import read_e4_to_raw_list, read_e4_to_raw, read_e4_raw_to_df, e4_concatenate, e4_concatenate_par, read_e4_concat_to_raw
import apl_converter as apl
from apl_converter import read_apl_to_raw, apl_window_to_raw, read_apl_event_to_raw

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
def get_raw_by_date_and_time(filepath,  datetime_ts, duration_seconds,  wearable='zmx', fetch=['apl', 'emp'], resample_hz=None, offset_seconds=0.0):
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
	elif wearable == 'apl':
		raw = read_apl_to_raw(filepath)
		pass
	 
	# convert to dateframe and subset time window    
	raw_df = raw.to_data_frame(time_format='datetime')
	raw_df = raw_df[(raw_df.time > start_date) & (raw_df.time < end_date)]   
	raw_df = raw_df.set_index('time', drop=True)
	
	#list all files and search all the relevant files that fall within these time limits
	print("Searching all wearable files in directory...")
	for wearables in fetch:
		# Skip if we have the same modality
		if wearables != wearable:
			
			# check all files in directory to match window
			if wearables == 'zmx':
				file_list = find_wearable_files(sub_path, wearable="zmax")
				file_list = [x for x in file_list if ('merge') in x]
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
				raw_df = raw_df.merge(channel_df, how='left', left_index=True, right_index=True)
				
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
								channel_df = pandas.concat([channel_df, raw_channel], sort=False)
						except:
							pass
				raw_df = raw_df.merge(channel_df, left_index=True, right_index=True)
				
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


def merge_emp_to_apl_raw(path_to_week):
	"""
	Parameters
	----------
	path_to_week: str
	String containing path to directory containig week info
	Returns
	-------
	mne_obj: mne.RawArray
	An mne raw object combining activpal and empatica data based on time stamps
	"""

	# files
	emp_file = glob.glob(path_to_week + os.sep + "wrb/*emp_full.zip")[0]
	apl_file = glob.glob(path_to_week + os.sep + "wrb/*_events.csv")[0]

	# e4 to raw
	emp_raw = read_e4_to_raw(emp_file)
	emp_df = emp_raw.to_data_frame(time_format='datetime')
	emp_df.set_index('time', inplace=True, drop=True)

	# apl event to raw
	apl_raw = apl_converter.read_apl_event_to_raw(apl_file, resample_Hz=64)
	apl_df = apl_raw.to_data_frame(time_format='datetime')
	apl_df.set_index('time', inplace=True, drop=True)

	# merge and convert to mne raw
	merge_df = apl_df.merge(emp_df, left_index = True, right_index =True, how='outer')
	# mne info
	mne_start = merge_df.index[0]
	mne_head = list(merge_df.columns)
	mne_np = merge_df.to_numpy().transpose()
	# convert to raw
	mne_info = mne.create_info(ch_names=mne_head, sfreq=64, ch_types="misc")
	mne_obj = mne.io.RawArray(mne_np, mne_info, first_samp=0)

	# set start time and return an mne object
	mne_obj = mne_obj.set_meas_date(mne_start.timestamp())

	return mne_obj


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
#  Chuncking
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

def chunk_signal_at_app(signal, channel_name, app_data, app_starttime, app_endtime, app_window='before', window=10):
	"""
	Parameters
	----------
	signal: pandas.DataFrame()
		A raw signal converted to a pandas dataframe with a datetime index
	app_data: pandas.DataFrame() or str
		Pandas data frame or path to EMA data for subject
	app_starttime: str
		Name of column containing beep start time
	app_endtime: str
		Name of column containing beep end time
	app_window: str
		Window around EMA to search. Can be 'before', 'after', or 'around'
	window: int
		Window length
	Returns
	-------
	signal_chuncks: list
		list containing signal chuncks as pandas data frane
	time_chunks: list
		list containing EMA start time used for chunking
	"""
	# if path instead of dataframe supplied
	if type(app_data) == str:
		app_data = app_to_long(app_data)

	# reformat dates
	app_data[app_starttime] = pandas.to_datetime(app_data[app_starttime], exact=True)
	app_data[app_endtime] = pandas.to_datetime(app_data[app_endtime], exact=True)

	# loop over time stamps and create signal chunks
	signal_chunks = list()
	time_chunks = list()
	for i in range(len(app_data[app_starttime])):

		# decide on window
		start_time = app_data[app_starttime][i]
		end_time = app_data[app_endtime][i]
		if app_window == 'before':
			t1 = start_time - datetime.timedelta(minutes=window)
			t2 = start_time
		elif app_window == 'after':
			t1 = datetime.timedelta(minutes=window)
			t2 = end_time + datetime.timedelta(minutes=window)
		elif app_window == 'around':
			t1 = start_time - datetime.timedelta(minutes=window)
			t2 = end_time + datetime.timedelta(minutes=window)

		# convert to datetimes and set to UTC tz
		t1 = t1.to_pydatetime()
		t1 = t1.astimezone(pytz.timezone('Europe/Amsterdam'))
		t1 = t1.astimezone(pytz.utc)
		t2 = t2.to_pydatetime()
		t2 = t2.astimezone(pytz.timezone('Europe/Amsterdam'))
		t2 = t2.astimezone(pytz.utc)

		# subset temp dataframe from window and add to a list
		signal_temp = signal[(signal['time'] > t1) & (signal['time'] < t2)]
		if channel_name != 'APActivity_code':
			signal_temp = signal_temp[channel_name]
		signal_chunks.extend([signal_temp])
		time_chunks.extend([start_time])

	return signal_chunks, time_chunks

# =============================================================================
#  Feature Extraction
# =============================================================================

def features_eda_from_raw(raw, channel_name, device='emp', window=10, features=['tonic', 'phasic'], delta=0.02, app_data=None, app_starttime=None,
						  app_endtime=None, app_window='before'):
	# convert to data frame with time index
	if device == 'emp':
		eda, sfreq, sfreq_ms = read_e4_raw_to_df(raw)
	else:
		signal = raw.to_data_frame(time_format='datetime')
		sfreq = raw.info['sfreq']

	# create chunks depending on window
	if app_data == None:
		eda_chunks_list = chunk_signal(eda[channel_name], sfreq, window)
		time_chunks_list = chunk_signal(eda.index, sfreq, window)
	else:
		eda_chunks_list, time_chunks_list = chunk_signal_at_app(signal=eda, channel_name=channel_name, app_data=app_data, app_starttime=app_starttime, app_endtime=app_endtime, app_window=app_window, window=window)

	# initialize lists for features
	eda_features = list()
	eda_features_labs = list()
	eda_features_times = list()

	for i, eda_chunk in enumerate(eda_chunks_list):

		# log sample start time
		if app_data == None:
			chunk_start = time_chunks_list[i][0]
		else:
			chunk_start = time_chunks_list[i]

		# convert to pyphysio evenly signal
		eda_signal = ph.EvenlySignal(eda_chunk, sampling_freq=sfreq, signal_type='EDA')
		if (len(eda_signal) > ((window*60*sfreq)/2)) & (numpy.isnan(numpy.mean(eda_signal)) == False) & (numpy.mean(eda_signal) >= 0.02):
			
			# resample and filter according to Tronstad (2015) and Foll(2021)
			eda_signal = eda_signal.resample(fout=8, kind='linear')
			eda_filtered = ph.KalmanFilter(R=3, ratio=2)(eda_signal)
			# estimate drivers and determine tonic and phasic components (Foll 2021)
			eda_driver = ph.DriverEstim()(eda_filtered)
			phasic, tonic, _ = ph.PhasicEstim(0.01, win_pre=3, win_post=8)(eda_driver)

			# get features
			if 'tonic' in features:
				# get features
				feat_ton_mean = ph.TimeDomain.Mean()(tonic)
				feat_ton_sd = ph.TimeDomain.StDev()(tonic)
				feat_ton_range = ph.TimeDomain.Range()(tonic)
				# append to lists
				eda_features_labs.extend(['eda_tonic_mean', 'eda_tonic_sd', 'eda_tonic_range'])
				eda_features_times.extend([chunk_start] * 3)
				eda_features.extend([feat_ton_mean, feat_ton_sd, feat_ton_range])
			if 'phasic' in features:
				# Phasic components
				feat_pha_mean = ph.TimeDomain.Mean()(phasic)
				feat_pha_sd = ph.TimeDomain.StDev()(phasic)
				feat_pha_range = ph.TimeDomain.Range()(phasic)
				# append to list
				eda_features_labs.extend(['eda_phasic_mean', 'eda_phasic_sd', 'eda_phasic_range'])
				eda_features_times.extend([chunk_start] * 3)
				eda_features.extend([feat_pha_mean, feat_pha_sd, feat_pha_range])
				# Phasic Peaks
				feat_pha_mag = ph.PeaksDescription.PeaksMean(delta, win_pre=3, win_post=8)(phasic)
				feat_pha_dur = ph.PeaksDescription.DurationMean(delta, win_pre=3, win_post=8)(phasic)
				feat_pha_num = ph.PeaksDescription.PeaksNum(delta)(phasic)
				feat_pha_auc = ph.TimeDomain.AUC()(phasic)
				# append to list
				eda_features_labs.extend(['eda_phasic_magnitude', 'eda_phasic_duration', 'eda_phasic_number', 'eda_phasic_auc'])
				eda_features_times.extend([chunk_start] * 4)
				eda_features.extend([feat_pha_mag, feat_pha_dur, feat_pha_num, feat_pha_auc])

			# Calculate quality metrics
			## slope
			feat_eda_slope = eda_chunk.resample('1S').ffill()
			feat_eda_slope = feat_eda_slope.reset_index()
			y = numpy.array(feat_eda_slope['eda'].fillna(method='bfill').values, dtype=float)
			x = numpy.array(pandas.to_datetime(feat_eda_slope['time'].dropna()).index.values, dtype=float)
			feat_eda_slope = scipy.stats.linregress(x, y)[0]
			## Range (min - max)
			feat_eda_max = ph.TimeDomain.Max()(eda_signal)
			feat_eda_min = ph.TimeDomain.Min()(eda_signal)
			# append to list
			eda_features_labs.extend(['eda_qa_slope', 'eda_qa_min', 'eda_qa_max', 'eda_window'])
			eda_features_times.extend([chunk_start] * 4)
			eda_features.extend([feat_eda_slope, feat_eda_min, feat_eda_max, window])

		else: # log that file was empty or poor quality
			eda_features_labs.extend(['eda_tonic_mean', 'eda_tonic_sd', 'eda_tonic_range', 'eda_phasic_mean', 'eda_phasic_sd', 'eda_phasic_range', 'eda_phasic_magnitude', 'eda_phasic_duration', 'eda_phasic_number', 'eda_phasic_auc', 'eda_qa_slope', 'eda_qa_min', 'eda_qa_max', 'eda_window'])
			eda_features_times.extend([chunk_start] * 14)
			if (len(eda_signal) < ((window*60*sfreq)/2)) or (numpy.mean(eda_signal) < 0.02):
				eda_features.extend(['qa_fail']*14)
			else:
				eda_features.extend(['nan'] * 14)

		# convert to df for output
		eda_df = {'time': eda_features_times, 'feature': eda_features_labs, 'eda': eda_features}
		eda_df = pandas.DataFrame(eda_df)
		eda_df = eda_df.pivot(index='time', columns='feature', values='eda')

	return eda_df



def features_hr_from_raw(raw, channel_name, device='emp', window=10, app_data=None, app_starttime=None, app_endtime=None, app_window='before'):

	# convert to data frame with time index
	if device == 'emp':
		hr, sfreq, sfreq_ms = read_e4_raw_to_df(raw)
	else:
		signal = raw.to_data_frame(time_format='datetime')
		sfreq = raw.info['sfreq']

	# create chunks depending on window
	if app_data == None:
		hr_chunks_list = chunk_signal(hr[channel_name], sfreq, window)
		time_chunks_list = chunk_signal(hr.index, sfreq, window)
	else:
		hr_chunks_list, time_chunks_list = chunk_signal_at_app(signal=hr, channel_name=channel_name, app_data=app_data, app_starttime=app_starttime, app_endtime=app_endtime, app_window=app_window, window=window)

	# initialize lists for features
	hr_features = list()
	hr_features_labs = list()
	hr_features_times = list()
	for i, hr_chunk in enumerate(hr_chunks_list):
		# TODO: Add Quality assessments
		# log sample start time
		if app_data == None:
			chunk_start = time_chunks_list[i][0]
		else:
			chunk_start = time_chunks_list[i]
		try:
			# turn into hrv signal class
			signal = rhv.Signal(hr_chunk.to_numpy(), sample_rate=int(sfreq))
			# high pass filter
			preprocessed = rhv.preprocess(signal, highpass_cutoff=0.06, lowpass_cutoff=8, sg_settings=(4, 200), resample_rate=32)
			# Preprocess using a 1-second sliding time window
			analyzed = rhv.analyze(preprocessed, window_width=10,  window_overlap=9, outlier_detection_settings="moderate", amplitude_threshold=30)
			analyzed.dropna(inplace=True)
			# trim outliers and get values
			hr_trunc = scipy.stats.trim_mean(analyzed[['BPM', 'RMSSD', 'SDNN', 'SDSD', 'pNN20', 'pNN50', 'HF']], 0.05)
			hr_trunc  = numpy.append(hr_trunc, analyzed.BPM.max())
			hr_trunc  = numpy.append(hr_trunc, analyzed.BPM.min())
			hr_trunc = numpy.append(hr_trunc, window)
			hr_trunc  = hr_trunc.tolist()
			# add features to list
			hr_features.extend(hr_trunc)
			hr_features_labs.extend(['hr_bpm', 'hr_rmssd', 'hr_sdnn', 'hr_sdsd', 'hr_pnn20', 'hr_pnn50', 'hr_hf', 'hr_max', 'hr_min', 'hr_window'])
			hr_features_times.extend([chunk_start]*10)
		except:
			hr_features_labs.extend(['hr_bpm', 'hr_rmssd', 'hr_sdnn', 'hr_sdsd', 'hr_pnn20', 'hr_pnn50', 'hr_hf', 'hr_max', 'hr_min', 'hr_window'])
			hr_features_times.extend([chunk_start]*10)
			hr_features.extend(['nan'] * 10)

	hr_df = {'time':hr_features_times, 'feature':hr_features_labs, 'hr':hr_features}
	hr_df = pandas.DataFrame(hr_df)
	hr_df = hr_df.pivot(index='time', columns='feature', values='hr')

	return hr_df


def features_temp_from_raw(raw, channel_name, device='emp', window=10, app_data=None, app_starttime=None, app_endtime=None, app_window='before'):

	# convert to data frame with time index
	if device == 'emp':
		signal, sfreq, sfreq_ms = read_e4_raw_to_df(raw)
	else:
		signal = raw.to_data_frame(time_format='datetime')
		sfreq = raw.info['sfreq']

	# create chunks depending on window
	if app_data == None:
		signal_chunks_list = chunk_signal(signal[channel_name], sfreq, window)
		time_chunks_list = chunk_signal(signal.index, sfreq, window)
	else:
		signal_chunks_list, time_chunks_list = chunk_signal_at_app(signal=signal, channel_name=channel_name, app_data=app_data, app_starttime=app_starttime, app_endtime=app_endtime, app_window=app_window, window=window)

	# initialize lists for features
	temp_features = list()
	temp_labs = list()
	temp_times = list()
	for i, signal_chunk in enumerate(signal_chunks_list):
		# TODO: Add Quality assessments
		# log sample start time
		if app_data == None:
			chunk_start = time_chunks_list[i][0]
		else:
			chunk_start = time_chunks_list[i]

		# turn into pyphysio signal class
		signal_chunk_evenly = ph.EvenlySignal(signal_chunk, sfreq)
		# remove spikes
		signal_chunk_evenly = ph.Filters.RemoveSpikes()(signal_chunk_evenly)

		# estimate features
		temp_mean = ph.TimeDomain.Mean()(signal_chunk_evenly)
		temp_min = ph.TimeDomain.Min()(signal_chunk_evenly)
		temp_max = ph.TimeDomain.Max()(signal_chunk_evenly)

		## slope
		feat_slope = signal_chunk.resample('1S').ffill()
		feat_slope = feat_slope.reset_index()
		y = numpy.array(feat_slope['temp'].fillna(method='bfill').values, dtype=float)
		x = numpy.array(pandas.to_datetime(feat_slope['time'].dropna()).index.values, dtype=float)
		temp_slope = scipy.stats.linregress(x, y)[0]

		# check if acceptable range
		if (temp_max < 45) and (temp_min > 15):
			# add features to list
			temp_features.extend([temp_mean, temp_min, temp_max, temp_slope, window])
			temp_labs.extend(['temp_mean', 'temp_min', 'temp_max', 'temp_slope', 'temp_window'])
			temp_times.extend([chunk_start]*5)
		else:
			temp_features.extend(['qa_fail','qa_fail', 'qa_fail', 'qa_fail', window])
			temp_labs.extend(['temp_mean', 'temp_min', 'temp_max', 'temp_slope', 'temp_window'])
			temp_times.extend([chunk_start]*5)

	# convert to df
	temp_df = {'time':temp_times, 'feature':temp_labs, 'temp':temp_features}
	temp_df = pandas.DataFrame(temp_df)
	temp_df = temp_df.pivot(index='time', columns='feature', values='temp')

	return temp_df


def features_acc_from_raw(raw, channel_name_x, channel_name_y, channel_name_z, device='emp', window=10, app_data=None,
						  app_starttime=None, app_endtime=None, app_window='before'):
	# convert to data frame with time index
	if device == 'emp':
		signal, sfreq, sfreq_ms = read_e4_raw_to_df(raw)
	else:
		signal = raw.to_data_frame(time_format='datetime')
		sfreq = raw.info['sfreq']
	# create chunks depending on window
	if app_data == None:
		signal_chunks_x = chunk_signal(signal[channel_name_x], sfreq, window)
		signal_chunks_y = chunk_signal(signal[channel_name_y], sfreq, window)
		signal_chunks_z = chunk_signal(signal[channel_name_z], sfreq, window)
		time_chunks_list = chunk_signal(signal.index, sfreq, window)
	else:
		signal_chunks_x, time_chunks_list = chunk_signal_at_app(signal=signal, channel_name=channel_name_x,
																app_data=app_data, app_starttime=app_starttime,
																app_endtime=app_endtime, app_window=app_window,
																window=window)
		signal_chunks_y, _ = chunk_signal_at_app(signal=signal, channel_name=channel_name_y, app_data=app_data,
												 app_starttime=app_starttime, app_endtime=app_endtime,
												 app_window=app_window, window=window)
		signal_chunks_z, _ = chunk_signal_at_app(signal=signal, channel_name=channel_name_z, app_data=app_data,
												 app_starttime=app_starttime, app_endtime=app_endtime,
												 app_window=app_window, window=window)

	# initialize lists for features
	acc_features = list()
	acc_labs = list()
	acc_times = list()
	for i, _ in enumerate(signal_chunks_x):

		# log sample start time
		if app_data == None:
			chunk_start = time_chunks_list[i][0]
		else:
			chunk_start = time_chunks_list[i]

		# subset signals
		acc_x = signal_chunks_x[i].to_numpy()
		acc_y = signal_chunks_y[i].to_numpy()
		acc_z = signal_chunks_z[i].to_numpy()

		feature_prefix = ["acc_x", "acc_y", "acc_z", "acc_mag", "acc_x_deriv", "acc_y_deriv", "acc_z_deriv",
						  "acc_mag_deriv"]
		feature_list = ['mean', 'sd', 'median_abs_dev', 'minimum', 'maximum', 'energy', 'iqr', 'ar_coef', 'entropy']
		# Feature estimation based on Zhu et al (2017)
		if len(acc_x) > 0:
			# Magnitude: i.e., mean displacement
			acc_x_dis = numpy.array(abs(numpy.diff(acc_x, prepend=acc_x[0])))
			acc_y_dis = numpy.array(abs(numpy.diff(acc_y, prepend=acc_y[0])))
			acc_z_dis = numpy.array(abs(numpy.diff(acc_z, prepend=acc_z[0])))
			acc_mag = numpy.sqrt(acc_x_dis ** 2 + acc_y_dis ** 2 + acc_z_dis ** 2)

			# jerk (derivative)
			deriv_acc_x = numpy.gradient(acc_x)
			deriv_acc_y = numpy.gradient(acc_y)
			deriv_acc_z = numpy.gradient(acc_z)
			deriv_acc_mag = numpy.gradient(acc_mag)

			for num, j in enumerate(
					[acc_x_dis, acc_y_dis, acc_z_dis, acc_mag, deriv_acc_x, deriv_acc_y, deriv_acc_z, deriv_acc_mag]):
				# time domain features
				mean = j.mean()
				sd = j.std()
				median_abs_dev = scipy.stats.median_abs_deviation(j)
				minimum = j.min()
				maximum = j.max()
				# energy and IQR
				energy = numpy.sum(j * j) / j.size
				q1, q2 = numpy.percentile(j, [75, 25])
				iqr = q1 - q2
				# autoregressive coefficient
				ar_coef, _ = sm.regression.linear_model.burg(j, order=1, demean=True)
				#  estimate entropy
				n_j = len(j)
				if n_j <= 1:
					entropy = 0
				value, counts = numpy.unique(j, return_counts=True)
				probs = counts / n_j
				n_classes = numpy.count_nonzero(probs)
				if n_classes <= 1:
					entropy = 0
				else:
					entropy = 0
					# Compute entropy
					base = e
					for k in probs:
						entropy -= k * log(k, base)
				# add all features to list
				acc_features.extend([mean, sd, median_abs_dev, minimum, maximum, energy, iqr, ar_coef[0], entropy])
				feature_i = feature_prefix[num]
				acc_labs.extend([(feature_i + '_') + feat_name for feat_name in feature_list])
				acc_times.extend([chunk_start] * 9)
			SMA = acc_mag.sum() / len(acc_mag)
			acc_features.extend([SMA])
			acc_labs.extend(['acc_SMA'])
			acc_times.extend([chunk_start])
		else:
			for feat_pre in feature_prefix:
				acc_features.extend(["nan"] * 10)
				acc_labs.extend([(feat_pre + '_') + feat_name for feat_name in feature_list])
				acc_labs.extend(['acc_SMA'])
				acc_times.extend([chunk_start] * 10)
	# convert to df
	acc_df = {'time': acc_times, 'feature': acc_labs, 'acc': acc_features}
	acc_df = pandas.DataFrame(acc_df)
	acc_df.drop_duplicates(inplace=True)
	acc_df = acc_df.pivot(index='time', columns='feature', values='acc')

	return acc_df


def features_apl_from_events(apl_events, window=10, bout_duration=10, app_data=None, app_starttime=None,
							 app_endtime=None, app_window='before'):
	"""
	Estimate features from activpal events file
	Parameters
	----------
	apl_events: str or raw
		Path to acitvpal file ending with "tagged_events.csv" or an already converted raw file
	window: int
		Window (in minutes) to use for signal chunking. Features estimated from window.
	bout_duration: int
		Definition of bout duration (in minutes) to quantify if excercise bout happened.
	app_data: str or pandas.DataFrame
		Path or data frame containing app EMA  data to use for signal search
	app_starttime: str
		String indicating column containing survey start times
	app_endtime: str
		String indicating coloumn containing survery end times
	app_window: str
		Window to use around app data, can be 'before', 'after' or 'around'
	Returns
	-------
	df:
		A dataframe containing features and timestamps
	"""
	# read data frame
	if type(apl_events) == str:
		apl_events = apl_converter.read_apl_event_to_raw(apl_events)
	df_apl = apl_events.to_data_frame(time_format='datetime')
	df_apl = df_apl.set_index(df_apl.time, drop=True)
	sfreq = apl_events.info['sfreq']

	if app_data == None:
		signal_chunks_list = chunk_signal(df_apl, sfreq, window)
		time_chunks_list = chunk_signal(df_apl.index, sfreq, window)
	else:
		signal_chunks_list, time_chunks_list = chunk_signal_at_app(df_apl, channel_name='APActivity_code',
																   app_data=app_data, app_starttime=app_starttime,
																   app_endtime=app_endtime, app_window=app_window,
																   window=window)

	# extract features

	time_list = list()
	header_list = list()
	feature_list = list()
	for i, signal_chunk in enumerate(signal_chunks_list):

		# log sample start time
		if app_data == None:
			chunk_start =  time_chunks_list[i][0]
		else:
			chunk_start = time_chunks_list[i]

		# resample to 1 HZ
		signal_chunk = signal_chunk.resample('1S').mean()

		# estimate percent in:
		# Activity codes: 0=sedentary 1=standing 2=stepping 2.1=cycling 3.1=primary lying, 3.2=secondary lying 4=non-wear 5=travelling
		code_0 = len(signal_chunk[signal_chunk['APActivity_code'] == 0]) / len(signal_chunk) * 100  # sedentary
		code_1 = len(signal_chunk[signal_chunk['APActivity_code'] == 1]) / len(signal_chunk) * 100  # standing
		code_2 = len(signal_chunk[signal_chunk['APActivity_code'] == 2]) / len(signal_chunk) * 100  # stepping
		code_2_1 = len(signal_chunk[signal_chunk['APActivity_code'] == 2.1]) / len(signal_chunk) * 100  # cycling
		code_3 = len(signal_chunk[signal_chunk['APActivity_code'] == 3]) / len(signal_chunk) * 100  # laying
		code_3_1 = len(signal_chunk[signal_chunk['APActivity_code'] == 3.1]) / len(signal_chunk) * 100  # laying prim
		code_3_2 = len(signal_chunk[signal_chunk['APActivity_code'] == 3.2]) / len(signal_chunk) * 100  # laying second
		code_4 = len(signal_chunk[signal_chunk['APActivity_code'] == 4]) / len(signal_chunk) * 100  # non-wear
		code_5 = len(signal_chunk[signal_chunk['APActivity_code'] == 5]) / len(signal_chunk) * 100  # travelling

		# step count
		wind_step_count = signal_chunk.APCumulativeStepCount[len(signal_chunk)-1] - signal_chunk.APCumulativeStepCount[0]

		# estimate bouts in window
		movement_bout = 'no'
		bout_length = 0
		if window > bout_duration:
			j = 0
			bout_duration_s = int(bout_duration * 60 )
			while (j + (bout_duration_s) <= len(signal_chunk)):
				bout = signal_chunk['APActivity_code'][j:(j + bout_duration_s)]
				bout = bout.round()
				if (len(bout.unique()) == 1) and (round(bout.unique()[0]) == 2):
					movement_bout = 'yes'
					if bout_length == 0:
						bout_length = bout_duration_s
					else:
						bout_length = 1 + bout_length
				j = j + 1
		else:
			movement_bout = 'window too short'

		time_list.extend([chunk_start] * 13)
		header_list.extend(
			['apl_window','apl_step_count', 'apl_per_sedentary', 'apl_per_standing', 'apl_per_stepping', 'apl_per_cycling', 'apl_per_laying',
			 'apl_per_lay_prim', 'apl_per_lay_second', 'apl_per_nonwear', 'apl_per_travel', 'apl_bout_yn', 'apl_bout_length_s'])
		feature_list.extend(
			[window, wind_step_count, code_0, code_1, code_2, code_2_1, code_3, code_3_1, code_3_2, code_4, code_5, movement_bout, bout_length])

	# convert to df for output
	apl_df_full = {'time': time_list, 'feature': header_list, 'apl': feature_list}
	apl_df_full = pandas.DataFrame(apl_df_full)
	apl_df_full = apl_df_full.pivot(index='time', columns='feature', values='apl')

	return apl_df_full



def sub_feature_extraction(sub_path, weeks, devices, channels, window=10, apl_window=None, apl_bout=5, app_data=False, app_starttime='EMA_timestamp__start_beep_', app_endtime='EMA_timestamp_end_beep_', app_window='before', output=False, anon_datetime=True):
	"""
	Given a subject, and a week, implement feature extraction from sepcificed devices.
	Parameters
	----------
	sub_path: str
	Path to subject folder
	week: str
	Specification of which week to look into. Can be any of 'pre-1', 'pre-2' or 'pre-3'
	devices: list
	List containing devices to extract features from. Currently limited to 'apl' and 'emp'
	channels: list
	List containing channles to investigate for emp data. Future scripts will also include the Activpal ACC files.
	window: int
	Window in minutes to use for feature extraction
	apl_window: int
	Optional. whether to use a different window for apl data
	apl_bout: int
	Minutes of continuous activity to be considered as an activity bout for apl data
	app_data: bool
	Whether to use app data for feature extraction windoes. Requires specfication of other app variables
	app_starttime: str
	String containing name of column with app start time
	app_endtime: str
	String containing name of column with app end time
	app_window: str
	Defines search window around app data. Any of 'before', 'after', 'around'.
	output: bool
	If true, writes out CSV file using naming conventions of input file.
	Returns
	dt_anon: bool
	Anonymize datetimes in final output
	-------
	output: pandas.DataFrame or csv file
	"""
	# make name for report file
	start = datetime.datetime.now()
	start = start.strftime("%Y_%m_%d_%H%M%S")
	device_str = '_'.join(map(str, devices))
	sub_id = fileparts(sub_path)[1]
	# open report file
	error_file = open(sub_path + os.sep + 'feature_report_' + device_str + '_' + start + '.csv', 'w+')
	writer = csv.writer(error_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
	writer.writerow(['sub_id', 'date_time','week','app_file', 'device', 'output_generation'])

	for week in weeks:
		# set directory
		sub_week = sub_path + os.sep + week

		# get app file if specified
		if app_data == True:
			app_file = glob.glob(sub_week + os.sep + 'app' +  os.sep + "*ema.csv")
			if len(app_file) == 0:
				app_file_stat = 'missing'
			else:
				app_file_stat = 'found'
				app_file = app_file[0]
		else:
			app_file = None
			app_file_stat = 'n.v.t'

		# if both empatica + apl
		if ('emp' in devices) & ('apl' in devices):
			try:
				raw_wrb = merge_emp_to_apl_raw(sub_week)
				emp_file_stat = 'found'
				apl_file_stat = 'found'
			except:
				warnings.warn('One or both the APL and EMP files are missing! Skipping...')
				raw_wrb = None
				# check which is missing for reporting
				file_check = find_wearable_files(sub_week, wearable='empatica')
				file_check = [x for x in file_check if ('full') in x]
				if len(file_check)>0:
					emp_file_stat = 'found'
				else:
					emp_file_stat = 'missing'
				file_check = find_wearable_files(sub_week, wearable='apl')
				file_check = [x for x in file_check if ('events') in x]
				if len(file_check)>0:
					apl_file_stat = 'found'
				else:
					apl_file_stat = 'missing'

		# otherwise only select one
		else:

			if 'emp' in devices:
				# find e4 devices file + convert to raw
				files = find_wearable_files(sub_week, wearable='empatica')
				file = [x for x in files if ('full') in x]
				file = sub_week + os.sep + file[0]
				if os.path.isfile(file):
					raw_wrb = read_e4_to_raw(file)
					emp_file_stat = 'found'
				else:
					warnings.warn('No concatenated E4 file found!')
					raw_wrb = None
					emp_file_stat = 'missing'

			if 'apl' in devices:
				# find apl date and convert to raw
				files = find_wearable_files(sub_week, wearable='apl')
				file = [x for x in files if ('events') in x]
				file = sub_week + os.sep + file[0]
				if os.path.isfile(file):
					raw_wrb = read_apl_event_to_raw(file)
					apl_file_stat = 'found'
				else:
					warnings.warn('No Activpal events file found!')
					apl_file_stat = 'missing'
					raw_wrb = None

		# extract features to list
		if 'emp' in devices:
			try:
				emp_feats = list()
				if 'hr' in channels:
					emp_feats.extend([features_hr_from_raw(raw_wrb, channel_name='bvp', window=window, app_data=app_file, app_starttime=app_starttime, app_endtime=app_endtime, app_window=app_window)])
				if 'eda' in channels:
					emp_feats.extend([features_eda_from_raw(raw_wrb, channel_name='eda', delta=0.02,	window=window, app_data=app_file, app_starttime=app_starttime, app_endtime=app_endtime, app_window=app_window)])
				if 'temp' in channels:
					emp_feats.extend([features_temp_from_raw(raw_wrb, channel_name='temp', window=window, app_data=app_file, app_starttime=app_starttime, app_endtime=app_endtime, app_window=app_window)])
				if 'acc' in channels:
					emp_feats.extend([features_acc_from_raw(raw_wrb, channel_name_x='acc_x', channel_name_y='acc_y', channel_name_z='acc_z', window=window, app_data=app_file, app_starttime=app_starttime, app_endtime=app_endtime, app_window=app_window)])
				# merge list to single df
				emp_df = emp_feats[0]
				for i in range(1, len(emp_feats)):
					emp_df = emp_df.merge(emp_feats[i], left_index=True, right_index=True)
			except:
				emp_df = pandas.DataFrame()

		# extract activpal features
		if 'apl' in devices:
			if apl_window == None:
				apl_window = window
			try:
				apl_df = features_apl_from_events(raw_wrb, window=apl_window, bout_duration=apl_bout, app_data=app_file, app_starttime=app_starttime, app_endtime=app_endtime, app_window=app_window)
			except:
				apl_df = pandas.DataFrame()

		# reset raw to reduce memory
		raw_wrb = None

		# check save directory
		if app_data == False:
			type = 'wrb'
		else:
			type = 'app'

		# make output depending on what data is used
		if ('emp' in devices) & ('apl' in devices):
			out_df = pandas.merge(emp_df, apl_df, left_index=True, right_index=True, how='outer')
			output_file = sub_week + os.sep + type + os.sep +  fileparts(sub_path)[1] + "_features_emp_apl_" + str(window) + "min.csv"
		# If using only one device
		else:
			# create output naming conventon
			file_name = parse_wearable_filepath_info(file)['subject_file_id'] + '_features_' + devices[0]  +"_"+ str(window) + "min.csv"
			output_file = (sub_week + os.sep + type + os.sep + file_name)
			# what type wearable
			if 'emp' in devices:
				out_df = emp_df
			elif 'apl' in devices:
				out_df = apl_df

		# make random dates and times for anonimization
		random_time = numpy. random.randint(-5, 5, 1)[0]
		random_week = numpy.random.randint(-20, 20, 1)[0]
		# merge to app data if specified and randomize timestamps
		if (app_data == True) & (app_file_stat == 'found'):

			app_df = app_to_long(app_file)
			app_df = app_df.set_index(pandas.to_datetime(app_df['EMA_timestamp__start_beep_']))
			out_df = pandas.merge(app_df, out_df, left_index=True, right_index=True, how='outer')

			# anonymize app time stamps
			if anon_datetime == True:
				# convert to datetimes and round the seconds down, then keep the time
				out_df['EMA_timestamp__start_beep_'] = pandas.to_datetime(out_df['EMA_timestamp__start_beep_']).dt.floor('T')
				out_df['EMA_timestamp_end_beep_'] = pandas.to_datetime(out_df['EMA_timestamp_end_beep_']).dt.floor('T')
				# Anonymize week numbers
				out_df['week_number'] = out_df['EMA_timestamp__start_beep_'].dt.isocalendar().week + random_week
				# keep week day variable as is
				out_df['week_day'] = out_df['EMA_timestamp__start_beep_'].dt.isocalendar().day
				# anonymize beep start time, and keep only the time. Do same for beep end time
				out_df['start_time']  = out_df['EMA_timestamp__start_beep_'] + pandas.Timedelta(minutes=random_time)
				out_df['start_time'] = out_df['start_time'].dt.time
				out_df['end_time']  = out_df['EMA_timestamp_end_beep_']  + pandas.Timedelta(minutes=random_time)
				out_df['end_time'] = out_df['end_time'].dt.time
				# drop the original datetimes
				out_df = out_df.drop(['EMA_timestamp__start_beep_','EMA_timestamp_end_beep_'], axis=1)
				out_df = out_df.reset_index(drop=True)
			extraction_stat = 'success'
		elif (app_data==True) & (app_file_stat == 'missing'):
				extraction_stat = 'missing app file'

		# anonimize index
		if (app_data == False) & (anon_datetime == True):
			out_df= out_df.set_index(pandas.to_datetime(out_df.index).floor('T') + pandas.Timedelta(minutes=random_time))
			out_df['start_time'] = out_df.index
			out_df['week_number'] = out_df['start_time'].dt.isocalendar().week + random_week
			out_df['week_day'] = out_df['start_time'].dt.isocalendar().day
			out_df['start_time'] = out_df['start_time'].dt.time
			out_df = out_df.reset_index(drop=True)

		# save or return
		if output == True:
			if len(out_df.index) > 0:
				output_file = output_file + '.zip'
				out_df.to_csv(output_file, sep=',', compression='zip')
				extraction_stat = 'success'
			else:
				extraction_stat = 'fail'
		else:
			return out_df

		# write report file
		if 'emp' in devices:
			writer.writerow([sub_id, start, week, 'emp', app_file_stat, emp_file_stat, extraction_stat])
		if 'apl' in devices:
			writer.writerow([sub_id, start, week, 'apl', app_file_stat, apl_file_stat, extraction_stat])


	# memory clean up
	gc.collect()
"""	
	comment
	TODO:
	# delete and correct after no good duration (less than 10 min) or voltage too low at end of recording
	# create annotation like what is wrong with the recording
	# check the dates if this is a standard date and if the order needs to be adapted.
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

