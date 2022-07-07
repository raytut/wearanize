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
...

or to save the enironment package requirements (also unused)
python.exe -m pip freeze > requirements.txt

or use project specific used packages:
pip install pipreqs
pipreqs /path/to/this/project


"""
# imports #

import mne
import matplotlib
import numpy
import warnings
import os
import glob
import csv
import datetime
import heartpy
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


import sys
if sys.version_info >= (3, 6):
	import zipfile
	from zipfile import ZipFile
else:
	import zipfile36 as zipfile

import tempfile

from joblib import Parallel, delayed

# constants #
FILE_EXTENSION_WEARABLE_ZMAX_REEXPORT = "_merged"

# functions #

# =============================================================================
#
# =============================================================================
def fileparts(filepath):
	filepath = os.path.normpath(filepath)
	path_filename = os.path.split(filepath)
	filename = path_filename[1]
	path = path_filename[0]
	name_extension = os.path.splitext(filename)
	name = name_extension[0]
	extension = name_extension[1]
	return path, name, extension

# =============================================================================
#
# =============================================================================
def zip_directory(folderpath, zippath, deletefolder=False, compresslevel=6):
	with zipfile.ZipFile(zippath, mode='w') as zf:
		len_dir_path = len(folderpath)
		for root, _, files in os.walk(folderpath):
			for file in files:
				filepath = os.path.join(root, file)
				zf.write(filepath, filepath[len_dir_path:], compress_type=zipfile.ZIP_DEFLATED, compresslevel=compresslevel)
	if not deletefolder:
		shutil.rmtree(folderpath)

# =============================================================================
#
# =============================================================================
def safe_zip_dir_extract(filepath):
	temp_dir = tempfile.TemporaryDirectory()
	#temp_dir = tempfile.mkdtemp()
	with zipfile.ZipFile(filepath, 'r') as zipObj:
		zipObj.extractall(path=temp_dir.name)
	#temp_dir.cleanup()
	return temp_dir

# =============================================================================
#
# =============================================================================
def safe_zip_dir_cleanup(temp_dir):
	temp_dir.cleanup()

# =============================================================================
# 
# =============================================================================
def parse_wearable_filepath_info(filepath):
	split_str = '_'

	path, name, extension = fileparts(filepath)

	name_parts = name.split(split_str)

	subject_id = name_parts[0]
	period = name_parts[1]
	datatype = name_parts[2]
	device_wearable = name_parts[3]

	if name_parts.__len__() > 4:
		session = split_str.join(name_parts[4:])
	else:
		session = ''

	return {'subject_id': subject_id, 'filepath':  filepath, 'period':  period, 'datatype':  datatype, 'device_wearable':  device_wearable, 'session': session}

# =============================================================================
# 
# =============================================================================
def read_edf_to_raw(filepath, preload=True, format="zmax_edf", drop_zmax = ['BODY TEMP', 'LIGHT', 'NASAL L', 'NASAL R', 'NOISE', 'OXY_DARK_AC', 'OXY_DARK_DC', 'OXY_R_AC', 'OXY_R_DC', 'RSSI']):
	path, name, extension = fileparts(filepath)
	if (extension).lower() != ".edf":
		warnings.warn("The filepath " + filepath + " does not seem to be an EDF file.")
	raw = None
	if format == "zmax_edf":

		"""
		This reader is largely similar to the one for edf but gets and assembles all the EDFs in a folder if they are in the zmax data format
		"""
		path, name, extension = fileparts(filepath)
		check_channel_filenames = ['BATT', 'BODY TEMP', 'dX', 'dY', 'dZ', 'EEG L', 'EEG R', 'LIGHT', 'NASAL L', 'NASAL R', 'NOISE', 'OXY_DARK_AC', 'OXY_DARK_DC', 'OXY_IR_AC', 'OXY_IR_DC', 'OXY_R_AC', 'OXY_R_DC', 'RSSI']
		raw_avail_list = []
		channel_avail_list = []
		channel_read_list = []
		for iCh, name in enumerate(check_channel_filenames):
			checkname = path + os.sep + name + '.edf'
			if os.path.isfile(checkname):
				channel_avail_list.append(check_channel_filenames[iCh])
				if not name in drop_zmax:
					raw_avail_list.append(read_edf_to_raw(checkname, format="edf"))
					channel_read_list.append(check_channel_filenames[iCh])
		print("zmax edf channels found:")
		print(channel_avail_list)
		print("zmax edf channels read in:")
		print(channel_read_list)

		# append the raws together
		raw = raw_avail_list[0].add_channels(raw_avail_list[1:])

		# also append the units as this is not handled by raw.add_channels
		raw._raw_extras[0]['cal'] = [].append(raw._raw_extras[0]['cal'])
		for r in raw_avail_list[1:]:
			raw._orig_units.update(r._orig_units)
			ch_name = r._raw_extras[0]['ch_names'][0]
			raw._raw_extras[0]['cal'] = numpy.append(raw._raw_extras[0]['cal'],r._raw_extras[0]['cal'])
			raw._raw_extras[0]['ch_names'].append(r._raw_extras[0]['ch_names'])
			raw._raw_extras[0]['ch_types'].append(r._raw_extras[0]['ch_types'])
			if ch_name in ['EEG L', 'EEG R']:
				raw._raw_extras[0]['digital_max'] = numpy.append(raw._raw_extras[0]['digital_max'],32767)
				raw._raw_extras[0]['physical_max'] = numpy.append(raw._raw_extras[0]['physical_max'],1976)
				try: # as in nme this is deleted while reading in
					raw._raw_extras[0]['digital_min'] = numpy.append(raw._raw_extras[0]['digital_min'],-32767)
					raw._raw_extras[0]['physical_min'] = numpy.append(raw._raw_extras[0]['physical_min'],-1976)
				except:
					pass
			else:
				raw._raw_extras[0]['digital_max'] = numpy.append(raw._raw_extras[0]['digital_max'],r._raw_extras[0]['digital_max'])
				raw._raw_extras[0]['physical_max'] = numpy.append(raw._raw_extras[0]['physical_max'],r._raw_extras[0]['physical_max'])
				try: # as in nme this is deleted while reading in
					raw._raw_extras[0]['digital_min'] = numpy.append(raw._raw_extras[0]['digital_min'],r._raw_extras[0]['digital_min'])
					raw._raw_extras[0]['physical_min'] = numpy.append(raw._raw_extras[0]['physical_min'],r._raw_extras[0]['physical_min'])
				except:
					pass
			raw._raw_extras[0]['highpass'] = numpy.append(raw._raw_extras[0]['highpass'],r._raw_extras[0]['highpass'])
			raw._raw_extras[0]['lowpass'] = numpy.append(raw._raw_extras[0]['lowpass'],r._raw_extras[0]['lowpass'])
			raw._raw_extras[0]['n_samps'] = numpy.append(raw._raw_extras[0]['n_samps'],r._raw_extras[0]['n_samps'])
			raw._raw_extras[0]['offsets'] = numpy.append(raw._raw_extras[0]['offsets'],r._raw_extras[0]['offsets'])
			raw._raw_extras[0]['units'] = numpy.append(raw._raw_extras[0]['units'],r._raw_extras[0]['units'])

		raw._raw_extras[0]['sel'] = range(channel_avail_list.__len__())
		raw._raw_extras[0]['n_chan'] = channel_avail_list.__len__()
		raw._raw_extras[0]['orig_n_chan'] = channel_avail_list.__len__()

		#raw.info['chs'][0]['unit']
	else:
		raw = mne.io.read_raw_edf(filepath, preload = True)
	return raw

# =============================================================================
# E4 to mne.raw
# =============================================================================
def read_e4_to_raw(project_folder, emp_file, resample=None):
	
	# Read in the e4 file
	emp_zip=zipfile.ZipFile(os.path.join(project_folder,emp_file))
	channels=['BVP.csv', 'EDA.csv','TEMP.csv', 'ACC.csv']
	mne_list=["unretrieved"]*len(channels)
	# Check if single session or full recording
	if "full" not in emp_file:
		# Run over all signals
		for i, signal_type in enumerate(channels):
			if signal_type!="ACC.csv":
				# Read signal
				raw=pandas.read_csv(emp_zip.open(signal_type))
				# create channel info for mne.info file
				chanel=signal_type.split(".")
				chanel=chanel[0].lower()
				sfreq=int(raw.iloc[0,0])
				timestamp=int(float(raw.columns[0]))
				mne_info=mne.create_info(ch_names=[chanel], sfreq=sfreq, ch_types="misc")
				# Create MNE Raw object and add to a list of objects
				mne_obj=mne.io.RawArray([raw.iloc[1:,0]], mne_info, first_samp=timestamp)
				mne_list[i]=mne_obj
			else:
				# Read signal
				raw=pandas.read_csv(emp_zip.open(signal_type))
				# create channel info for mne.info file
				chanel=signal_type.split(".")
				chanel=chanel[0].lower()
				sfreq=int(raw.iloc[0,0])
				timestamp=int(float(raw.columns[0]))
				mne_info=mne.create_info(ch_names=["acc_x", "acc_y", "acc_z"], sfreq=sfreq, ch_types="misc")
				# Create MNE Raw object and add to a list of objects
				mne_obj=mne.io.RawArray([raw.iloc[1:,0], raw.iloc[1:,1], raw.iloc[1:,2]], mne_info, first_samp=timestamp)
				mne_list[i]=mne_obj
	e4_to_raw=[channels, mne_list]
	return(e4_to_raw)

# =============================================================================
# 		
# =============================================================================
def edfWriteAnnotation(edfWriter, onset_in_seconds, duration_in_seconds, description, str_format='utf-8'):
	edfWriter.writeAnnotation(onset_in_seconds, duration_in_seconds, description, str_format)

# =============================================================================
# 
# =============================================================================
def write_raw_to_edf(raw, filepath, format="zmax_edf"):
	path, name, extension = fileparts(filepath)
	if (extension).lower() != ".edf":
		warnings.warn("The filepath " + filepath + " does not seem to be an EDF file.")
	if format == "zmax_edf":
		channel_dimensions_zmax = {'BATT': 'V', 'BODY TEMP': "C", 'dX': "g", 'dY': "g", 'dZ': "g", 'EEG L': "uV", 'EEG R': "uV", 'LIGHT': "", 'NASAL L': "", 'NASAL R': "", 'NOISE': "", 'OXY_DARK_AC': "", 'OXY_DARK_DC': "", 'OXY_IR_AC': "", 'OXY_IR_DC': "", 'OXY_R_AC': "", 'OXY_R_DC': "", 'RSSI': ""}

		#EDF_format_extention = ".edf"
		EDF_format_filetype = pyedflib.FILETYPE_EDFPLUS
		#temp_filterStringHeader = 'HP ' + str(self.prefilterEDF_hp) + ' Hz'
		nAnnotation = 1
		has_annotations = False
		nChannels = raw.info['nchan']
		sfreq = raw.info['sfreq']
		edfWriter = pyedflib.EdfWriter(filepath, nChannels, file_type=EDF_format_filetype)

		"""
		Only when the number of annotations you want to write is more than the number of seconds of the duration of the recording, you can use this function to increase the storage space for annotations */
		/* Minimum is 1, maximum is 64 */
		"""
		if has_annotations:
			edfWriter.set_number_of_annotation_signals(nAnnotation) #nAnnotation*60 annotations per minute on average
		edfWriter.setTechnician('')
		edfWriter.setRecordingAdditional('merged from single zmax files')
		edfWriter.setPatientName('')
		edfWriter.setPatientCode('')
		edfWriter.setPatientAdditional('')
		edfWriter.setAdmincode('')
		edfWriter.setEquipment('Hypnodyne zmax')
		edfWriter.setGender(0)
		edfWriter.setBirthdate(datetime.date(2000, 1, 1))
		#edfWriter.setStartdatetime(datetime.datetime.now())
		edfWriter.setStartdatetime(raw.info['meas_date'])
		edfWriteAnnotation(edfWriter,0, -1, u"signal_start")

		for iCh in range(0,nChannels):
			ch_name = raw.info['ch_names'][iCh]
			dimension = channel_dimensions_zmax[ch_name] #'uV'
			sf = int(round(sfreq))
			pysical_min = raw._raw_extras[0]['physical_min'][iCh]
			pysical_max = raw._raw_extras[0]['physical_max'][iCh]
			digital_min = raw._raw_extras[0]['digital_min'][iCh]
			digital_max = raw._raw_extras[0]['digital_max'][iCh]
			prefilter = 'HP:0.1Hz LP:75Hz'

			channel_info = {'label': ch_name, 'dimension': dimension, 'sample_rate': sf,
							'physical_max': pysical_max, 'physical_min': pysical_min,
							'digital_max': digital_max, 'digital_min': digital_min,
							'prefilter': prefilter, 'transducer': 'none'}

			edfWriter.setSignalHeader(iCh, channel_info)
			edfWriter.setLabel(iCh, ch_name)

		data = raw.get_data()
		data_list = []
		for iCh in range(0,nChannels):
			data_list.append(data[iCh,] / raw._raw_extras[0]['units'][iCh])
		edfWriter.writeSamples(data_list, digital = False) # write physical samples

		#for iChannel_all in range(0, nChannels):
		#	edfWriter.writePhysicalSamples(data[iChannel_all,])

		edfWriter.close()
	else:
		raw.export(filepath,fmt='edf', physical_range='auto', add_ch_type=False, overwrite=True, verbose=None)
	return filepath

# =============================================================================
# 
# =============================================================================
def read_edf_to_raw_zipped(filepath, format="zmax_edf", drop_zmax=['BODY TEMP', 'LIGHT', 'NASAL L', 'NASAL R', 'NOISE', 'OXY_DARK_AC', 'OXY_DARK_DC', 'OXY_R_AC', 'OXY_R_DC', 'RSSI']):
	temp_dir = safe_zip_dir_extract(filepath)
	raw = None
	if format == "zmax_edf":
		raw = read_edf_to_raw(temp_dir.name + os.sep + "EEG L.edf", format=format, drop_zmax=drop_zmax)
	elif format == "edf":
		fileendings = ('*.edf', '*.EDF')
		filepath_list_edfs = []
		for fileending in fileendings:
			filepath_list_edfs.extend(glob.glob(temp_dir.name + os.sep + fileending,recursive=True))
		if filepath_list_edfs:
			raw = read_edf_to_raw(filepath_list_edfs[0], format=format)
	safe_zip_dir_cleanup(temp_dir)
	return raw

# =============================================================================
# 
# =============================================================================
def write_raw_to_edf_zipped(raw, zippath, format="zmax_edf", compresslevel=6):
	temp_dir = tempfile.TemporaryDirectory()
	filepath = temp_dir.name + os.sep + fileparts(zippath)[1] + '.edf'
	write_raw_to_edf(raw, filepath, format)
	zip_directory(temp_dir.name, zippath, deletefolder=True, compresslevel=compresslevel)
	safe_zip_dir_cleanup(temp_dir)
	return zippath

# =============================================================================
# 
# =============================================================================
def raw_zmax_data_quality(raw):
		# the last second of Battery voltage
		quality = None
		try:
			quality = statistics.mean(raw.get_data(picks=['BATT'])[-256:])
		except:
			pass
		return quality

# =============================================================================
# 
# =============================================================================
def get_raw_by_date_and_time(raw, datetime, duration_seconds, offset_seconds=0.0): 
	"""get raw data file according to time stamps
	see definition of mne.io.Raw.crop:
		Limit the data from the raw file to go between specific times. Note
		that the new ``tmin`` is assumed to be ``t=0`` for all subsequently
		called functions (e.g., :meth:`~mne.io.Raw.time_as_index`, or
		:class:`~mne.Epochs`). New :term:`first_samp` and :term:`last_samp`
		are set accordingly.
		Thus function operates in-place on the instance.
		Use :meth:`mne.io.Raw.copy` if operation on a copy is desired.
		
		Parameters
		----------
		%(tmin_raw)s
		%(tmax_raw)s
		%(include_tmax)s
		%(verbose)s
		
		Returns
		-------
		raw : instance of Raw
			The cropped raw object, modified in-place.
		"""
	#find the start date and end date
	#list all files and search all the relevant files that fall within these time limits
	#load those files in a raw format
	#extract the time points using  raw.crop(tmin=0+duration_seconds)
	#check if all channels are present in all files
	#fill the rest periods with nans/empty recordings and concatenate the recorings in time

# =============================================================================
# 
# =============================================================================
def raw_detect_heart_rate_PPG(raw, ppg_channel):
	"""
	Detect the PPG artifacts in heart rate
	Detect the heartrate and outlier heart rates, 
	Output the heartrate signal with inter RR intervals and timepoints and artifact periods annotated.
	Optionally add to the raw data as a new channel with nans where there is not heart rate detected or artifactious
	"""

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

# =============================================================================
# 
# =============================================================================
def find_wearable_files(parentdirpath, wearable):
	"""
	finds all the wearable data from different wearables in the HB file structure given the parent path to the subject files
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

	# compatible with python versions < 3.10 remove the root_dir
	for i, filepath in enumerate(filepath_list):
		filepath_list[i] = filepath.replace(parentdirpath + os.sep,"")

	return filepath_list

# =============================================================================
# 
# =============================================================================
def parse_wearable_data_write_csv(parentdirpath, filepath_csv_out, device='all'):

	filepath_list = find_wearable_files(parentdirpath, device)

	with open(filepath_csv_out, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
		writer.writerow(['subject_id', 'filepath', 'period', 'datatype', 'device_wearable', 'session'])
		for filepath in filepath_list:
			info = parse_wearable_filepath_info(filepath)
			writer.writerow([info['subject_id'], info['filepath'], info['period'], info['datatype'], info['device_wearable'], info['session']])

# =============================================================================
# Parser: adds info
# =============================================================================
def parse_wearable_data_with_csv_annotate_datetimes(parentdirpath, filepath_csv_in, filepath_csv_out, device='all', reexport=True):
	df_csv_in = pandas.read_csv(filepath_csv_in)
	df_csv_in.reset_index()  # make sure indexes pair with number of rows

	with open(filepath_csv_out, 'w', newline='') as csvfile2:
		writer = csv.writer(csvfile2, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')

		header_new = numpy.append(df_csv_in.columns.values, ['signal', 'rec_start_datetime', 'rec_stop_datetime', 'rec_duration_datetime', 'sampling_rate_max_Hz', 'rec_quality'])
		writer.writerow(header_new)
		for i, row in df_csv_in.iterrows():
			filepath = row['filepath']
			device_wearable = row['device_wearable']
			session = row['session']
			
			signal='unretrieved'
			rec_start_datetime = 'unretrieved'
			rec_stop_datetime = 'unretrieved'
			rec_duration_datetime = 'unretrieved'
			sampling_rate_max_Hz = 'unretrieved'
			rec_quality = 'unretrieved'


			try:
				if device_wearable == 'zmx':
					if session in ["1", "2", "3", "4", "5", "6", "7", "8"]:
						filepath_full = parentdirpath + os.sep + filepath
						raw = read_edf_to_raw_zipped(filepath, format="zmax_edf")
						rec_start_datetime = raw.info['meas_date']
						rec_stop_datetime = rec_start_datetime + datetime.timedelta(seconds=(raw._last_time - raw._first_time))
						rec_duration_datetime = datetime.timedelta(seconds=(raw._last_time - raw._first_time))
						sampling_rate_max_Hz = raw.info['sfreq']
						rec_quality = raw_zmax_data_quality(raw)
						if reexport:
							path, name, extension = fileparts(filepath)
							reexport_filepath = path + os.sep + name + FILE_EXTENSION_WEARABLE_ZMAX_REEXPORT + ".zip"
							try:
								write_raw_to_edf_zipped(raw, reexport_filepath, format="zmax_edf") # treat as a speacial zmax read EDF for export
								print("Re-exported '%s' to '%s'" % (filepath, reexport_filepath))
							except:
								print("Failed to re-export '%s' to '%s' was not existent and was created" % (filepath, reexport_filepath))

						signal="zmx_all"
				elif device_wearable == 'emp':
				# Make this a try, to avoid the improper files and concatenated ones
					try: 
						# If we cant turn into integer, its probably right	
						session=int(session)
						filepath_full = parentdirpath + os.sep + filepath
						emp_zip=zipfile.ZipFile(filepath_full)
						tzinfo=datetime.timezone(datetime.timedelta(0))
						# Estimate different parameters per signal
						for signal_types in ['IBI.csv','BVP.csv', 'HR.csv','EDA.csv','TEMP.csv', 'ACC.csv']:
							raw = pandas.read_csv(emp_zip.open(signal_types))
							if signal_types == "IBI.csv":
								signal = signal_types
								rec_start_datetime = datetime.datetime.fromtimestamp(int(float(raw.columns[0])), tz=tzinfo)
								rec_stop_datetime = rec_start_datetime + datetime.timedelta(seconds=(raw.iloc[-1,0]))
								rec_duration_datetime=(rec_stop_datetime - rec_start_datetime)
								sampling_rate_max_Hz = "custom"
								rec_quality= raw[" IBI"].sum()/raw.iloc[-1,0]
								row_new = numpy.append(row.values, [signal, rec_start_datetime, rec_stop_datetime, rec_duration_datetime, sampling_rate_max_Hz, rec_quality])
								writer.writerow(row_new)
							else:
								signal = signal_types
								rec_start_datetime=datetime.datetime.fromtimestamp(int(float(raw.columns[0])), tz=tzinfo)
								rec_stop_datetime = rec_start_datetime + datetime.timedelta(((((len(raw.index)-1)*(1/raw.iloc[0,0]))/60)/60)/24)
								rec_duration_datetime=(datetime.timedelta(((len(raw.index)-1)*(1/raw.iloc[0,0])/60)/60)/24)
								sampling_rate_max_Hz=str(raw.iloc[0,0])
								row_new = numpy.append(row.values, [signal, rec_start_datetime, rec_stop_datetime, rec_duration_datetime, sampling_rate_max_Hz, rec_quality])
								writer.writerow(row_new)
					except:
						pass
				elif device_wearable == 'apl':
					pass
				elif device_wearable == 'app':
					pass
			except:
				print("cannot read infos from file: " + filepath_full)
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
			list[1]:index of the original start_stop pair in reach with the later start
			list[2]:the offset in datetime.timedelta to add to the first pair to match the second pair start time
			list[3]:the overlap in datetime.timedelta that the events share
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


def raw_get_integrated_acc(raw, ch_name_acc_x, ch_name_acc_y, ch_name_acc_z, resample_Hz=None):
	# TODO make integrated activity from the three channel data
	# TODO also resample the activity to the required sampling rate
	if resample_Hz is not None:
		pass


def get_signal(filepath, wearable, type = 'acc'):
	path, name, extension = fileparts(filepath)

	if type not in ['acc', 'hr']:
		TypeError('type not known')
	if wearable == 'zmx':
		if FILE_EXTENSION_WEARABLE_ZMAX_REEXPORT not in name:
			raw = read_edf_to_raw_zipped(filepath, format="zmax_edf")
		else:
			raw = read_edf_to_raw_zipped(filepath, format="edf")
		if type == 'acc':
			raw_get_integrated_acc(raw, ch_name_acc_x='dX', ch_name_acc_y='dY', ch_name_acc_z='dZ', resample_Hz=2)
		elif type == 'hr':
			pass
	elif wearable == 'emp':
		if type == 'acc':
			pass
		elif type == 'hr':
			pass
	elif wearable == 'apl':
		if type == 'acc':
			pass
		elif type == 'hr':
			TypeError('activPAL does not have heart rate signal')



def sync_wearables(parentdirpath, filepath_csv_in, filepath_csv_out, device='all'):
	df_csv_in = pandas.read_csv(filepath_csv_in)
	df_csv_in.reset_index()  # make sure indexes pair with number of rows

	grouped_by_subject = df.groupby('subject_id', sort=False)
	signal_types_valid = ["zmx_all", 'ACC.csv']
	first_rows_written = False
	for subject_id, df_by_subject_id in grouped_by_subject:
		df_by_subject_id_filtered = df_by_subject_id.signal.isin(signal_types_valid)

		# create empty columns to fill potentially with sync infos
		df_by_subject_id_filtered['rec_start_datetime_reference'] = numpy.datetime64("NaT")
		df_by_subject_id_filtered['rec_stop_datetime_reference'] = numpy.datetime64("NaT")
		df_by_subject_id_filtered['rec_duration_datetime_reference'] = numpy.datetime64("NaT")
		df_by_subject_id_filtered['sampling_rate_max_Hz_reference_adaption'] = numpy.NAN


		['rec_start_datetime_reference', 'rec_stop_datetime_reference', 'rec_duration_datetime_reference', 'sampling_rate_max_Hz_reference_adaption']
		list_datetimes_paired = []
		for row in df_by_subject_id_filtered:
			list_datetimes_paired.append([row.rec_start_datetime, row.df_by_subject_id_filtered.rec_stop_datetime])

		list_indices_paired_offsetTo2nd_overlap = sync_reach(list_datetimes_paired)
		for i_i2_p_o_o in list_indices_paired_offsetTo2nd_overlap:
			i = i_i2_p_o_o[0]
			i2 = i_i2_p_o_o[1]
			device_wearable_i = df_by_subject_id_filtered.device_wearable[i]
			device_wearable_i2 = df_by_subject_id_filtered.device_wearable[i2]
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

			device_wearable_sync = df_by_subject_id_filtered.device_wearable[i_sync]
			device_wearable_ref = df_by_subject_id_filtered.device_wearable[i_ref]
			fsample = 2
			signal_sync = get_signal(filepath=df_by_subject_id_filtered.filepath[i_sync], wearable=device_wearable_sync, type='acc', resample=fsample)
			signal_ref = get_signal(filepath=df_by_subject_id_filtered.filepath[i_ref], wearable=device_wearable_ref, type='acc', resample=fsample)
			lag_seconds, dilation, lag_after_dilation_seconds, sample_rate_adaptation_factor = sync_signals(signal_ref, signal_sync, fsample)

			rec_start_datetime_reference_i_sync = df_by_subject_id_filtered.rec_start_datetime[i_sync] + datetime.timedelta(seconds=lag_seconds)
			rec_stop_datetime_reference_i_sync = df_by_subject_id_filtered.rec_stop_datetime[i_sync] + datetime.timedelta(seconds=lag_seconds)
			sampling_rate_max_Hz_reference_adaption_i_sync = df_by_subject_id_filtered.sampling_rate_max_Hz_reference_adaption[i_ref] * sample_rate_adaptation_factor

			rec_start_datetime_reference_i_ref = df_by_subject_id_filtered.rec_start_datetime[i_ref]
			rec_stop_datetime_reference_i_ref = df_by_subject_id_filtered.rec_stop_datetime[i_ref]
			sampling_rate_max_Hz_reference_adaption_i_ref = df_by_subject_id_filtered.sampling_rate_max_Hz_reference_adaption[i_ref]

			df_by_subject_id_filtered.loc[i_sync, 'rec_start_datetime_reference'] = rec_start_datetime_reference_i_sync
			df_by_subject_id_filtered.loc[i_sync, 'rec_stop_datetime_reference'] = rec_stop_datetime_reference_i_sync
			df_by_subject_id_filtered.loc[i_sync, 'rec_duration_datetime_reference'] = rec_stop_datetime_reference_i_sync - rec_start_datetime_reference_i_sync
			df_by_subject_id_filtered.loc[i_sync, 'sampling_rate_max_Hz_reference_adaption'] = sampling_rate_max_Hz_reference_adaption_i_sync

			df_by_subject_id_filtered.loc[i_ref, 'rec_start_datetime_reference'] = rec_start_datetime_reference_i_ref
			df_by_subject_id_filtered.loc[i_ref, 'rec_stop_datetime_reference'] = rec_stop_datetime_reference_i_ref
			df_by_subject_id_filtered.loc[i_ref, 'rec_duration_datetime_reference'] = rec_stop_datetime_reference_i_ref - rec_start_datetime_reference_i_ref
			df_by_subject_id_filtered.loc[i_ref, 'sampling_rate_max_Hz_reference_adaption'] = sampling_rate_max_Hz_reference_adaption_i_ref

		#write out in chunks of rows
		if first_rows_written:
			df_by_subject_id_filtered.to_csv(filepath_csv_out, mode='a', index=False, header=False)
		else:
			df_by_subject_id_filtered.to_csv(filepath_csv_out, mode='a', index=False, header=True)






# =============================================================================
#  E4concatenation function
# =============================================================================
def e4_concatenate(project_folder, sub_nr, resampling=None):  
	
	# Set sub nr as string
	sub=str(sub_nr)
	# Make array with the sessions for loop
	sessions = glob.glob(os.path.join(project_folder, str(sub)) + "/pre-*/wrb")
		
	# Reset for memory 
	full_df=None
	df=None
	# Loop over all session files
	for session_type in sessions:
	   
		#Path with E4 files. Only run if the files exist
		filepath = (str(session_type))
		if os.path.isdir(filepath)==True:
			 
			# Get all directories with E4 sessions for subject, merge directory from the list
			dir_list = glob.glob(filepath+"/*wrb_emp_*.zip")
			# Only keep the empatica folders, drop the folder with concatenated data
			dir_list=[ x for x in  dir_list if "wrb_emp" in x ]
			dir_list=[ x for x in  dir_list if "wrb_emp_full" not in x ]
			
			# Only Run if there are recordings 
			if len(dir_list)>0:
				
				# Check if merge directory (for output) exists, if not then make it
				try:
					# Make a directory that matches the HBS format
					conc_file=dir_list[0][:-7]
					os.makedirs(str(conc_file))
				except FileExistsError:
					pass
				
				# Set E4 data types for loop
				data_types=['EDA.csv','TEMP.csv', 'IBI.csv','BVP.csv', 'HR.csv', 'ACC.csv']
								
				for data_type in data_types:
				   
					#Make Empty DF as master df for data type
					full_df=pandas.DataFrame()


					 #IBI is special case
					if data_type=='IBI.csv':
							
						#Select Directory from available list
						for k in dir_list:
									
							#Select File for single session, import as df
							zipdir=ZipFile(k)
									
							# Sometime IBI files are empty, so try this instead
							try:
								df=pandas.read_csv(zipdir.open(data_type))
								#Get time stamp
								time=list(df)
								time=time[0]
								time=float(time)
									 
								#Rename time column to time, data to Data
								df=df.rename(columns={ df.columns[0]: "time" })
								df=df.rename(columns={ df.columns[1]: "data" })
									 
								#Add the starttime from time stamp (time) to the column+Convert to datetime
							   # time=dt.datetime.fromtimestamp(time)
								df['time']=time + df['time']
								df['time']=pandas.to_datetime(df['time'],unit='s')

								#Append to master data frame the clear it for memory
								full_df =pandas.concat([full_df, df])
								df=pandas.DataFrame() 
							except: 
								pass
								 
						#Convert IBI to ms and sort by date:
						full_df['data']=full_df['data']*1000
						full_df = full_df.sort_values('time', ascending=True)
								 
						#Set Output Names and direcotries, save as csv
						fullout=(str(conc_file)+"/"+str(data_type))
						full_df.to_csv(str(fullout),sep='\t',index=True)
						# Clear dataframes for more memory
						full_df=pandas.DataFrame()
							 	 
					#ACC also special case, implement alternate combination method
					elif data_type=='ACC.csv':
							 
						#Select Directory, go through files
						for k in dir_list:
								 
							#Select File, Import as df
							zipdir=ZipFile(k)
							df=pandas.read_csv(zipdir.open(data_type))
									 
							#Get time stamp (Used Later)
							time=list(df)
							time=time[0]
							time=float(time)
									
							#Get Sampling Frequency, convert to time
							samp_freq=df.iloc[0,0]
							samp_freq=float(samp_freq)
							samp_time=1/samp_freq
								
							#Drop sampling rate from df (first row)
							df=df.drop([0])
								
							#Rename data columns to corresponding axes
							df=df.rename(columns={ df.columns[0]: "acc_x" })
							df=df.rename(columns={ df.columns[1]: "acc_y" })
							df=df.rename(columns={ df.columns[2]: "acc_z" })
											
							#Make array of time stamps
							df_len=len(df)
							time=pandas.to_datetime(time,unit='s')
							times = [time]
							for i in range (1,(df_len)):
								time = time + datetime.timedelta(seconds=samp_time)
								times.append (time)
										
							#Add time and data to dataframe
							df['time'] = times
											
							# Do resampling if specified
							if resampling!=None:
								# If downsampling
								if resampling>samp_time:
									# Upsample data to 256HZ here to avoid large memory costs
									df=df.resample((str(resampling)+"S"), on="time").mean()
								# If Upsampling
								else:
									df=df.set_index("time")
									df=df.resample((str(resampling)+"S")).ffill()	
								   
							#Append to master data frame
							full_df =pandas.concat([full_df, df])
							df=pandas.DataFrame()
						 
						#Sort master by date:
						full_df = full_df.sort_index()
							 
						#Set Output Names and direcotries, save as csv
						fullout=(str(conc_file)+"/"+str(data_type))
						full_df.to_csv(str(fullout),sep='\t',index=True)
							 
						# Clear dataframe and free memory
						full_df=pandas.DataFrame()
							 							
					#All other data structures:
					else:
						for k in dir_list:
							
							#Select File, Import as df
							zipdir=ZipFile(k)
							df=pandas.read_csv(zipdir.open(data_type))
								
							##Get start time+sampling frequency
							start_time = list(df)
							start_time=start_time[0]
							samp_freq=df.iloc[0,0]
								
							#Change samp freq to samp time
							samp_time=1/samp_freq
							
							#Drop sampling rate from df
							df=df.drop([0])

							#Convert start time to date time
							start_time=int(float(start_time))
							start_time=pandas.to_datetime(start_time,unit='s')
								 
							#Make array of time
							file_len=len(df)	
							times = [start_time]
							for i in range (1,(file_len)):
								start_time = start_time + datetime.timedelta(seconds=samp_time)
								times.append (start_time)
								
							#Add time and data to dataframe
							df['time']= times
							 
							#Rename first column to Data
							df=df.rename(columns={df.columns[0]: "data" })
								 
							# Do resampling if specified
							if resampling!=None:
								# If downsampling
								if resampling>samp_time:
									# Upsample data to 256HZ here to avoid large memory costs
									df=df.resample((str(resampling)+"S"), on="time").mean()
								# If Upsampling
								else:
									df=df.set_index("time")
									df=df.resample((str(resampling)+"S")).ffill()

							#Append to master data frame
							full_df =pandas.concat([full_df, df])
							df=pandas.DataFrame()

						#Sort by date:
						full_df = full_df.sort_index()

						#Set Output Names and direcotries, save as csv
						fullout=(str(conc_file)+"/"+str(data_type))
						full_df.to_csv(str(fullout),sep='\t',index=True)

						# Clear data frame and free up memory
						full_df=pandas.DataFrame()

				# Zip file
				zf = ZipFile((conc_file +".zip"), "w")
				for dirname, subdirs, files in os.walk(conc_file):
					zf.write(dirname)
					for filename in files:
						zf.write(os.path.join(dirname, filename), compress_type=zipfile.ZIP_DEFLATED)
				shutil.rmtree(conc_file)

# =============================================================================
#  E4 concatenation in parallel
# =============================================================================

def e4_concatente_par(project_folder, verbose=0): # TODO: Test
	# Get list of subjects
	sub_list=glob.glob(project_folder + "/sub-*")
	Parallel(n_jobs=-2, verbose=verbose)(delayed(e4_concatenate)(project_folder, i) for i in sub_list)


def nullable_string(val):
	if not val:
		return None
	return val


def dir_path(pathstring):
	pathstring = os.path.normpath(pathstring)
	if nullable_string(pathstring):
		if os.path.isdir(pathstring):
			return pathstring
		else:
			raise NotADirectoryError(pathstring)
	return None
	
	
def dir_path_create(pathstring):
	if nullable_string(pathstring):
		try:
			pathstring = dir_path(pathstring)
			return pathstring
		except NotADirectoryError:
			try:
				os.makedirs(pathstring, exist_ok=True)
				print("Directory '%s' was not existent and was created" %pathstring)
			except:
				print("Directory '%s' was could not be created" %pathstring)
				NotADirectoryError(pathstring)
			finally:
				return nullable_string(pathstring)
	else:
		return None

"""
	comment
	TODO:
	# delete and correct after no good duration (less than 10 min) or voltage too low at end of recording
	# create annotation like what is wrong with the recording
	# check the dates if this is a standard date and if the order needs to be adapted.
	# PPG to HR signal
	# cross correlation, on 10 min snippets with linear extrapolation.
	# read activPAL data: https://github.com/R-Broadley/python-uos-activpal
"""
if __name__ == "__main__":

	# Instantiate the argument parser
	parser = argparse.ArgumentParser(prog='wearanize', description='this is wearanize software')

	# Required positional argument
	parser.add_argument('function', type=str,
					help="either 'init' to initialize the recording or 'request' to request data with additional arguments")

	# Optional argument
	parser.add_argument('--path_data', type=dir_path,
					help='An optional path to the parent folder where the data is stored and to be initialized (default is current directory)')

	# Optional argument
	parser.add_argument('--path_init', type=dir_path_create,
					help='An optional path to the parent folder where the data is stored and to be initialized (default is current directory)')

	# Optional argument
	parser.add_argument('--path_output', type=dir_path_create,
					help="An optional path to the folder where the requested output will be stored (default is in the init dirctory subfolder 'output')")

	# Switch
	parser.add_argument('--no_reexport_on_init', action='store_false',
					help='switch to indicate if on init also some wearable files should NOT be reexported in more concise formats')

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

	path_data = pathlib.Path().resolve() # the current working directory
	if args.path_data is not None:
		path_data = args.path_data

	path_init = pathlib.Path().resolve() + os.sep + '.wearanize'
	if args.path_init is not None:
		path_init = args.path_init

	path_output = path_init + os.sep + 'output'
	if args.path_init is not None:
		path_output = args.path_init

	reexport_on_init = True
	if args.no_reexport_on_init is not None:
		reexport_on_init = args.no_reexport_on_init

	if args.function == 'test':
		#--tests--#

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

		raw = read_edf_to_raw_zipped("Y:/HB/data/test_data_zmax/FW.zip", format="zmax_edf")
		#write_raw_to_edf(raw, "Y:/HB/data/test_data_zmax/FW.merged.edf", format="zmax_edf")  # treat as a speacial zmax read EDF for export
		#write_raw_to_edf_zipped(raw, "Y:/HB/data/test_data_zmax/FW.merged.zip", format="zmax_edf") # treat as a speacial zmax read EDF for export
		#raw_reread = read_edf_to_raw_zipped("Y:/HB/data/test_data_zmax/FW.merged.zip", format="edf")
		#write_raw_to_edf_zipped(raw, "Y:/HB/data/test_data_zmax/FW.merged.reread.zip", format="zmax_edf") # treat as a speacial zmax read EDF for export

		delay_ref = 16
		signal_ref = raw.get_data(picks=['EEG L'],start=0+delay_ref, stop=256*60*10*3+delay_ref)[0,]
		signal_sync = raw.get_data(picks=['EEG R'],start=0, stop=256*60*10*3)[0,]

		lag_seconds, dilation, lag_after_dilation_seconds, sample_rate_adaptation_factor = sync_signals(signal_ref, signal_sync, 256, chunk_size_seconds=60*10, chunk_step_seconds=60*5, lag_merge_window_seconds=60*20, max_merge_lag_difference_seconds=0.5, threshold_chunk_min_match=2, allow_anti_correlation=False)
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
		parse_wearable_data_with_csv_annotate_datetimes(parentdirpath=path_data,filepath_csv_in=wearable_file_structure_annotation_csv,filepath_csv_out=wearable_file_structure_annotation_datetime_csv,device='zmax', reexport=reexport_on_init)

		df = pandas.read_csv(wearable_file_structure_annotation_datetime_csv)
		#df.iloc[[0]]

		print("tests finished")

	elif args.function == 'init':
		wearable_file_structure_annotation_csv = "wearabout_0.csv"
		wearable_file_structure_annotation_datetime_csv = "wearabout_1_annotation.csv"

		parse_wearable_data_write_csv(parentdirpath=path_data,filepath_csv_out=wearable_file_structure_annotation_csv,device='zmax')
		parse_wearable_data_with_csv_annotate_datetimes(parentdirpath=path_data,filepath_csv_in=wearable_file_structure_annotation_csv,filepath_csv_out=wearable_file_structure_annotation_datetime_csv,device='zmax', reexport=reexport_on_init)

	elif args.function == 'request':
		pass
	else:
		parser.error("function unknown")

