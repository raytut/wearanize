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

import sys
if sys.version_info >= (3, 6):
	import zipfile
else:
	import zipfile36 as zipfile

import tempfile

# functions #

def fileparts(filepath):
	path_filename = os.path.split(filepath)
	filename = path_filename[1]
	path = path_filename[0]
	name_extension = os.path.splitext(filename)
	name = name_extension[0]
	extension = name_extension[1]
	return path, name, extension


def zip_directory(folderpath, zippath, deletefolder=False, compresslevel=6):
	with zipfile.ZipFile(zippath, mode='w') as zf:
		len_dir_path = len(folderpath)
		for root, _, files in os.walk(folderpath):
			for file in files:
				filepath = os.path.join(root, file)
				zf.write(filepath, filepath[len_dir_path:], compress_type=zipfile.ZIP_DEFLATED, compresslevel=compresslevel)
	if not deletefolder:
		shutil.rmtree(folderpath)


def safe_zip_dir_extract(filepath):
	temp_dir = tempfile.TemporaryDirectory()
	#temp_dir = tempfile.mkdtemp()
	with zipfile.ZipFile(filepath, 'r') as zipObj:
		zipObj.extractall(path=temp_dir.name)
	#temp_dir.cleanup()
	return temp_dir


def safe_zip_dir_cleanup(temp_dir):
	temp_dir.cleanup()

def parse_wearable_filepath_info(filepath):
	split_str = '_'

	path_name_extension = fileparts(filepath)

	name = path_name_extension[1]
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


def read_edf_to_raw(filepath, preload=True, format="zmax_edf"):
	path_name_extension = fileparts(filepath)
	if (path_name_extension[2]).lower() != ".edf":
		warnings.warn("The filepath " + filepath + " does not seem to be an EDF file.")
	raw = None
	if format == "zmax_edf":

		"""
		This reader is largely similar to the one for edf but gets and assembles all the EDFs in a folder if they are in the zmax data format
		"""
		path_name_extension = fileparts(filepath)
		path = path_name_extension[0]
		check_channel_filenames = ['BATT', 'BODY TEMP', 'dX', 'dY', 'dZ', 'EEG L', 'EEG R', 'LIGHT', 'NASAL L', 'NASAL R', 'NOISE', 'OXY_DARK_AC', 'OXY_DARK_DC', 'OXY_IR_AC', 'OXY_IR_DC', 'OXY_R_AC', 'OXY_R_DC', 'RSSI']
		raw_avail_list = []
		channel_avail_list = []

		for iCh,name in enumerate(check_channel_filenames):
			checkname = path + os.sep + name + '.edf'
			if os.path.isfile(checkname):
				raw_avail_list.append(read_edf_to_raw(checkname, format="edf"))
				channel_avail_list.append(check_channel_filenames[iCh])
		print("zmax edf channels found:")
		print(channel_avail_list)

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

def edfWriteAnnotation(edfWriter, onset_in_seconds, duration_in_seconds, description, str_format='utf-8'):
	edfWriter.writeAnnotation(onset_in_seconds, duration_in_seconds, description, str_format)


def write_raw_to_edf(raw, filepath, format="zmax_edf"):
	path_name_extension = fileparts(filepath)
	if (path_name_extension[2]).lower() != ".edf":
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

def read_edf_to_raw_zipped(filepath, format="zmax_edf"):
	temp_dir = safe_zip_dir_extract(filepath)
	raw = None
	if format == "zmax_edf":
		raw = read_edf_to_raw(temp_dir.name + os.sep + "EEG L.edf", format=format)
	elif format == "edf":
		fileendings = ('*.edf', '*.EDF')
		filepath_list_edfs = []
		for fileending in fileendings:
			filepath_list_edfs.extend(glob.glob(temp_dir.name + os.sep + fileending,recursive=True))
		if filepath_list_edfs:
			raw = read_edf_to_raw(filepath_list_edfs[0], format=format)
	safe_zip_dir_cleanup(temp_dir)
	return raw


def write_raw_to_edf_zipped(raw, zippath, format="zmax_edf", compresslevel=6):
	temp_dir = tempfile.TemporaryDirectory()
	filepath = temp_dir.name + os.sep + fileparts(zippath)[1] + '.edf'
	write_raw_to_edf(raw, filepath, format)
	zip_directory(temp_dir.name, zippath, deletefolder=True, compresslevel=compresslevel)
	safe_zip_dir_cleanup(temp_dir)
	return zippath


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


def raw_detect_heart_rate_PPG(raw, ppg_channel):
	"""
	Detect the PPG artifacts in heart rate
	Detect the heartrate and outlier heart rates, 
	Output the heartrate signal with inter RR intervals and timepoints and artifact periods annotated.
	Optionally add to the raw data as a new channel with nans where there is not heart rate detected or artifactious
	"""


def interpolate_heart_rate(raw, ppg_channel):
	"""
	"""


def check_zmax_integrity():
	"""
	#Check if all the files are in order of the files
	#Are there any files missing in the dates
	#Check if a default date is present after a series of recordings or already starting with the first
	#Exlude files that are just very short recordings if the number of recordings is XXX
	#Optionally look for low Voltage in Battery (some lower threshold that was crossed to mark some forced shuttoff
    """


def find_wearable_files(parentdirpath, wearable):
	"""
	finds all the wearable data from different wearables in the HB file structure given the parent path to the subject files
	:param wearable:
	:return:
	"""
	"""
	:param wearable:
	:return:
	"""
	filepath_list = []
	if wearable == 'zmax':
		wearable = 'zmx'
	elif wearable == 'empathica':
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


def parse_wearable_data_write_csv(parentdirpath, filepath_csv_out, device='all'):

	filepath_list = find_wearable_files(parentdirpath, device)

	with open(filepath_csv_out, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
		writer.writerow(['subject_id', 'filepath', 'period', 'datatype', 'device_wearable', 'session'])
		for filepath in filepath_list:
			info = parse_wearable_filepath_info(filepath)
			writer.writerow([info['subject_id'], info['filepath'], info['period'], info['datatype'], info['device_wearable'], info['session']])


def parse_wearable_data_with_csv_annotate_datetimes(parentdirpath, filepath_csv_in, filepath_csv_out, device='all'):

	with open(filepath_csv_in, 'r', newline='') as csvfile:
		with open(filepath_csv_out, 'w', newline='') as csvfile2:
			reader = csv.reader(csvfile, delimiter=',')
			next(reader, None)  # skip the header of the read in csv

			writer = csv.writer(csvfile2, delimiter=',', quoting=csv.QUOTE_NONE)
			writer.writerow(['subject_id', 'filepath', 'period', 'datatype', 'device_wearable', 'session', 'rec_start_datetime', 'rec_stop_datetime', 'rec_duration_datetime', 'sampling_rate_max_Hz'])

			for row in reader:
				subject_id = row[0]
				filepath = row[1]
				period = row[2]
				datatype = row[3]
				device_wearable = row[4]
				session = row[5]

				rec_start_datetime = 'unretrieved'
				rec_stop_datetime = 'unretrieved'
				rec_duration_datetime = 'unretrieved'
				sampling_rate_max_Hz = 'unretrieved'

				try:
					if device_wearable == 'zmx':
						if session in ["1", "2", "3", "4", "5", "6", "7", "8"]:
							filepath_full = parentdirpath + os.sep + filepath
							raw = read_edf_to_raw_zipped(filepath_full, format="zmax_edf")
							rec_start_datetime = raw.info['meas_date']
							rec_stop_datetime = rec_start_datetime + datetime.timedelta(seconds=(raw._last_time - raw._first_time))
							rec_duration_datetime = datetime.timedelta(seconds=(raw._last_time - raw._first_time))
							sampling_rate_max_Hz = raw.info['sfreq']

					elif device_wearable == 'emp': # TODO: Rayyan add some info for reach file
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

				writer.writerow([subject_id, filepath, period, datatype, device_wearable, session, rec_start_datetime, rec_stop_datetime, rec_duration_datetime, sampling_rate_max_Hz])


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
	#--tests--#

	"""
	raw = read_edf_to_raw_zipped("Y:/HB/data/test_data_zmax/FW.zip", format="zmax_edf")
	write_raw_to_edf(raw, "Y:/HB/data/test_data_zmax/FW.merged.edf", format="zmax_edf")  # treat as a speacial zmax read EDF for export
	write_raw_to_edf_zipped(raw, "Y:/HB/data/test_data_zmax/FW.merged.zip", format="zmax_edf") # treat as a speacial zmax read EDF for export
	raw_reread = read_edf_to_raw_zipped("Y:/HB/data/test_data_zmax/FW.merged.zip", format="edf")
	write_raw_to_edf_zipped(raw, "Y:/HB/data/test_data_zmax/FW.merged.reread.zip", format="zmax_edf") # treat as a speacial zmax read EDF for export


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

	wearable_file_structure_annotation_csv = "wearable_file_structure_annotation.csv"
	wearable_file_structure_annotation_datetime_csv = "wearable_file_structure_datetime_annotation.csv"
	parentdirpath="Y:/HB/data/example_subjects/data"

	parse_wearable_data_write_csv(parentdirpath=parentdirpath,filepath_csv_out=wearable_file_structure_annotation_csv,device='zmax')
	parse_wearable_data_with_csv_annotate_datetimes(parentdirpath=parentdirpath,filepath_csv_in=wearable_file_structure_annotation_csv,filepath_csv_out=wearable_file_structure_annotation_datetime_csv,device='zmax')

	df = pandas.read_csv(wearable_file_structure_annotation_datetime_csv)
	df.iloc[[0]]
	print("tests finished")