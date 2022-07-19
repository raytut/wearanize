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
python.exe -m pip install pyinstaller
...

or to save the enironment package requirements (also unused)
python.exe -m pip freeze > requirements.txt

or use project specific used packages:
pip install pipreqs
pipreqs /path/to/this/project


"""

import mne
import numpy
import warnings
import os
import glob
import datetime
import pyedflib
import shutil
import argparse
import pathlib

import sys
if sys.version_info >= (3, 6):
	import zipfile
	from zipfile import ZipFile
else:
	import zipfile36 as zipfile

import tempfile
import traceback
import subprocess
import logging
import statistics

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
def raw_prolong_constant(raw, to_n_samples, contant=0, prepend=False):
	append_samples = to_n_samples - raw.n_times

	#raw_append = mne.io.RawEDF(numpy.full([raw._data.shape[0], append_samples], contant), info=raw.info)
	raw_append = raw.copy()
	raw_append.crop(tmin=raw_append.times[0], tmax=raw_append.times[append_samples-1], include_tmax=True, verbose=False)
	raw_append._data = numpy.full([raw_append._data.shape[0], append_samples], contant)
	if prepend:
		raw_append.append([raw])
		return raw_append
	else:
		raw.append([raw_append])
		return raw
		#mne.concatenate_raws([raw, raw_append])

# =============================================================================
#
# =============================================================================
def read_edf_to_raw(filepath, preload=True, format="zmax_edf", zmax_ppgparser=False, zmax_ppgparser_exe_path=None, zmax_ppgparser_timeout=None, drop_zmax=['BODY TEMP', 'LIGHT', 'NASAL L', 'NASAL R', 'NOISE', 'OXY_DARK_AC', 'OXY_DARK_DC', 'OXY_R_AC', 'OXY_R_DC', 'RSSI', 'PARSED_NASAL R', 'PARSED_NASAL L', 'PARSED_OXY_R_AC', 'PARSED_HR_r', 'PARSED_HR_r_strength']):
	path, name, extension = fileparts(filepath)
	if (extension).lower() != ".edf":
		warnings.warn("The filepath " + filepath + " does not seem to be an EDF file.")
	raw = None
	if format == "zmax_edf":

		"""
		This reader is largely similar to the one for edf but gets and assembles all the EDFs in a folder if they are in the zmax data format
		"""
		path, name, extension = fileparts(filepath)
		#check_channel_filenames = ['BATT', 'BODY TEMP', 'dX', 'dY', 'dZ', 'EEG L', 'EEG R', 'LIGHT', 'NASAL L', 'NASAL R', 'NOISE', 'OXY_DARK_AC', 'OXY_DARK_DC', 'OXY_IR_AC', 'OXY_IR_DC', 'OXY_R_AC', 'OXY_R_DC', 'RSSI']
		check_channel_filenames = ['BATT', 'BODY TEMP', 'dX', 'dY', 'dZ', 'EEG L', 'EEG R', 'LIGHT', 'NASAL L', 'NASAL R', 'NOISE', 'OXY_DARK_AC', 'OXY_DARK_DC', 'OXY_IR_AC', 'OXY_IR_DC', 'OXY_R_AC', 'OXY_R_DC', 'RSSI', 'PARSED_NASAL R', 'PARSED_OXY_IR_AC', 'PARSED_NASAL L', 'PARSED_HR_r', 'PARSED_HR_r_strength', 'PARSED_OXY_R_AC', 'PARSED_HR_ir', 'PARSED_HR_ir_strength']
		raw_avail_list = []
		channel_avail_list = []
		channel_read_list = []
		for iCh, name in enumerate(check_channel_filenames):
			checkname = path + os.sep + name + '.edf'
			if os.path.isfile(checkname):
				channel_avail_list.append(name)

		if zmax_ppgparser and zmax_ppgparser_exe_path is not None:
			print('ATTEMPT to reparse heart signals using the PPGParser' + filepath)
			exec_string =  "\"" + zmax_ppgparser_exe_path + "\""
			for iCh, name in enumerate(channel_avail_list):
				addfilepath = path + os.sep + name + '.edf'
				exec_string = exec_string + " " + "\"" + addfilepath + "\""
			try:
				subprocess.run(exec_string, shell=False, timeout=zmax_ppgparser_timeout)
			except:
				print(traceback.format_exc())
				print('FAILED to reparse' + filepath)
			channel_avail_list = []
			for iCh, name in enumerate(check_channel_filenames):
				checkname = path + os.sep + name + '.edf'
				if os.path.isfile(checkname):
					channel_avail_list.append(name)

		for iCh, name in enumerate(channel_avail_list):
			if not name in drop_zmax:
				readfilepath = path + os.sep + name + '.edf'
				try:
					raw_read = read_edf_to_raw(readfilepath, format="edf")
					if 'PARSED_' in name:
						raw_read.rename_channels({raw_read.info["ch_names"][0]: name})
					raw_avail_list.append(raw_read)
					channel_read_list.append(name)
				except Exception:
					print(traceback.format_exc())
					print('FAILED TO read in channel: ' + check_channel_filenames[iCh])

		print("zmax edf channels found:")
		print(channel_avail_list)
		print("zmax edf channels read in:")
		print(channel_read_list)

		if raw_avail_list[0] is not None:
			nSamples_should = raw_avail_list[0].n_times

		for i, r in enumerate(raw_avail_list):
			if r is not None:
				sfreq_temp = r.info['sfreq']
				if sfreq_temp != 256.0:
					raw_avail_list[i] = r.resample(256.0)
					nSamples = raw_avail_list[i].n_times
					if nSamples < nSamples_should:
						raw_avail_list[i] = raw_prolong_constant(raw_avail_list[i], nSamples_should, contant=0, prepend=True)

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
		raw = mne.io.read_raw_edf(filepath, preload=preload)
	return raw

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
		channel_dimensions_zmax = {'BATT': 'V', 'BODY TEMP': "C", 'dX': "g", 'dY': "g", 'dZ': "g", 'EEG L': "uV", 'EEG R': "uV", 'LIGHT': "", 'NASAL L': "", 'NASAL R': "", 'NOISE': "", 'OXY_DARK_AC': "", 'OXY_DARK_DC': "", 'OXY_IR_AC': "", 'OXY_IR_DC': "", 'OXY_R_AC': "", 'OXY_R_DC': "", 'RSSI': "", 'PARSED_NASAL R': "", 'PARSED_OXY_IR_AC': "", 'PARSED_NASAL L': "", 'PARSED_HR_r': "bpm", 'PARSED_HR_r_strength': "", 'PARSED_OXY_R_AC': "", 'PARSED_HR_ir': "bpm", 'PARSED_HR_ir_strength': ""}

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
			try:
				dimension = channel_dimensions_zmax[ch_name] #'uV'
			except KeyError:
				dimension = ""
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
def read_edf_to_raw_zipped(filepath, format="zmax_edf", zmax_ppgparser=False, zmax_ppgparser_exe_path=None, zmax_ppgparser_timeout=None, drop_zmax=['BODY TEMP', 'LIGHT', 'NASAL L', 'NASAL R', 'NOISE', 'OXY_DARK_AC', 'OXY_DARK_DC', 'OXY_R_AC', 'OXY_R_DC', 'RSSI', 'PARSED_NASAL R', 'PARSED_NASAL L', 'PARSED_OXY_R_AC', 'PARSED_HR_r', 'PARSED_HR_r_strength']):
	temp_dir = safe_zip_dir_extract(filepath)
	raw = None
	if format == "zmax_edf":
		raw = read_edf_to_raw(temp_dir.name + os.sep + "EEG L.edf", format=format, zmax_ppgparser=zmax_ppgparser, zmax_ppgparser_exe_path=zmax_ppgparser_exe_path, zmax_ppgparser_timeout=zmax_ppgparser_timeout, drop_zmax=drop_zmax)
	elif format == "edf":
		fileendings = ('*.edf', '*.EDF')
		filepath_list_edfs = []
		for fileending in fileendings:
			filepath_list_edfs.extend(glob.glob(temp_dir.name + os.sep + fileending,recursive=True))
		if filepath_list_edfs:
			raw = read_edf_to_raw(filepath_list_edfs[0], format=format, zmax_ppgparser=zmax_ppgparser, zmax_ppgparser_exe_path=zmax_ppgparser_exe_path, zmax_ppgparser_timeout=zmax_ppgparser_timeout)
	safe_zip_dir_cleanup(temp_dir)
	return raw

# =============================================================================
#
# =============================================================================
def write_raw_to_edf_zipped(raw, zippath, edf_filename=None, format="zmax_edf", compresslevel=6):
	temp_dir = tempfile.TemporaryDirectory()
	if edf_filename is None:
		filepath = temp_dir.name + os.sep + fileparts(zippath)[1] + '.edf'
	else:
		filepath = temp_dir.name + os.sep + fileparts(edf_filename)[1] + '.edf'
	write_raw_to_edf(raw, filepath, format)
	zip_directory(temp_dir.name, zippath, deletefolder=True, compresslevel=compresslevel)
	safe_zip_dir_cleanup(temp_dir)
	return zippath


def nullable_string(val):
	if not val:
		return None
	return val

def dir_path_or_file(pathstring):
	pathstring = os.path.normpath(pathstring)
	if nullable_string(pathstring):
		if os.path.isdir(pathstring):
			return pathstring
		if os.path.isfile(pathstring):
			return pathstring
		else:
			print("'%s' is not a directory or file" % pathstring)
			raise NotADirectoryError(pathstring)
	return None

def dir_path(pathstring):
	pathstring = os.path.normpath(pathstring)
	if nullable_string(pathstring):
		if os.path.isdir(pathstring):
			return pathstring
		else:
			print("'%s' is not a directory" % pathstring)
			raise NotADirectoryError(pathstring)
	return None

def dir_path_new(pathstring):
	pathstring = os.path.normpath(pathstring)
	if nullable_string(pathstring):
		return pathstring
	return None

def file_path(pathstring):
	pathstring = os.path.normpath(pathstring)
	if nullable_string(pathstring):
		if os.path.isfile(pathstring):
			return pathstring
		else:
			print("'%s' is not a file" % pathstring)
			raise NotADirectoryError(pathstring)
	return None

# =============================================================================
# 
# =============================================================================
def find_zmax_files(parentdirpath, readzip=False, zipfile_match_string='', zipfile_nonmatch_string=''):
	"""
	finds all the zmax data from different wearables in the HB file structure given the parent path to the subject files
	:param wearable:
	:return:
	"""
	filepath_list = []
	if readzip:
		filepath_list = glob.glob(parentdirpath + os.sep + "**" + os.sep + "*.zip",recursive=True)

		if zipfile_match_string != '' or  (zipfile_match_string is None):
			for include_string in zipfile_match_string.split('|'):
				filepath_list = list(filter(lambda x: (include_string in fileparts(x)[1]), filepath_list))

		if zipfile_nonmatch_string != '' or  (zipfile_nonmatch_string is None):
			for exclude_string in zipfile_nonmatch_string.split('|'):
				filepath_list = list(filter(lambda x: (exclude_string not in fileparts(x)[1]), filepath_list))
	else:
		filepath_list = glob.glob(parentdirpath + os.sep + "**" + os.sep + "EEG L.edf",recursive=True)

	# # compatible with python versions < 3.10 remove the root_dir
	# for i, filepath in enumerate(filepath_list):
	# 	filepath_list[i] = filepath.replace(parentdirpath + os.sep,"")

	return filepath_list

def path_create(path, isFile=False):
	if isFile:
		path, name, extension = fileparts(path)
	if not os.path.exists(path):
		os.makedirs(path)

if __name__ == "__main__":
# Instantiate the argument parser
	parser = argparse.ArgumentParser(prog='zmax_edf_merge_converter.exe', description='This is useful software to reuse EDF from zmax to repackage the original exported EDFs and reparse them if necessary or zip them. Copyright 2022, Frederik D. Weber')

	# Required positional argument
	parser.add_argument('parent_dir_paths', type=dir_path_or_file,
					help='A path or multiple paths to the parent folder where the data is stored and converted from (and by default also converted to)', nargs='+')

	# Optional argument
	parser.add_argument('--write_redirection_path', type=dir_path_new,
					help='An optional path to redirect writing to a different parent folder (so to not accidentally overwrite other files). Original folder structure is keept in the subfolders.')

	# Switch
	parser.add_argument('--read_zip', action='store_true',
					help='Switch to indicate if the input edfs are zipped and end with .zip')

	# Optional argument
	parser.add_argument('--zipfile_match_string', type=str,
					help='An optional string to match the name of the zip files to search for. Use the pipe to separate different search/match strings, e.g. --zipfile_match_string=\"this|that\" will search for \"this\" and then for \"that\". If parent_dir_paths contains direct paths to .zip files this does not apply.')

	# Optional argument
	parser.add_argument('--zipfile_nonmatch_string', type=str,
					help='An optional string to NOT match (i.e. exclude or filter out) after all the zipfile_match_string zip files have been found. Use the pipe to separate different search/match strings, e.g. --zipfile_nonmatch_string=\"this|that\" will search for \"this\" and then for \"that\". If parent_dir_paths contains direct paths to .zip files this does not apply.')

	# Switch
	parser.add_argument('--zmax_ppgparser', action='store_true',
					help='Switch to indicate if ZMax PPGParser.exe is used to reparse some heart rate related channels. you also need to specify zmax_ppgparser_exe_path if it is not already in the current directory. This will take time to reprocess each data.')

	# Optional argument
	parser.add_argument('--zmax_ppgparser_exe_path', type=file_path,
					help='direct and full path to the ZMax PPGParser.exe in the Hypnodyne ZMax software folder')

	# Optional argument
	parser.add_argument('--write_name_postfix', type=str,
					help='file name post fix for the written files or directories. Default is \"_merged\"')

	# Optional argument
	parser.add_argument('--temp_file_postfix', type=str,
					help='file name post fix for the written files or directories that are not completely written yet. Default is \"_TEMP_\"')

	# Optional argument
	parser.add_argument('--zmax_ppgparser_timeout', type=float,
					help='An optional timeout to run the ZMax PPGParser.exe in seconds. If empty no timeout is used')

	# Switch
	parser.add_argument('--zmax_lite', action='store_true',
					help='Switch to indicate if the device is a ZMax lite version and not all channels have to be included')

	# Switch
	parser.add_argument('--no_overwrite', action='store_true',
					help='Switch to indicate if files should be overwritten if existent')

	# Switch
	parser.add_argument('--exclude_empty_channels', action='store_true',
					help='Switch to indicate if channels that are constant (i.e. empty and likely not recorded) should be excluded/dropped. Requires some more computation time but saves space in case it is not zipped.')

	# Switch
	parser.add_argument('--write_zip', action='store_true',
					help='Switch to indicate if the output edfs should be zipped in one .zip file')





	args = parser.parse_args()

	parent_dir_paths = [pathlib.Path().resolve()] # the current working directory
	if args.parent_dir_paths is not None:
		parent_dir_paths = args.parent_dir_paths

	write_redirection_path = None
	if args.write_redirection_path is not None:
		write_redirection_path = args.write_redirection_path

	exclude_empty_channels = False
	if args.exclude_empty_channels is not None:
		exclude_empty_channels = args.exclude_empty_channels

	isliteversion = False
	if args.zmax_lite is not None:
		isliteversion = args.zmax_lite
	
	write_zip = False
	if args.write_zip is not None:
		write_zip = args.write_zip

	read_zip = False
	if args.read_zip is not None:
		read_zip = args.read_zip

	zipfile_match_string = ''
	if args.zipfile_match_string is not None:
		zipfile_match_string = args.zipfile_match_string

	zipfile_nonmatch_string = ''
	if args.zipfile_nonmatch_string is not None:
		zipfile_nonmatch_string = args.zipfile_nonmatch_string

	zmax_ppgparser = False
	if args.zmax_ppgparser is not None:
		zmax_ppgparser = args.zmax_ppgparser

	no_overwrite = False
	if args.no_overwrite is not None:
		no_overwrite = args.no_overwrite

	zmax_ppgparser_exe_path = 'PPGParser.exe' # in the current working directory
	if args.zmax_ppgparser_exe_path is not None:
		zmax_ppgparser_exe_path = args.zmax_ppgparser_exe_path

	zmax_ppgparser_timeout = None # in the current working directory
	if args.zmax_ppgparser_timeout is not None:
		zmax_ppgparser_timeout = args.zmax_ppgparser_timeout

	write_name_postfix = "_merged"
	if args.write_name_postfix is not None:
		write_name_postfix = args.write_name_postfix

	temp_file_postfix = "_TEMP_"
	if args.temp_file_postfix is not None:
		temp_file_postfix = args.temp_file_postfix

	#if len(sys.argv) != 3:
	#	print('expecting path to a parent folders with zmax edfs converted from HDrecorder as the only argument')
	#	exit(0)

	#parentdirpath = sys.argv[1]
	for parentdirpath in parent_dir_paths:
		read_zip_temp = read_zip

		if os.path.isfile(parentdirpath):
			p, n, e = fileparts(parentdirpath)
			if e.lower() == ".zip":
				read_zip_temp = True
				filepath_list = [parentdirpath]
			else:
				filepath_list = [] # do not process
				#parentdirpath = p
			#if e.lower() != ".edf":
			#	read_zip_temp = False
			#	parentdirpath = p

		else:
			try:
				parentdirpath = dir_path(parentdirpath)
			except NotADirectoryError:
				print("argument '%s' is not a directory" %parentdirpath)
				continue #exit(0)

			if parentdirpath is None:
				print("argument '%s' is not parsable" %parentdirpath)
				continue #exit(0)

			print("Finding file paths...")
			filepath_list = find_zmax_files(parentdirpath, readzip=read_zip_temp, zipfile_match_string=zipfile_match_string, zipfile_nonmatch_string=zipfile_nonmatch_string)

		print("FOUND %d matching file paths " % len(filepath_list))
		for iFn, fn in enumerate(filepath_list):
			print("%d: %s" % (iFn, fn))

		if len(filepath_list) < 1:
			print("no zmax edf files found")
			#exit(0)

		number_of_conversions = len(filepath_list)
		for i, filepath in enumerate(filepath_list):
			print("PROCESSING %d of %d: '%s' " % (i+1, number_of_conversions, filepath))
			path, name, extension = fileparts(filepath)
			parentfoldername = os.path.basename(path)
			pathup, nametmp, extensiontmp = fileparts(path)
			drop_channels = []
			if isliteversion:
				drop_channels = ['BODY TEMP', 'LIGHT', 'NASAL L', 'NASAL R', 'NOISE', 'OXY_DARK_AC', 'OXY_DARK_DC', 'OXY_R_AC', 'OXY_R_DC', 'RSSI', 'PARSED_NASAL R', 'PARSED_NASAL L', 'PARSED_OXY_R_AC', 'PARSED_HR_r', 'PARSED_HR_r_strength']
			try:

				if read_zip_temp:
					export_filepath = path + os.sep + name + write_name_postfix
				else:
					export_filepath = pathup + os.sep +  parentfoldername + write_name_postfix

				if write_redirection_path is not None:
					indFound = export_filepath.find(parentdirpath)
					if indFound >= 0:
						export_filepath = write_redirection_path + export_filepath[(indFound+len(parentdirpath)):]
						#if write_zip:
						path_create(export_filepath,isFile=True)
						#else:
						#	path_create(export_filepath,isFile=True)

				export_filepath_unfinished = export_filepath + temp_file_postfix

				if write_zip:
					export_filepath_final_to_rename = export_filepath_unfinished + ".zip"
				else:
					export_filepath_final_to_rename = export_filepath_unfinished + ".edf"

				export_filepath_final = export_filepath_final_to_rename.replace(temp_file_postfix,'')

				if no_overwrite:
					if os.path.exists(export_filepath_final):
						print('skipping file: %s' % export_filepath_final)
						continue

				#reading
				if read_zip_temp:
					raw = read_edf_to_raw_zipped(filepath, format="zmax_edf", zmax_ppgparser=zmax_ppgparser, zmax_ppgparser_exe_path=zmax_ppgparser_exe_path, zmax_ppgparser_timeout=zmax_ppgparser_timeout, drop_zmax=drop_channels)
				else:
					raw = read_edf_to_raw(filepath, format="zmax_edf", zmax_ppgparser=zmax_ppgparser, zmax_ppgparser_exe_path=zmax_ppgparser_exe_path, zmax_ppgparser_timeout=zmax_ppgparser_timeout, drop_zmax = drop_channels)
				print("READ %d of %d: '%s' " % (i+1, number_of_conversions, filepath))

				if exclude_empty_channels:
					flat_channel_names = []
					for iCh, ch_name in enumerate(raw.info['ch_names']):
						ch_name = raw.info['ch_names'][iCh]
						nNotFlat = numpy.count_nonzero(raw._data[iCh]-statistics.median(raw._data[iCh])) # this is fastest so far
						if nNotFlat <= 10:
							flat_channel_names.append(ch_name)
					raw.drop_channels(flat_channel_names)

				#writing
				print("Attempting to write %d of %d: '%s' " % (i+1, number_of_conversions, export_filepath_final))
				# check again just before writing
				if no_overwrite:
					if os.path.exists(export_filepath_final):
						print('skipping file: %s' % export_filepath_final)
						continue
				if write_zip:
					export_filepath_final_to_rename_2 = write_raw_to_edf_zipped(raw, export_filepath_final_to_rename, edf_filename=export_filepath_final, format="zmax_edf") # treat as a speacial zmax read EDF for export
				else:
					export_filepath_final_to_rename_2 = write_raw_to_edf(raw, export_filepath_final_to_rename, format="zmax_edf")  # treat as a speacial zmax read EDF for export

				try:
					# check again just before writing
					if no_overwrite:
						os.rename(export_filepath_final_to_rename_2, export_filepath_final)
					else:
						if os.path.exists(export_filepath_final):
							try:
								os.remove(export_filepath_final)
							except FileNotFoundError:
								pass
						shutil.move(export_filepath_final_to_rename_2, export_filepath_final)
					print("WROTE %d of %d: '%s' " % (i+1, number_of_conversions, export_filepath_final))
				except:
					print('FAILED TO RENAME FINAL FILE %s FROM TEMPORARY FILE: %s' % (export_filepath_final, export_filepath_final_to_rename_2))
					print(traceback.format_exc())
					#finally remove the temporary file if exists
					try:
						try:
							os.remove(export_filepath_final_to_rename_2)
						except FileNotFoundError:
							pass
					except:
						print('FAILED TO DELETE THE LEFT TEMPORARY FILE: %s' % export_filepath_final_to_rename_2)
						print(traceback.format_exc())

			except Exception as e:
				print(traceback.format_exc())
				print("FAILED %d of %d: '%s' " % (i+1, number_of_conversions, filepath))
