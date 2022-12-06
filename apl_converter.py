

# imports
import os
import glob
import numpy
import pandas
import warnings
import mne
import datetime
import pytz
import xlrd
from collections import namedtuple

## from other scripts within
from zmax_edf_merge_converter import fileparts, read_edf_to_raw, read_edf_to_raw_zipped
from e4_converter import read_e4_to_raw

# functions
_Meta = namedtuple('Meta', [
	'firmware', 'bitdepth', 'resolution', 'hz', 'axes',
	'start_datetime', 'stop_datetime', 'duration',
	'start_condition', 'stop_condition', 'file_code', 'device_id'
])

class Meta(_Meta):
	"""
	A namedtuple with fields for the activPAL raw data's metadata.
	Parameters
	----------
	firmware : int
	bitdepth : int
	resolution : int
	hz : int
	axes : int
	start_datetime : datetime.datetime
	stop_datetime : datetime.datetime
	duration : datetime.timedelta
	start_condition : str
	stop_condition : str
	file_code : str
	device_id : int
	"""
	__slots__ = ()


class ActivpalData(object):
	"""
	An object to wrap activPAL data.
	Methods
	-------
	TODO
	See Also
	--------
	load_activpal_data : Returns the data from an activPAL data file as a
		tuple (metadata, signals).
	"""

	def __init__(self, file_path):
		"""
		Create an instance of an activpal_data object.
		Parameters
		----------
		file_path : str
			The path to an activPAL raw data file.
		"""
		data = load_activpal_data(file_path)
		self._metadata = data[0]
		data_g = (numpy.array(data[1], dtype=numpy.float64, order='F') - 127) / 63
		interval = pandas.tseries.offsets.Milli() * (1000 / data[0].hz)
		ind = pandas.date_range(data[0].start_datetime, periods=len(data[1]),
								freq=interval)
		self._signals = pandas.DataFrame(data_g, columns=['x', 'y', 'z'], index=ind)

	@property
	def metadata(self):
		"""namedtuple : The information extracted from the files header."""
		return self._metadata

	@property
	def signals(self):
		"""pandas.DataFrame : The sensor signals."""
		return self._signals.copy()

	@property
	def data(self):
		"""pandas.DataFrame : Depricated - use signals."""
		warnings.warning('activpal_data.data is depricated use activpal_data.signals')
		return self.signals

	@property
	def timestamps(self):
		"""pandas.DatetimeIndex : The timestams of the signals."""
		return self.signals.index

	@property
	def x(self):
		"""pandas.Series : The signal from the x axis."""
		if 'x' not in self._signals.columns:
			raise AttributeError('activpal_data property X no longer exists.\
								 The signals must have been interfered with.')
		return self._signals['x'].copy()

	@property
	def y(self):
		"""pandas.Series : The signal from the y axis."""
		if 'y' not in self._signals.columns:
			raise AttributeError('activpal_data property Y no longer exists.\
								 The signals must have been interfered with.')
		return self._signals['y'].copy()

	@property
	def z(self):
		"""pandas.Series : The signal from the z axis."""
		if 'z' not in self._signals.columns:
			raise AttributeError('activpal_data property Z no longer exists.\
								 The signals must have been interfered with.')
		return self._signals['z'].copy()

	@property
	def rss(self):
		"""pandas.Series : The Root Sum of Squares of the x, y, z axes."""
		if 'rss' not in self._signals.columns:
			sqr = numpy.square(self._signals[['x', 'y', 'z']])
			sumsqr = numpy.sum(sqr, axis=1)
			self._signals['rss'] = numpy.sqrt(sumsqr)
		return self._signals['rss'].copy()


def change_file_code(filepath, new_code):
	"""
	Modifiy the file code in the header of an activPAL raw data file.
	Parameters
	----------
	file_path : str
		The path to an activPAL raw data file.
	new_code : str
		The upto 8 char string which the file code should be changed to.
	"""
	if not os.path.isfile(filepath):
		raise FileNotFoundError(errno.ENOENT,
								os.strerror(errno.ENOENT), filepath)
	# Chech new code is str or convertable to str
	if not isinstance(new_code, str):
		str(new_code)
	# Check the str is 8 chars or less
	assert (len(new_code) <= 8), 'New file code longer than 8 characters'
	# Format bytes to write to file
	new_bytes = bytes(new_code, 'ascii').ljust(8, b'\x00')
	# Write to file
	with open(filepath, 'r+b') as f:
		f.seek(512, 0)
		f.write(new_bytes)


def extract_apl_metadata_from_file(filepath):
	"""
	Return a Meta object with the metadata from the given activPAL data file.
	Parameters
	----------
	file_path : str
		The path to an activPAL raw data file.
	Returns
	-------
	meta : uos_activpal.io.raw.Meta
		The information extracted from the files header.
	See Also
	--------
	ActivpalData : An object to wrap activPAL data.
	load_activpal_data : Returns the data from an activPAL data file.
	change_file_code : Modifies the file code of an activPAL raw data file.
	extract_accelerometer_data : Extracts the signals from an activPAL raw data
		file body.
	"""
	header = numpy.fromfile(filepath, dtype=numpy.uint8, count=1024, sep='')
	return extract_apl_metadata(header)


def extract_apl_metadata(header):
	"""
	Return a Meta object with the metadata from the given uint8 array.
	Parameters
	----------
	header : numpy.uint8
		The header section of an activPAL raw data file.
	Returns
	-------
	meta : uos_activpal.io.raw.Meta
		The information extracted from the files header in a structured format.
	See Also
	--------
	extract_metadata_from_file : Returns a Meta object with the metadata from
		the given activPAL data file.
	ActivpalData : An object to wrap activPAL data.
	load_activpal_data : Returns the data from an activPAL data file.
	change_file_code : Modifies the file code of an activPAL raw data file.
	extract_accelerometer_data : Extracts the signals from an activPAL raw data
		file body.
	"""
	firmware = header[39] * 255 + header[17]  # Should it be 256?

	if header[38] < 128:
		bitDepth = 8
		resolution_byte = header[38]
	else:
		bitDepth = 10
		resolution_byte = header[38] - 128

	resolution_map = {0: 2, 1: 4, 2: 8}
	resolution = resolution_map.get(resolution_byte)

	hz = header[35]

	axes_map = {0: 3, 1: 1}
	axes = axes_map.get(header[280])

	start_datetime = datetime.datetime(header[261] + 2000, header[260], header[259],
									   header[256], header[257], header[258])

	stop_datetime = datetime.datetime(header[267] + 2000, header[266], header[265],
									  header[262], header[263], header[264])

	duration = stop_datetime - start_datetime
	# duration = '{:.3f} days'.format(duration.days + duration.seconds / 86400)

	start_condition_map = {0: 'Trigger', 1: 'Immediately', 2: 'Set Time'}
	start_condition = start_condition_map.get(header[268])

	stop_condition_map = {0: 'Memory Full', 3: 'Low Battery', 64: 'USB',
						  128: 'Programmed Time'}
	stop_condition = stop_condition_map.get(header[275])

	file_code = ''.join([(chr(x) if not x == 0 else '') for x in header[512:520]])
	# Header 10 is the year code, old device use 12 for 2012 newer ones use 4
	# for 2014. Device ID needs first digit to be the last digit of the year
	# % means mod, anything mod 10 returns the last digit
	device_id = ((header[10] % 10) * 100000 + header[14] * 10000 +
				 header[40] * 4096 + header[11] * 256 + header[12] * 16 +
				 header[13])

	return Meta(firmware, bitDepth, resolution, hz, axes,
				start_datetime, stop_datetime, duration,
				start_condition, stop_condition, file_code, device_id)


def extract_apl_accelerometer_data(body, firmware, datx):
	"""
	Return a numpyndarray with the signals from the given uint8 array.
	Parameters
	----------
	body : numpy.ndarray, dype=numpy.uint8
		The body section of an activPAL raw data file.
	firmware : int
		The firmware version used to create the file from which body came.
	datx : bool
		Whether the source file had extension .datx (True) or .dat (False).
	Returns
	-------
	signals : numpy.ndarray
		The signals extracted from body in a column array.
	See Also
	--------
	extract_metadata_from_file : Returns a Meta object with the metadata from
		the given activPAL data file.
	ActivpalData : An object to wrap activPAL data.
	load_activpal_data : Returns the data from an activPAL data file.
	"""
	length = len(body)
	max_rows = int(numpy.floor(length / 3) * 255)
	signals = numpy.zeros([max_rows, 3], dtype=numpy.uint8, order='C')

	adjust_nduplicates = firmware < 218

	row = 0
	for i in range(0, length - 1, 3):
		x = body[i]
		y = body[i + 1]
		z = body[i + 2]

		if datx:
			tail = (x == 116 and y == 97 and z == 105 and body[i + 3] == 108)
		else:
			# TODO change this to use _old_tail_check?
			# Would ^ slow it down - how would numba handle it?
			tail = (x == 0 and y == 0 and z > 0 and
					body[i + 3] == 0 and body[i + 4] == 0 and
					body[i + 5] > 0 and body[i + 6] > 0 and body[i + 7] == 0)

		two54 = (x == 254 and y == 254 and z == 254)
		two55 = (x == 255 and y == 255 and z == 255)
		invalid = two54 or two55

		compressed = (x == 0 and y == 0)

		if tail:
			signals = signals[:row]
			break
		elif invalid:
			signals_prev = signals[row - 1]
			signals[row] = signals_prev
			row += 1
		elif compressed:
			signals_prev = signals[row - 1]
			if adjust_nduplicates:
				nduplicates = z + 1
			else:
				nduplicates = z
			for r in range(nduplicates):
				signals[row] = signals_prev
				row += 1
		else:
			signals[row, 0] = x
			signals[row, 1] = y
			signals[row, 2] = z
			row += 1
	return signals


def load_activpal_data(filepath):
	"""
	Return the data from an activPAL data file as (metadata, signals).
	Parameters
	----------
	file_path : str
		The path to an activPAL raw data file.
	Returns
	-------
	metadata : uos_activpal.io.raw.Meta
		A namedtuple containing information extracted from the files header.
	signals : numpy.ndarray
		An array with a column for each axis of the device.
	See Also
	--------
	ActivpalData : An object to wrap activPAL data.
	"""
	file_ext = os.path.splitext(filepath)[1]
	if file_ext == '.datx':
		header_end = 1024
	elif file_ext == '.dat':
		header_end = 1023
	else:
		raise ValueError(''.join(('Unknown file extension "', file_ext,
								  '" for file "', filepath, '"')))

	file_content = numpy.fromfile(filepath, dtype=numpy.uint8, count=-1, sep='')

	# compression = file_content[36]  # True(1) / False(0)

	metadata = extract_apl_metadata(file_content[:header_end])
	signals = extract_apl_accelerometer_data(file_content[header_end:],
											 metadata.firmware, file_ext == '.datx')
	return (metadata, signals)


def read_apl_to_raw(filepath):
	# TODO: Sort out memory issue (needs 50+gb now)
	meta, raw = load_activpal_data(filepath)
	info = mne.create_info(["acc_x", 'acc_y', 'acc_z'], sfreq=meta[3])
	raw = raw.transpose()
	raw = mne.io.RawArray(raw, info, first_samp=0, verbose=True)
	# Set time
	tz = pytz.timezone('Europe/Amsterdam')
	timestamp = meta[5]
	timestamp = tz.localize(timestamp)
	raw.set_meas_date(timestamp.timestamp())
	return (raw)


def apl_window_to_raw(filepath, wearable, buffer_seconds=0):
	# get wearable data in raw
	if wearable == 'zmx':
		raw_source = read_edf_to_raw_zipped(filepath)
	elif wearable == 'zmx-merge':
		raw_source = read_edf_to_raw_zipped(filepath, format='edf')
	elif wearable == 'emp':
		raw_source = read_e4_to_raw(filepath)

	# locate and read in apl data into df
	path, sub, ext = fileparts(filepath)
	meta, apl_data = load_activpal_data(glob.glob(path + "/*apl.datx")[0])
	apl_df = pandas.DataFrame(apl_data)

	# get start times for apl
	apl_start_ts = meta[5]
	tz = pytz.timezone('Europe/Amsterdam')
	apl_start_ts = tz.localize(apl_start_ts)
	# get start times for source
	raw_source_start_ts = raw_source.info['meas_date']
	raw_source_duration_s = (raw_source.last_samp) * (1 / raw_source.info['sfreq'])
	raw_source_end_ts = raw_source_start_ts + datetime.timedelta(seconds=raw_source_duration_s)

	# window apl time
	time_diff_start_s = (apl_start_ts - raw_source_start_ts).total_seconds()
	time_diff_end_s = (apl_start_ts - raw_source_end_ts).total_seconds()
	apl_start_samps = int(abs(time_diff_start_s * meta[3]) - (buffer_seconds * meta[3]))
	apl_end_samps = int(abs(time_diff_end_s * meta[3]) + (buffer_seconds * meta[3]))
	apl_df = apl_df.iloc[apl_start_samps:apl_end_samps, ]

	# convert to mne raw
	info = mne.create_info(["apl_acc_x", 'apl_acc_y', 'apl_acc_z'], sfreq=meta[3])
	apl_np = apl_df.to_numpy().transpose()
	raw_apl = mne.io.RawArray(apl_np, info, first_samp=0, verbose=True)
	raw_apl.set_meas_date(raw_source_start_ts - datetime.timedelta(seconds=buffer_seconds))

	return raw_apl


def read_apl_event_to_raw(filepath, resample_Hz=1):

	# Read file
	df_apl = pandas.read_csv(filepath)
	try:
		df_apl.APDatetimevar = df_apl.APDatetimevar.apply(xlrd.xldate_as_datetime, convert_dtype=True, datemode=0)
		df_apl = df_apl.set_index(df_apl.APDatetimevar)
		df_apl.index = df_apl.index.tz_localize(tz='Europe/Amsterdam')
	except:
		warnings.warn("Initial APL reading failed. Removing first line and retrying...")
		df_apl = df_apl.iloc[1:, :]
		df_apl.APDatetimevar = df_apl.APDatetimevar.apply(xlrd.xldate_as_datetime, convert_dtype=True, datemode=0)
		df_apl = df_apl.set_index(df_apl.APDatetimevar)
		df_apl.index = df_apl.index.tz_localize(tz='Europe/Amsterdam')
	# resample
	samp_time = 1/resample_Hz
	samp_str = str(samp_time) + 'S'
	df_apl = df_apl.resample(samp_str).ffill()
	df_apl = df_apl.bfill()
	df_apl['time'] = df_apl.index

	# subset relevant cols
	df_apl = df_apl.iloc[:,3:7]

	# set info for raw conversion
	apl_start = df_apl.index[0]
	apl_head = list(df_apl.columns)
	apl_np = df_apl.to_numpy().transpose()

	# convert to raw
	mne_info = mne.create_info(ch_names=apl_head, sfreq=resample_Hz, ch_types="misc")
	mne_obj = mne.io.RawArray(apl_np, mne_info, first_samp=0)

	# set start time and return an mne object
	mne_obj = mne_obj.set_meas_date(apl_start.timestamp())

	return mne_obj
