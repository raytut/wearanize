# -*- coding: utf-8 -*-
"""
Copyright 2022, Rayyan Tutunji
"""
# imports
import os
import glob
import pandas
import datetime
import zipfile
from zipfile import ZipFile
import mne
import shutil
from joblib import Parallel, delayed
import matplotlib
import matplotlib.pyplot as plt

# define file parts based on path
def fileparts(filepath):

	filepath = os.path.normpath(filepath)
	path_filename = os.path.split(filepath)
	filename = path_filename[1]
	path = path_filename[0]
	name_extension = os.path.splitext(filename)
	name = name_extension[0]
	extension = name_extension[1]
	return path, name, extension

# convert e4 data to mne.raw format
def read_e4_to_raw_list(filepath):

	filepath = os.path.join(filepath)
	filepath = os.path.normpath(filepath)
	path, name, extension = fileparts(filepath)

	# Read in the e4 file
	emp_zip = zipfile.ZipFile(filepath)
	channels = ['BVP.csv', 'HR.csv', 'EDA.csv', 'TEMP.csv', 'ACC.csv']
	sampling_frequencies = [64, 1, 4, 4, 32]
	mne_raw_list = ["unretrieved"] * len(channels)

	# Check if single session or full recording
	if "full" not in name:
		# Run over all signals
		for i, signal_type in enumerate(channels):
			if signal_type != "ACC.csv":
				# Read signal
				raw = pandas.read_csv(emp_zip.open(signal_type))
				# create channel info for mne.info file
				channel = signal_type.split(".")
				channel = channel[0].lower()
				sfreq = int(raw.iloc[0, 0])
				timestamp = int(float(raw.columns[0]))
				mne_info = mne.create_info(ch_names=[channel], sfreq=sfreq, ch_types="misc")
				# Create MNE Raw object and add to a list of objects
				mne_obj = mne.io.RawArray([raw.iloc[1:, 0]], mne_info, first_samp=0)
				mne_obj.set_meas_date(timestamp)
				mne_raw_list[i] = mne_obj
			else:
				# Read signal
				raw = pandas.read_csv(emp_zip.open(signal_type))
				# create channel info for mne.info file
				channel = signal_type.split(".")
				channel = channel[0].lower()
				sfreq = int(raw.iloc[0, 0])
				timestamp = int(float(raw.columns[0]))
				mne_info = mne.create_info(ch_names=["acc_x", "acc_y", "acc_z"], sfreq=sfreq, ch_types="misc")
				# convert to standard gunits
				raw.iloc[1:, 0] = raw.iloc[1:, 0] * 2 / 128
				raw.iloc[1:, 1] = raw.iloc[1:, 1] * 2 / 128
				raw.iloc[1:, 2] = raw.iloc[1:, 2] * 2 / 128
				# Create MNE Raw object and add to a list of objects
				mne_obj = mne.io.RawArray([raw.iloc[1:, 0], raw.iloc[1:, 1], raw.iloc[1:, 2]], mne_info, first_samp=0)
				mne_obj.set_meas_date(timestamp)
				mne_raw_list[i] = mne_obj
	else:
		# Run over all signals
		for i, signal_type in enumerate(channels):
			if signal_type != "ACC.csv":
				# Read signal
				raw = pandas.read_csv(emp_zip.open(signal_type), sep="\t", index_col=0)

				# create channel info for mne.info file
				channel = signal_type.split(".")
				channel = channel[0].lower()
				sfreq = sampling_frequencies[i]
				mne_info = mne.create_info(ch_names=["timestamp_ux", channel], sfreq=sfreq, ch_types="misc")

				# create timestamp arra
				raw.time = pandas.to_datetime(raw.time, format='%Y-%m-%d %H:%M:%S.%f', exact=True, utc=True)
				timestamp = raw.iloc[0:1, 1]
				raw['time'] = pandas.to_numeric(raw['time'])

				# create mne object. time not relevant here
				mne_obj = mne.io.RawArray([raw.iloc[1:, 1], raw.iloc[1:, 0]], mne_info, first_samp=0)
				mne_raw_list[i] = mne_obj
			else:

				# Read signal
				raw = pandas.read_csv(emp_zip.open(signal_type), sep="\t", index_col=0)
				pandas.to_datetime(raw.time)

				# create channel info for mne.info file
				channel = signal_type.split(".")
				channel = channel[0].lower()
				sfreq = sampling_frequencies[i]
				mne_info = mne.create_info(ch_names=["timestamp_ux", "acc_x", "acc_y", "acc_z"], sfreq=sfreq,
										   ch_types="misc")

				# create timestamp arra
				raw.time = pandas.to_datetime(raw.time, format='%Y-%m-%d %H:%M:%S.%f', exact=True, utc=True)
				timestamp = raw.iloc[0:1, 3]
				raw['time'] = pandas.to_numeric(raw['time'])

				# Create MNE Raw object and add to a list of objects
				mne_obj = mne.io.RawArray([raw.iloc[1:, 3], raw.iloc[1:, 0], raw.iloc[1:, 1], raw.iloc[1:, 2]],
										  mne_info, first_samp=0)
				mne_raw_list[i] = mne_obj

	return mne_raw_list

# convert e4 data to single mne raw file (flattened)
def read_e4_to_raw(filepath, resample_Hz=64, interpolate_method='ffill'):
	"""
	Read in Empatica files to raw format with resampling for different channels
	Parameters:
	----------
	filepath: str
		Path to zip file containing E4 Data. Must be original and not concatenated file.
	resample: int
		Resampling frequency
	interpolate_method: str
		Method with which interpolation is done
	"""
	mne_raw_list = read_e4_to_raw_list(filepath)
	mne_raw_list_new = []
	mne_raw_df = pandas.DataFrame()
	for i, raw in enumerate(mne_raw_list):
		if raw != "unretrieved":
			mne_raw_list_new.append(raw.resample(resample_Hz))
			mne_temp_df = mne_raw_list_new[i].to_data_frame(time_format="datetime")
			mne_temp_df = mne_temp_df.set_index('time', drop=True)
			mne_raw_df = pandas.concat([mne_raw_df, mne_temp_df], axis=1)

	# append the raws together
	mne_raw_info = mne.create_info(ch_names=list(mne_raw_df.columns), sfreq=resample_Hz)
	mne_raw_df = mne_raw_df.interpolate(method=interpolate_method)
	mne_raw_np = mne_raw_df.to_numpy().transpose()
	raw = mne.io.RawArray(mne_raw_np, mne_raw_info)
	raw.set_meas_date(mne_raw_list[0].info['meas_date'])
	return raw

# temporally concatenate e4 data and save as separate zip file in same directory
def e4_concatenate(project_folder, sub_nr, resampling=None):
	# Set sub nr as string
	sub = str(sub_nr)
	# Make array with the sessions for loop
	sessions = glob.glob(os.path.join(project_folder, str(sub)) + os.sep + "pre-*" + os.sep + "wrb")

	# Reset for memory
	full_df = None
	df = None

	print('Concatinating E4 sessions for ' + sub + '. This may take a while...')
	# Loop over all session files
	for session_type in sessions:

		# Path with E4 files. Only run if the files exist
		filepath = (str(session_type))
		if os.path.isdir(filepath) == True:

			# Get all directories with E4 sessions for subject, merge directory from the list
			dir_list = glob.glob(filepath + "/*wrb_emp_*.zip")
			# Only keep the empatica folders, drop the folder with concatenated data
			dir_list = [x for x in dir_list if "wrb_emp" in x]
			dir_list = [x for x in dir_list if "wrb_emp_full" not in x]

			# Only Run if there are recordings
			if len(dir_list) > 0:

				# Check if merge directory (for output) exists, if not then make it
				try:
					# Make a directory that matches the HBS format
					conc_file = dir_list[0][:-7]
					conc_file = conc_file + '_full'
					os.makedirs(str(conc_file))
				except FileExistsError:
					pass

				# Set E4 data types for loop
				data_types = ['EDA.csv', 'TEMP.csv', 'IBI.csv', 'BVP.csv', 'HR.csv', 'ACC.csv']

				for data_type in data_types:

					# Make Empty DF as master df for data type
					full_df = pandas.DataFrame()

					# IBI is special case
					if data_type == 'IBI.csv':

						# Select Directory from available list
						for k in dir_list:

							# Select File for single session, import as df
							zipdir = ZipFile(k)

							# Sometime IBI files are empty, so try this instead
							try:
								df = pandas.read_csv(zipdir.open(data_type))
								# Get time stamp
								time = list(df)
								time = time[0]
								time = float(time)

								# Rename time column to time, data to Data
								df = df.rename(columns={df.columns[0]: "time"})
								df = df.rename(columns={df.columns[1]: "data"})

								# Add the starttime from time stamp (time) to the column+Convert to datetime
								# time=dt.datetime.fromtimestamp(time)
								df['time'] = time + df['time']
								df['time'] = pandas.to_datetime(df['time'], unit='s')

								# Append to master data frame the clear it for memory
								full_df = pandas.concat([full_df, df])
								full_df = full_df.sort_values(by='time')
								df = pandas.DataFrame()
							except:
								pass

						# Convert IBI to ms and sort by date:
						full_df['data'] = full_df['data'] * 1000
						full_df = full_df.sort_values('time', ascending=True)

						# Set Output Names and direcotries, save as csv
						fullout = (str(conc_file) + os.sep + str(data_type))
						full_df.to_csv(str(fullout), sep='\t', index=True)
						# Clear dataframes for more memory
						full_df = pandas.DataFrame()

					# ACC also special case, implement alternate combination method
					elif data_type == 'ACC.csv':

						# Select Directory, go through files
						for k in dir_list:

							# Select File, Import as df
							zipdir = ZipFile(k)
							df = pandas.read_csv(zipdir.open(data_type))

							# Get time stamp (Used Later)
							time = list(df)
							time = time[0]
							time = float(time)

							# Get Sampling Frequency, convert to time
							samp_freq = df.iloc[0, 0]
							samp_freq = float(samp_freq)
							samp_time = 1 / samp_freq

							# Drop sampling rate from df (first row)
							df = df.drop([0])

							# Rename data columns to corresponding axes
							df = df.rename(columns={df.columns[0]: "acc_x"})
							df = df.rename(columns={df.columns[1]: "acc_y"})
							df = df.rename(columns={df.columns[2]: "acc_z"})

							# Make array of time stamps
							df_len = len(df)
							time = pandas.to_datetime(time, unit='s')
							times = [time]
							for i in range(1, (df_len)):
								time = time + datetime.timedelta(seconds=samp_time)
								times.append(time)

							# Add time and data to dataframe
							df['time'] = times

							# Do resampling if specified
							if resampling != None:
								# If downsampling
								if resampling > samp_time:
									# Upsample data to 256HZ here to avoid large memory costs
									df = df.resample((str(resampling) + "S"), on="time").mean()
								# If Upsampling
								else:
									df = df.set_index("time")
									df = df.resample((str(resampling) + "S")).ffill()

							# Append to master data frame
							full_df = pandas.concat([full_df, df])
							df = pandas.DataFrame()

						# Sort master by date:
						full_df = full_df.sort_values(by='time')

						# Set Output Names and direcotries, save as csv
						fullout = (str(conc_file) + os.sep + str(data_type))
						full_df.to_csv(str(fullout), sep='\t', index=True)

						# Clear dataframe and free memory
						full_df = pandas.DataFrame()

					# All other data structures:
					else:
						for k in dir_list:

							# Select File, Import as df
							zipdir = ZipFile(k)
							df = pandas.read_csv(zipdir.open(data_type))

							# Get start time+sampling frequency
							start_time = list(df)
							start_time = start_time[0]
							samp_freq = df.iloc[0, 0]

							# Change samp freq to samp time
							samp_time = 1 / samp_freq

							# Drop sampling rate from df
							df = df.drop([0])

							# Convert start time to date time
							start_time = int(float(start_time))
							start_time = pandas.to_datetime(start_time, unit='s')

							# Make array of time
							file_len = len(df)
							times = [start_time]
							for i in range(1, (file_len)):
								start_time = start_time + datetime.timedelta(seconds=samp_time)
								times.append(start_time)

							# Add time and data to dataframe
							df['time'] = times

							# Rename first column to Data
							df = df.rename(columns={df.columns[0]: "data"})

							# Do resampling if specified
							if resampling != None:
								# If downsampling
								if resampling > samp_time:
									# Upsample data to 256HZ here to avoid large memory costs
									df = df.resample((str(resampling) + "S"), on="time").mean()
								# If Upsampling
								else:
									df = df.set_index("time")
									df = df.resample((str(resampling) + "S")).ffill()

							# Append to master data frame
							full_df = pandas.concat([full_df, df])
							df = pandas.DataFrame()

						# Sort by date:
						full_df = full_df.sort_values(by='time')

						# Set Output Names and direcotries, save as csv
						fullout = (str(conc_file) + os.sep + str(data_type))
						full_df.to_csv(str(fullout), sep='\t', index=True)

						# Clear data frame and free up memory
						full_df = pandas.DataFrame()

				# Zip file
				zippath = (conc_file + ".zip")
				with zipfile.ZipFile(zippath, mode='w') as zf:
					len_dir_path = len(conc_file)
					for root, _, files in os.walk(conc_file):
						for file in files:
							filepath = os.path.join(root, file)
							zf.write(filepath, filepath[len_dir_path:], compress_type=zipfile.ZIP_DEFLATED,
									 compresslevel=6)
				shutil.rmtree(conc_file)


def e4_concatente_par(project_folder, verbose=0):
	# Get list of subjects
	sub_list = glob.glob(project_folder + os.sep + "sub-*")
	Parallel(n_jobs=-2, verbose=verbose)(delayed(e4_concatenate)(project_folder, i) for i in sub_list)


def e4_plot(emp_file):
	print("\nLoading...")

	# read in file in raw format
	raw = read_e4_to_raw_list(emp_file)
	df_bvp = raw[0].to_data_frame()
	df_hr = raw[1].to_data_frame()
	df_eda = raw[2].to_data_frame()
	df_temp = raw[3].to_data_frame()
	df_acc = raw[4].to_data_frame()

	# Plots
	Title = 'E4 Plot for ' + emp_file
	# plt.rcParams["font.family"] = 'Cambria'
	plt.style.use('ggplot')
	# Plot
	fig, axs = plt.subplots(3, sharex=True)
	fig.suptitle(Title)

	# HR
	axs[0].plot(pandas.to_datetime(df_bvp.timestamp_ux, exact=True), df_bvp.bvp, 'purple')
	axs[0].grid(b=True, which='both', axis='both', color='lightgrey', markevery=5)
	axs[0].set(xlabel=' ', ylabel='Blood Volume Pulse')
	axs[0].set_ylim(ymin=50, ymax=160)
	# IBI
	axs[1].plot(pandas.to_datetime(df_hr.timestamp_ux, exact=True), df_hr.hr, 'steelblue')
	axs[1].grid(b=True, which='both', axis='both', color='lightgrey', markevery=5)
	axs[1].set(xlabel=' ', ylabel='Heart Rate (ms)')
	# SCR
	axs[2].plot(pandas.to_datetime(df_eda.timestamp_ux, exact=True), df_eda.eda, 'blue')
	axs[2].grid(b=False, which='both', axis='both', color='lightgrey', markevery=5)
	axs[2].set(xlabel=' ', ylabel='SC ($\mu$S)')
	axs[2].set_ylim(ymin=-5, ymax=10)
	# Temp
	axs[3].plot(pandas.to_datetime(df_temp.timestamp_ux, exact=True), df_temp.temp, 'tomato')
	axs[3].grid(b=True, which='both', axis='both', color='lightgrey', markevery=5)
	axs[3].set(xlabel=' ', ylabel='Temp ($^\circ$C)')
	axs[3].set_ylim(ymin=15, ymax=45)
	# ACC
	axs[4].plot(pandas.to_datetime(df_acc.timestamp_ux, exact=True), df_acc.acc_x, 'blue', alpha=0.4)
	axs[4].plot(pandas.to_datetime(df_acc.timestamp_ux, exact=True), df_acc.acc_y, 'red', alpha=0.4)
	axs[4].plot(pandas.to_datetime(df_acc.timestamp_ux, exact=True), df_acc.acc_z, 'green', alpha=0.4)
	axs[4].grid(b=True, which='both', axis='both', color='lightgrey', markevery=5)
	axs[4].set(xlabel='Time', ylabel='ACC')
	axs[4].set_ylim(ymin=-10)
	# Plot
	plt.show()
