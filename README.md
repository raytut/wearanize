# wearanize

## Building the Environment

Having the correct environment is necessary for carrying out all processing steps for this data. We recommend the creation of a virtual python environment (for example, using Anaconda) to make sure the work is reproducable. To this end, we specify the required software below:

- Python 3.9>
- Hydnodyne Software for ZMAX ([https://hypnodynecorp.com/downloads.php](https://hypnodynecorp.com/downloads.php))
- Wearanize repository (i.e., this repository)

Note that the Hypnodyne software is available on windows PCs via the DCCN software center. In addition to the above software packages, you will also need to install the required python libraries from the supplied requirements.txt file of this repository. This can be done as follows:

```bash
pip install -r requirements.txt.
```

## Prep steps for data:

Some basic preprocessing will need to be done prior to any feature extraction. This varies for each device.

### Zmax:

In order to make the most out of the ZMAX data, you will first need to preprocess it. This will result in a cleaner EDF file that contains some preprocessed data including a cleaned BVP signal necessary for synchronization. This needs to be done on a Windows PC with the Hypnodyne suite installed on it. Using the below code, you can replace the project directory with that of the current data, and set the *zmax_ppgparser_exe_path* argument to the one on the current computer. 

```powershell
zmax_edf_merge_converter.exe "Y:\HB\data\example_subjects\data" --no_overwrite --temp_file_postfix="_TEMP_" --zipfile_match_string="_wrb_zmx_" --zipfile_nonmatch_string="_merged|_raw| - empty|_TEMP_" --exclude_empty_channels --zmax_lite --read_zip --write_zip --zmax_ppgparser --zmax_ppgparser_exe_path="C:\Program Files (x86)\Hypnodyne\ZMax_2022\PPGParser.exe" --zmax_ppgparser_timeout=1000
```

### E4:

While the E4 data can be used with functions within **Wearanize** directly, it is advised to create a concatenated file for all subjects. This can be done in python using multiple cores for faster processing on the DCCN cluster, and using the following command:

```python
from wearanize import e4_concatenate_par
project_folder = "/path/to/folder/with/all/subject"
e4_concatenate_par(project_folder)
```

Alternatively, single subjects can be analyzed as well using the following:

```python
e4_concatenate(project_folder, sub_nr, resampling=None)
```

## Feature Extraction:

**COMING SOON**
