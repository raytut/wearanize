#### zmax conversion on the HBS parent folder "Y:\HB\data\example_subjects\data" in-place the same folder structure (without redirection of writing)
```
zmax_edf_merge_converter.exe "Y:\HB\data\example_subjects\data" --no_overwrite --temp_file_postfix="_TEMP_" --zipfile_match_string="_wrb_zmx_" --zipfile_nonmatch_string="_merged|_raw| - empty|_TEMP_" --exclude_empty_channels --zmax_lite --read_zip --write_zip --zmax_ppgparser --zmax_ppgparser_exe_path="C:\Program Files (x86)\Hypnodyne\ZMax_2022\PPGParser.exe" --zmax_ppgparser_timeout=1000
```

#### zmax conversion on the HBS parent folder "Y:\HB\data\example_subjects\data" and redirect of writing
```
zmax_edf_merge_converter.exe "Y:\HB\data\example_subjects\data" --write_redirection_path="Y:\HB\data\example_subjects\redirect" --no_overwrite --temp_file_postfix="_TEMP_" --zipfile_match_string="_wrb_zmx_" --zipfile_nonmatch_string="_merged|_raw| - empty|_TEMP_" --exclude_empty_channels --zmax_lite --read_zip --write_zip --zmax_ppgparser --zmax_ppgparser_exe_path="C:\Program Files (x86)\Hypnodyne\ZMax_2022\PPGParser.exe" --zmax_ppgparser_timeout=1000
```

#### zmax conversion on multiple HBS subject folders and redirect of writing
```
zmax_edf_merge_converter.exe "Y:\HB\data\example_subjects\data\sub-HB0109563627639"  "Y:\HB\data\example_subjects\data\sub-HB0139930729424" --write_redirection_path="Y:\HB\data\example_subjects\redirect" --no_overwrite --temp_file_postfix="_TEMP_" --zipfile_match_string="_wrb_zmx_" --zipfile_nonmatch_string="_merged|_raw| - empty|_TEMP_" --exclude_empty_channels --zmax_lite --read_zip --write_zip --zmax_ppgparser --zmax_ppgparser_exe_path="C:\Program Files (x86)\Hypnodyne\ZMax_2022\PPGParser.exe" --zmax_ppgparser_timeout=1000
```