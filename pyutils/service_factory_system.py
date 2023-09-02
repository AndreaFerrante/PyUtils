import os
import re
import time
import psutil
import zipfile
import warnings
import pandas as pd


def get_files_in_path(path_to_search:str='', file_extension:str=None) -> list:


	'''
	This function returns all files in a given folder and given a specific file extension (if passed by the user !)

	PARAMETERS:
		path_to_search: string format, this is the a single folder path form which the files list is created
		file_extension: string format, this is the file extension of files inside returned files list (do not pass the dot)
	'''


	files  = os.listdir( path_to_search )
	files_ = []


	for file in files:

		################################################################################################
		# Use a REGEX to find all the comes after a comma inside the file name (i.e. the file extension)
		################################################################################################

		if file_extension:
			if re.findall('[^.]+$', file)[0] == file_extension:
				files_.append( file )
		else:
			files_.append( file )


	return files_


def get_file_in_root_path(path_to_search:str='', file_name:str=''):

	'''
	This function returns all file with a name in a root path.
	This function works top-down: from top filename path till the very root one.

	PARAMETERS:
		path_to_search: string format, this is the a single folder path form which the files list is created
		file_name: string format, this is the file name to be searched top-down given the root path
	'''

	result = []

	for root, dir, files in os.walk(path_to_search):
		if file_name in files:
			result.append(os.path.join(root, file_name))

	return result


def get_file_content(path_to_read:str=None, file_name:str=None):

	'''
	This function returns all the content of a file (read mode only): useful to read SQL file to be downloaded.

	PARAMETERS:
		path_to_read: string format, path where the file to be read is located
		file_name: string format, this is the file name (if file name is empty function will pop up a warning)

	RETURNS:
		all_lines: string format, this is the inner content of the file we just read
	'''

	if file_name and path_to_read:

		with open(path_to_read + '/' + file_name, 'r') as f:
			all_lines = f.read().replace('\n', ' ')

		return all_lines

	else:

		warnings.warn("Attention: file name must be passed", RuntimeWarning)


def get_files_timestamps_in_path(path_to_scan:str=None) -> list:

	'''
	This function returns most recent file in terms of timestamps given a single folder path.

	PARAMETERS:
		path_to_scan: string format, path where files to be scanned by creation timestamps are located.

	RETURNS:
		files_timestamps: list of tuples, the function returns a list of tuples composed by single filename and its associated timestamp.
						  This list of tuples is sorted by the very most recent file timestamps.
	'''

	files_timestamps = []

	if path_to_scan:

		files_timestamps  = []
		files_path        = os.listdir(path_to_scan)

		for file in files_path:
			files_timestamps.append( (file, time.ctime(os.path.getctime(path_to_scan + '/' + file) ) ) )

		return sorted(files_timestamps, key=lambda x: x[1], reverse=True)



	else:

		warnings.warn("Attention: a path to scan must be passed", RuntimeWarning)
		return 0


def get_files_zipped_in_folder(path_to_zip:str=None, file_extensions:list=None) -> None:

	'''
	This function returns compress in zip format all files in a given folder.

	PARAMETERS:
		path_to_zip: string format, path where files to be zipped are located.
		file_extensions: list format, this is the files list extensions to be scanned and zipped
	'''

	if path_to_zip and file_extensions:


		current_wd   = os.getcwd()
		files_to_zip = os.listdir(path_to_zip)
		os.chdir(path_to_zip)

		for file in files_to_zip:

			if re.findall('[^.]+$', file)[0] in file_extensions:

				print(f'Compressing file named {file} . . .')

				file_zipped_name = os.path.splitext(file)[0] + '.zip'
				file_zipped      = zipfile.ZipFile(path_to_zip + '/' + file_zipped_name, 'w')
				file_zipped.write(file)
				file_zipped.close()

		os.chdir(current_wd)

	else:
		warnings.warn("Attention: path to zip and file extensions must be passed", RuntimeWarning)


def get_csv_files_in_path_stacked(path_to_stack:str=None, file_extensions:str='csv', sep:str=';', encoding='cp1252', print_files:bool=True):

	'''
	This function returns all CSV files (with a given extension and folder path) stacked one over the other !

	PARAMETERS:
		path_to_stack: string format, path where files to be stacked are located.
		file_extensions: string format, this is the files list extensions to be read and stacked
		sep: string format, this is the CSV separator
		encoding: string format, this is the CSV file encoding style

	RETURNS:
		files_stacked: pandas dataframe format, this is a dataframe with all files stacked on each other BY ROWS
	'''

	if path_to_stack:

		files_to_stack = os.listdir(path_to_stack)
		files_stacked  = []

		for file in files_to_stack:

			if re.findall('[^.]+$', file)[0] in file_extensions:

				if print_files:
					print(f'Reading file named {file} . . .')
				files_stacked.append( pd.read_csv( path_to_stack + '/' + file, sep=sep, encoding=encoding ) )

		return pd.concat(files_stacked)

	else:
		warnings.warn("Attention: path to stack must be passed", RuntimeWarning)


def get_xlsx_files_in_path_stacked(path_to_stack:str=None, file_extensions:list=['xls, xlsx', 'xlsm'], print_files:bool=True):

	'''
	This function returns all Excel files (with a given extension and folder path) stacked one over the other !

	PARAMETERS:
		path_to_stack: string format, path where files to be stacked are located.
		file_extensions: list format, this is the files list extensions to be read and stacked

	RETURNS:
		files_stacked: pandas dataframe format, this is a dataframe with all files stacked on each other BY ROWS
	'''

	if path_to_stack:

		files_to_stack = os.listdir(path_to_stack)
		files_stacked  = []

		for file in files_to_stack:

			if re.findall('[^.]+$', file)[0] in file_extensions:

				if print_files:
					print(f'Reading file named {file} . . .')
				files_stacked.append( pd.read_excel( path_to_stack + '/' + file) )

		return pd.concat(files_stacked)

	else:
		warnings.warn("Attention: path to stack must be passed", RuntimeWarning)


def list_processes():

	'''
	List all currently running processes on the machine.
	:return: A list of dictionaries containing the PID and name of each running process.
	:rtype: list[dict]

	:Example:
	list_processes()
	[{'pid': 1, 'name': 'systemd'}, {'pid': 2, 'name': 'kthreadd'}, ...]
	'''

	process_list = list()

	for process in psutil.process_iter(['pid', 'name']):
		process_list.append(process.info)

	return process_list


def close_app(app_name:str):

	'''
	Terminate a specific application (using its pid) starting from its goven name.

	:param app_name: The name of the application to terminate.
	:type app_name: str

	:Example:
	close_app("Chrome")
	'''

	for process in psutil.process_iter(['pid', 'name']):

		if app_name in str(process.info['name']).lower():
			print(f'Closing app named {process.info["name"]}')
			process = psutil.Process(process.info['pid'])
			process.terminate()


