import pandas as pd

def csv_analyzer(path_to_csv:str, single_column:str='', separator:str=','):

	'''
	Analyzes a CSV file and prints statistical information about it.

	The function reads a CSV file specified by 'path_to_csv' and either calculates
	the mean of a specific column if 'single_column' is provided, or prints out
	the statistical summary of the DataFrame otherwise. It handles file errors gracefully.

	Args:
	- path_to_csv (str): The absolute or relative path to the input CSV file.
	- single_column (str, optional): The name of a single column for which to calculate the mean.
									 If left empty, the statistical summary of the entire DataFrame
									 will be printed. Defaults to ''.
	- separator (str, optional): The column separator in the CSV file. Defaults to ','.

	Returns:
	None
	'''

	try:
		data = pd.read_csv('data.csv', sep=separator)

		if single_column != '':
			mean = data[single_column].mean()
			print(f"Mean of {single_column} is: {mean}")
		else:
			data.describe()

	except Exception as ex:
		print(f'csv_analyzer saw error {ex}')