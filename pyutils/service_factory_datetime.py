import datetime


def get_yesterday(format='%Y%m%d', days=1):

	'''
	This function returns yesterday date using the format user passes as a parameter.
	Format for returned day has to be passed like a string.

	POSSIBLE FORMATS:
		%Y = year like 2021
		%m = month like 08
		%d = day like 01
	'''

	return (datetime.datetime.now() - datetime.timedelta(days=days)).strftime(format=format)


def get_today(format:str='%Y%m%d'):
	
	'''
	This function returns today date using the format user passes as a parameter.
	Format for returned day has to be passed like a string.

	POSSIBLE FORMATS:
		%Y = year like 2021
		%m = month like 08
		%d = day like 01
	'''

	return (datetime.datetime.now()).strftime(format=format)


def get_recent_monday(format:str='%Y%m%d'):

	'''
	This function returns the very most recent monday date using the format user passes as a parameter.
	If today is monday this function returns today as a returning date.
	Format for returned day has to be passed like a string.

	POSSIBLE FORMATS:
		%Y = year like 2021
		%m = month like 08
		%d = day like 01
	'''

	current_weekday = datetime.datetime.today().weekday()
	return (datetime.datetime.now() - datetime.timedelta(days = current_weekday)).strftime(format=format)


def get_last_monday(format:str='%Y%m%d'):

	'''
	This function returns the last monday (not during the current week) date using the format user passes as a parameter.
	If today is monday this function returns today as a returning date.
	Format for returned day has to be passed like a string.

	POSSIBLE FORMATS:
		%Y = year like 2021
		%m = month like 08
		%d = day like 01
	'''

	current_weekday = datetime.datetime.today().weekday()
	return (datetime.datetime.now() - datetime.timedelta(days = current_weekday + 7)).strftime(format=format)


def get_current_month_startdate(format:str='%Y%m%d'):

	'''
	This function returns the current month from its starting day (e.g. if the current month is March, function will return 20210301)

	POSSIBLE FORMATS:
		%Y = year like 2021
		%m = month like 08
		%d = day like 01
	'''

	return datetime.datetime.today().replace(day=1).strftime(format=format)


def get_current_year_startdate(format:str='%Y%m%d'):

	'''
	This function returns the current year from its starting day (e.g. if the current month is March, function will return 20210101)

	POSSIBLE FORMATS:
		%Y = year like 2021
		%m = month like 08
		%d = day like 01
	'''

	return datetime.datetime.today().replace(month=1, day=1).strftime(format=format)