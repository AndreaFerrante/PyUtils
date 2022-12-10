import datetime
import math


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


<<<<<<< HEAD
def month_reduction(month_delta:int, month_current:int, year_current:int) -> str:

	'''
	params:
		month_delta:int   : this param gets how many month the function should go back in time (use zero to receive the current month starting day)
		month_current:int : this param is the current month from which we should get month back in time
		year_current:int  : this param is the current year from which we should get month back in time
	returns:
		returns a string in format YYYYMMDD of the first month back in time given parameters
	'''

	month_unit      = 1 / 12
	year_current_   = year_current * 12 + month_current
	year_current_   = (year_current_ - month_delta) * month_unit

	final_year      = round(math.modf(year_current_)[1])
	final_month     = round(math.modf(year_current_)[0] / month_unit)

	final_year  = np.where(final_month == 0, final_year - 1, final_year)
	final_month = np.where(final_month == 0, 12 , final_month)
=======

def get_time_month(delta_month:int, current_month:int, current_year:int) -> str:

	'''
	This function returns the yearmonth which distance is delta_month far away.

	params:
	 - delta_month:   distance in month from current yearmonth
	 - current_month: current month from which the distance should be calculated
	 - current_year:  current yeat from which the distance should be calculated
	'''

	month_unit      = 1 / 12
	year_current_   = current_year * 12 + current_month
	year_current_   = (year_current_ - delta_month) * (1/12)

	final_year      = round(math.modf(year_current_)[1])
	final_month     = round(math.modf(year_current_)[0] / (1/12)) + 1
>>>>>>> c2bf691 (Datetime factory updated)

	return str(final_year) + str(final_month).zfill(2) + '01'


<<<<<<< HEAD
=======






>>>>>>> c2bf691 (Datetime factory updated)
