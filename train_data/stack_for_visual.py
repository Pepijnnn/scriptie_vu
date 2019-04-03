import numpy as np
import matplotlib
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import datetime as dt
from collections import OrderedDict
import cbor

class Store:

	def __init__(self):
		self.bac_list = list()
		self.dat_list = list()
		self.bacteria_date = list()

	def add_bacteria_date(self, bacteria, date):
		self.bacteria_date.append((bacteria,date))
		self.bac_list.append(bacteria)
		self.dat_list.append(date)

	def plot_bar_single_date(self, bac_count_dict):
		plt.bar(range(len(bac_count_dict)), list(bac_count_dict.values()), tick_label=list(bac_count_dict.keys()), align='center')
		plt.show()

	# show the development of a specific bacteria between 2 dates
	def show_bacteria_date(self, bacteria, date1, date2):
		relevant_bac_date = []
		for (bac, date) in self.bacteria_date:
			 if date > date1 and date < date2 and bac == bacteria:
				 relevant_bac_date.append((bac,date))

		return cbor.loads(relevant_bac_date)

	# show all the bacterias of a certain date
	def show_bacteria(self, date):
		rel_bacs_y = []

		for (bac, date_store) in self.bacteria_date:
			if date_store == date:
				for indiv in bac:
					rel_bacs_y.append(indiv)
		print(rel_bacs_y)
		count_d = {x:rel_bacs_y.count(x) for x in rel_bacs_y}
		self.plot_bar_single_date(count_d)

	# IMPROVEMENT NEEDED FOR DATETIME
	def plot_bar_single_bacteria(self, date_count_dict):
		plt.bar(range(len(date_count_dict)), list(date_count_dict.values()), tick_label=list(date_count_dict.keys()), align='center')
		plt.show()

	# show the bacteria through the time
	def show_dates(self, bacteria):
		rel_dates = []
		for (bacs, date) in self.bacteria_date:
			for indiv_bac in bacs:
				if str(indiv_bac) == str(bacteria):
					rel_dates.append(date)

		rel_dates.append("26-8-2016")
		rel_dates.append("26-8-2018")
		rel_dates.append("28-8-2016")
		rel_dates.append("26-08-2016")
		univ_dates = sorted([dt.datetime.strptime(d, '%d-%m-%Y').date() for d in rel_dates])

		odict = OrderedDict()
		for item in univ_dates:
			try:
				odict[item] += 1
			except KeyError:
				odict[item] = 1

		self.plot_bar_single_bacteria(odict)
		


######### OLD CODE ############
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# plt.plot(univ_dates)
# plt.gcf().autofmt.xdate()
# exit()