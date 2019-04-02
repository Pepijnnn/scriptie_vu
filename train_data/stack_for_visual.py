import numpy as np
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

	# show the development of a specific bacteria between 2 dates
	def show_bacteria_date(self, bacteria, date1, date2):
		relevant_bac_date = []
		for (bac, date) in self.bacteria_date:
			 if date > date1 and date < date2 and bac == bacteria:
				 relevant_bac_date.append((bac,date))

		return cbor.loads(relevant_bac_date)

	# show all the bacterias of a certain date
	def show_bacteria(self, date1, date2):
		relevant_bacs = []
		for (bac, date) in self.bacteria_date:
			 if date > date1 and date < date2:
				 relevant_bacs.append((bac,date))
		return cbor.loads(relevant_bacs)

	# show the bacteria through the time
	def show_dates(self, bacteria):
		relevant_dates = []
		for (bac, date) in self.bacteria_date:
			if str(bac) == str(bacteria):
				relevant_dates.append(date)
		return cbor.loads(relevant_dates)