import csv
 
# Set up input and output variables for the script
Data = open("data.csv", "r")
 
# Set up CSV reader and process the header
csvReader = csv.reader(Data)


 
# Make an empty list
Listofsentences=[]

for i in csvReader:
	Listofsentences.append(i)


