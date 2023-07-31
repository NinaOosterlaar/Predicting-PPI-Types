import json
import csv


def retrieve_json_data(filename: str):
    """This function retrieves the json data from the file and returns it.
        filename: the name of the file containing the json data
        return the json data"""
    with open(filename) as json_file:
        data = json.load(json_file)
        return data


def retrieve_csv_data(filename: str):
    """This function retrieves csv data from the file and returns it.
        filename: the name of the file containing the csv data
        return the csv data"""
    with open(filename) as csv_file:
        return list(csv.reader(csv_file, delimiter=','))
        
    

if (__name__ == "__main__"):
    pass
