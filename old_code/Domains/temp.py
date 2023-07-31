import os
import numpy as np
import csv
import json

def count_transposons():
    answer = {"total": [0, 0]}
    with open('Domains/transposons/transposon_chrXV.wig') as transposons:
        lines = list(csv.reader(transposons, delimiter = ' '))[1:]
    arr = np.array(lines)
    arr = arr.astype('float64')
    
    counts = np.sum(arr, axis=0)
    number = np.size(arr, axis=0)
    answer["test"] = [counts[0], counts.tolist()[1]]
    answer["total"][0] += counts[0]
    answer["total"][1] += counts[1]
    # with open("Domains/domains_result/total_transposon_count.json", "w") as json_file:
    #     json.dump(answer, json_file, indent=4)
    return answer

if __name__ == '__main__':
    count_transposons()