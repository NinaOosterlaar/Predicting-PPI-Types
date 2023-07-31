import os
import json
import numpy as np
import csv


def retrieve_activation_inhibition(filename: str, new_file: str):
    """This function retrieves the activation and inhibition ppi's from the file and writes them to a new file.
        filename: the name of the file containing the ppi's
        new_file: the name of the new file"""
    with open(filename) as f:
        lines = f.readlines()

    if os.path.exists(new_file):
        os.remove(new_file)

    with open(new_file, 'w') as w:
        w.write(lines[0])
        for line in lines:
            if "activation" in line or "inhibition" in line:
                w.write(line)


def count_confidence_scores(filename: str):
    """This function counts the number of ppi's per confidence score.
        filename: the name of the file containing the ppi's
        return a list containing the number of ppi's per confidence score"""
    scores = np.zeros(10)
    
    with open(filename) as f:
        lines = f.readlines()[1:]
        for line in lines:
            scores[int(float(line[-4:])/100)] += 1
    return scores
    

def create_json_file(input: dict or json, output_filename: str):
    """Create a json file from a dictionary or json object.
        input: the dictionary or json object
        output_filename: the name of the output file"""
    with open(output_filename, "w") as f:
        json.dump(input, f, indent=4)
        

def create_csv_file(input: list, output_filename: str):
    """Create a csv file from a list.
        input: the list
        output_filename: the name of the output file"""
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(input, )
        
        
def compare_csv_results(filename_results: str, filename_reference: str):
    """Compare two csv files to check the differenced in my results compared to the reference.
        filename1: the name of my results file
        filename2: the name of the reference file
        return the fraction of correct results"""
    with open(filename_results) as f:
        lines1 = f.readlines()
    with open(filename_reference) as f:
        lines2 = f.readlines()
    
    count = 0 
    mistake_i = 0
    mistake_a = 0
        
    for line in lines1:
        if line not in lines2:
            # temp = line.split(',')
            # try2 = temp[1] + ',' + temp[0] + ',' + temp[2]
            # if try2 not in lines2:
            if "inhibition" in line:
                mistake_i += 1
            if "activation" in line:
                mistake_a += 1
            print(f"mistake: {line}")
        else:
            print(f"Correct: {line}")
            count += 1
    print(count)
    result = count / len(lines1)
    print(f"Activation mistake: {mistake_a}")
    print(f"Inhibtion mistake: {mistake_i}")
    print(result)
    return result


def filter_by_scores(filename, output_file):
    """Filter all the ppi's by their confidence score, and only keep the high confidence ones.
        Write those to a new file.
        filename: the name of the file containing the ppi's
        output_file: the name of the new file"""
    result = ""
    with open(filename) as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip("\n").split('\t')
            if(float(line[-1]) >= 800):
                temp = line[0][5:] + ',' + line[1].strip("\t")[5:] + '\n'
                print(temp)
                result += temp

    with open(output_file, 'w') as f:
        f.write(result)


if __name__ == '__main__':
    # retrieve_activation_inhibition(
    #     'Code/Data/4932.protein.actions.v11.0.txt', 'Code/Data/protein.activation_inhibition.txt')
    # count_confidence_scores('Code/Data/4932.protein.actions.v11.0.txt')
    filter_by_scores('Code/Data/protein.activation_inhibition.txt', 'Code/Data/protein.activation_inhibition_input.csv')


