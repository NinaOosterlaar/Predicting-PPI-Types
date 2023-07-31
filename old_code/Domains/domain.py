import csv
import os


def filter_methods_tsv_file(filename: str):
    """Filter the methods in the tsv file and write them to a dictionary
        filename: the name of the tsv file to filter"""
    with open(filename) as f:
        lines = f.readlines()[1:]
        methods = {}
        
        for line in lines:
            line_split = line.split("\t")
            if(line_split[3] not in methods):
                methods[line_split[3]] = [line_split[2] + "\t" + line_split[4] + "\t" + line_split[5] + "\t" + line_split[6] + "\t" + line_split[7] + "\n"]
            else:
                methods[line_split[3]].append(line_split[2] + "\t" + line_split[4] + "\t" + line_split[5] + "\t" + line_split[6] + "\t" + line_split[7] + "\n")
        
        for method in methods:
            write_to_file(methods[method], method)
            

def filter_methods_ggf3_file(filename: str):
    """Filter the methods in the GGF3 file and write them to a dictionary
        filename: the name of the GGF3 file to filter"""
    with open(filename) as f:
        lines = f.readlines()[1:]
        methods = {}
        
        for line in lines:
            line_split = line.split("\t")
            chromosome = line_split[0]
            
            end = line_split[3]
            start = line_split[4]
            if(int(end) < int(start)):
                start, end = end, start
            
            line_2 = line_split[8].split(";")
            temp = line_2[-3].split("_")
            gene = temp[0]
            domain_name = temp[1]
            description = retrieve_all_descriptions(line_2)
            
            if(line_split[1] not in methods):
                methods[line_split[1]] = {}
                methods[line_split[1]][chromosome] = [f"{gene}\t{start}\t{end}\t{description}\t{domain_name}\n"]
            else:
                if(chromosome not in methods[line_split[1]]):
                    methods[line_split[1]][chromosome] = [f"{gene}\t{start}\t{end}\t{description}\t{domain_name}\n"]
                else:
                    methods[line_split[1]][chromosome].append(f"{gene}\t{start}\t{end}\t{description}\t{domain_name}\n")
        
        for method in methods:
            for chromosome in methods[method]:
                write_to_file(methods[method][chromosome], method, chromosome)


def retrieve_all_descriptions(line: list):
    """Retrieve all the description from the list representing the current line in the database
        Find out how many descriptions are there and concatenate them to a string
        line: the list that contains all the information of a domain including the descriptions
        return the string that contains all the descriptions"""
    description = ""
    description_length = len(line) - 3
    for i in range(1, description_length):
        description += line[i]
        if(i + 1 != description_length):
            description += ";"
    
    return description


def write_to_file(method: list, method_name: str, number: int):
    """Write one of the method from methods in a file
        method: the method to write to a file
        method_name: the name of the method to write to a file""" 
    filename = f"Domains/filtered_files/{method_name}/chromosome_{number}.txt"
    
    with open(filename, "w") as f:
        f.write("Gene\tstart\tend\tdescription\tdatabase_name\n")
        for line in method:
            f.write(line)
            
            
def sort_file(filename: str):
    """Sort the file by the second column
        filename: the name of the file to sort"""
    with open(filename) as f:
        lines = list(csv.reader(f, delimiter="\t"))[1:]
        lines.sort(key=lambda x: int(x[1]))
        
        with open(filename, "w", newline="") as f:
            f.write("Gene\tstart\tend\tdescription\tdatabase_name\n")
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(lines)
            

def sort_all_files_in_directory(directory: str):
    """Sort all the files in the directory by looping over all the files in the directory
        and calling the sort_file function
        directory: the directory that contains the files to sort"""
    for filename1 in os.listdir(directory):
        f1 = os.path.join(directory, filename1)
        for filename2 in os.listdir(f1):
            f2 = os.path.join(f1, filename2)
            if os.path.isfile(f2):
                sort_file(f2)


if __name__ == "__main__":
    # filter_methods_ggf3_file("Domains/Initial_datasets/protein_domains.GFF3")
    sort_all_files_in_directory("Domains/filtered_files")