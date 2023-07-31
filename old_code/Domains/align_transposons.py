import csv
import os
import json
import roman


def open_transposon_file(filename: str):
    """Open the transposon file and return all the data in lists
        First element in list is the location in the genome
        Second element in the list is the number of transposons in the location
        filename: the name of the file to open
        return the list of the lines in the file"""
    with open(filename) as f:
        lines = list(csv.reader(f, delimiter=" "))[1:]
        return lines


def open_domain_file(filename: str):
    """Open the domain file and return all the data in lists
        filename: the name of the file to open
        return the list of the lines in the file"""
    with open(filename) as f:
        lines = list(csv.reader(f, delimiter="\t"))[1:]
        return lines
    

def splits_transposons(filename: str):
    """Split the transposons file into multiple files
        Each file indicates one chromosome
        filename: the name of the file to split"""
    transposons = open_transposon_file(filename)
    lines = []
    number = 1
    for c, line in enumerate(transposons):
        lines.append(line)
        if c+1 != len(transposons) and "variableStep" in transposons[c+1]:
            with open("Domains/transposons/transposon_chr" + roman.toRoman(number) + ".wig", "w", newline = '') as f:
                csv.writer(f, delimiter=" ").writerows(lines)
            lines = []
            number += 1
        
    with open("Domains/transposons/transposon_chrmt" + ".wig", "w", newline = '') as f:
                csv.writer(f, delimiter=" ").writerows(lines)
            
    
def find_location_transposon(transposons: list, domains: list):
    """Find the location of the transposons in the domain file
        The transposons and domain files are sorted so you can loop from the beginning
        filename_tranposon: the name of the transposon file
        filename_domain: the name of the domain file
        return the domains with the transposons added and the number of transposons that are not in a domain
        """    
    not_in_domain = [0,0]
    found = True
    cur_row = 0
    
    for transposon in transposons:
        if(not found):
            not_in_domain[0] += 1
            not_in_domain[1] += transposon_n 
        found = False
        
        transposon_v = int(transposon[0])            
        transposon_n = int(transposon[1])
        
        # If the transposon is not in the domain, move to the next domain
        while(cur_row < len(domains) and int(domains[cur_row][2]) < transposon_v):
            cur_row += 1
            
        
        if cur_row >= len(domains):
            continue
        
        i_row = cur_row
        i_start = int(domains[i_row][1])
        i_end = int(domains[i_row][2])    
        
        # Add transposons to domain until the transposon is larger than the start of the domain
        while(i_row < len(domains) and int(domains[i_row][1]) <= transposon_v):
            
            if(i_start <= transposon_v and transposon_v <= i_end):
                found = True
                
                if(len(domains[i_row]) == 5):
                    domains[i_row].append(transposon_n)
                else:
                    domains[i_row][5] = str(int(domains[i_row][5]) + transposon_n)
                    
            i_row += 1
        
    if(not found):
        not_in_domain[0] += 1
        not_in_domain[1] += transposon_n 

    return domains, not_in_domain


def write_transposons_to_domains(transposon_directory: str, domain_directory: str):
    """Loop over all the domain files and then loop over all the transposons files
        To find the location of the transposons in all the domains
        Write the results to a new file
        Also keep track of all the transposons that are not in a domain
        transposon_directory: the directory of all the transposon files
        domain_directory: the directory of all the domain files"""
    not_in_domain = {"total": [0,0]}
    
    for domain_database in os.listdir(domain_directory):
        subdirectory = os.path.join(domain_directory, domain_database)
        for chromosome in os.listdir(subdirectory):
            chrom = chromosome[11:-4]
            f1 = os.path.join(subdirectory, chromosome)
            if os.path.isfile(f1):
                domains = open_domain_file(f1)
                for filename_transposon in os.listdir(transposon_directory):
                    # print("Transposon: " + filename_transposon[11:-4], "Chromosome: " + chrom)
                    if chrom == filename_transposon[11:-4]:
                        f2 = os.path.join(transposon_directory, filename_transposon)
                        if os.path.isfile(f2):
                            transposons = open_transposon_file(f2)
                            print("Finding location of transposons in domain: ", domain_database, 
                                "In the following chromsome: " + chrom +  
                                " and transposon: ", filename_transposon)
                            domains, not_present = find_location_transposon(transposons, domains)
                            subdirectory_name = subdirectory[23:]
                            if(subdirectory_name not in not_in_domain):
                                not_in_domain[subdirectory_name] = {"total": [0,0]}
                            not_in_domain[subdirectory_name][chrom] = [not_present[0], not_present[1]]
                            not_in_domain[subdirectory_name]["total"][0] += not_present[0]
                            not_in_domain[subdirectory_name]["total"][1] += not_present[1]
                            not_in_domain["total"][0] += not_present[0]
                            not_in_domain["total"][1] += not_present[1]
                    with open(f"Domains/deleted_data/{subdirectory_name}/transposon_result_{chrom}.txt", "w", newline = '') as f:
                        f.write("Gene\tstart\tend\tdescription\tdatabase_name\ttransposons\n")
                        csv.writer(f, delimiter="\t").writerows(domains)
    with open("Domains/deleted_data/not_in_domain.json", "w", newline = '') as f:
        f.write(json.dumps(not_in_domain, indent = 4))
                             
    
def normalize_transposons(domains: list):
    """Normalize the transposons by the length of the domain
        domains: the list of domains
        return the list of domains with the normalized transposons added"""
    for domain in domains:
        if(len(domain) >= 6):
            del domain[6:]
            normalize_transposon = round(int(domain[5]) / (int(domain[2]) - int(domain[1])), 3)
            domain.append(normalize_transposon)
        else:
            del domain[5:]
            domain.append(0)
            domain.append(0)
    return domains 


def final_files(domain_directory: str):
    """Write the final result with the normalized transposons added
        domain_directory: the directory of all the domain files"""
    for domain in os.listdir(domain_directory):
        f1 = os.path.join(domain_directory, domain)
        if os.path.isdir(f1):
            for chromosome in os.listdir(f1):
                f2 = os.path.join(f1, chromosome)
                if os.path.isfile(f2):
                    domains = open_domain_file(f2)
                    domains = normalize_transposons(domains)
                    with open(f2, "w", newline = '') as f:
                        f.write("Gene\tstart\tend\tdescription\tdatabase_name\ttransposons\ttransposons_normalized\n")
                        csv.writer(f, delimiter="\t").writerows(domains)


def combine_functions(transposon_directory: str, domain_directory: str, domain_result: str):
    """Combine all the functions that find the transposons and normalize it
        transposon_directory: the directory of all the transposon files
        domain_directory: the directory of all the domain files
        domain_result: the directory of all the domain files with the transposons added"""
    write_transposons_to_domains(transposon_directory, domain_directory)
    final_files(domain_result)
    


if __name__ == "__main__":
    # splits_transposons("Domains/Initial_datasets/transposon_file.wig")
    # print(roman.toRoman(10))
    # trial("Domains/filtered_files")
    write_transposons_to_domains("Domains/transposons", "Domains/filtered_files")
    final_files("Domains/deleted_data")