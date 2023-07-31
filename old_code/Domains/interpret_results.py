import os
import numpy as np
import csv
import json
import requests


def count_transposons(directory: str):
    answer = {"total": [0, 0]}
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        with open(f) as transposons:
            lines = list(csv.reader(transposons, delimiter = ' '))[1:]
            arr = np.array(lines)
            arr = arr.astype('float64')
            counts = np.sum(arr, axis=0)
            number = np.size(arr, axis=0)
            answer[f[31:-4]] = [number, counts.tolist()[1]]
            answer["total"][0] += number
            answer["total"][1] += counts[1]
    with open("Domains/deleted_data/total_transposon_count.json", "w") as json_file:
        json.dump(answer, json_file, indent=4)
    return answer


def relative_not_in_domain(filename_domains: str, filename_total: str):
    result = {}
    with open(filename_domains) as f1:
        not_in_domain = json.load(f1)
    with open(filename_total) as f2:
        total = json.load(f2)
    for key in not_in_domain:
        for chromosome in not_in_domain[key]:
            if key != "total":
                if(key not in result):
                    result[key] = {}
                print(chromosome)
                result[key][chromosome] = [not_in_domain[key][chromosome][0]/total[chromosome][0], not_in_domain[key][chromosome][1]/total[chromosome][1]]
    with open("Domains/deleted_data/relative_not_in_domain.json", "w") as json_file:
        json.dump(result, json_file, indent=4)
            
            
# 1091284

def count_domainless_spaces(filename: str, number: int):
    total = 0
    with open(filename) as f:
        lines = list(csv.reader(f, delimiter = '\t'))[1:]
        for line in lines:
            line[1] = int(line[1])
            line[2] = int(line[2])
        answer = []
        current = 0
        for line in lines:
            add = line[1] - current
            if(add > 0):
                answer.append((current, line[1]))
                current = line[2]
                total += add
        if(current < number):
            answer.append((current, number))
    # print(answer)
    print(total)
    return total
        
            
def check_smth(filename: str, not_domains: list):
    with open(filename) as f:
        lines = list(csv.reader(f, delimiter = ' '))[1:]
    count = 0
    transposon_in_domain = 0
    transposon_not_in_domain = 0
    count2 = 0
    for line in lines:
        for domain in not_domains:
            if(int(line[0]) > domain[0] and int(line[0]) < domain[1]):
                print(line)
                print(domain)
                count += 1
                transposon_not_in_domain += int(line[1])
                print(count)
                break
        else: 
            count2 += 1
            transposon_in_domain += int(line[1])
    print("final")
    print("count2: ", count2, "count: ", count)
    print("transposon_in_domain: ", transposon_in_domain, "transposon_not_in_domain: ", transposon_not_in_domain)
    print(transposon_not_in_domain / (transposon_in_domain + transposon_not_in_domain))
    
    
def average_transposons(filename: str):
    answer = np.zeros(17)
    with open(filename) as jsonfile:
        data = json.load(jsonfile)
    count = 0
    chromosomes = []
    for key in data:
        c = 0
        count += 1
        for chromosome in data[key]:
            if(chromosome != "total"):
                print(chromosome)
                answer[c] += float(data[key][chromosome])
                if(count == 1):
                    chromosomes.append(chromosome)
                print(answer)
                c += 1
    final = answer / count
    print(final)
    print(chromosomes)
    return final


def count_transposons_2(directory: str):
    max_value = 0
    min_value = float("inf")
    total = 0
    count = 0
    for filename in os.listdir(directory):
        filename = os.path.join(directory, filename)
        with open(filename) as f:
            data = list(csv.reader(f, delimiter = '\t'))[1:]
            for line in data:
                total += float(line[6])
                count += 1
                v = float(line[6])
                if(float(line[6]) > max_value):
                    max_value = float(line[6])
                    print(filename, line)
                if(float(line[6]) < min_value):
                    min_value = float(line[6])
    print(min_value, max_value)
    print(total/count)


def count_make_bins(directory: str):
    bins = np.zeros(12)
    for filename in os.listdir(directory):
        filename = os.path.join(directory, filename)
        with open(filename) as f:
            data = list(csv.reader(f, delimiter = '\t'))[1:]
            for line in data:
                if(float(line[6]) == 0):
                    print("small")
                    print(filename, line)
                    bins[0] += 1
                elif(float(line[6]) > 10):
                    bins[11] += 1
                    print("big")
                    print(filename, line)
                else:
                    bins[int(float(line[6])) + 1] += 1
    print(bins.tolist())
    
    
def connect_to_api(gene: str, addition: str):
    """
    This function connects to the SGD API and returns the data.
    gene: the name of the gene
    addition: the addition to the url
    return the data in a json format
    """
    SGD_BASE_URL = 'https://www.yeastgenome.org/backend/locus/'
    url = SGD_BASE_URL + gene + addition
    print(gene)
    try:
        response = requests.get(url=url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"Gene: {gene} not found in the SGD database.")
        raise SystemExit(err)
    
    return response.json()


def get_essential(gene: str):
    """
    This function returns the essentiality of the gene.
    gene: the name of the gene
    return the essentiality of the gene
    """
    data = connect_to_api(gene, "")
    print(data["phenotype_overview"]["paragraph"])
    if(data["phenotype_overview"]["paragraph"] == None):
        return "unknown"
    if "Essential gene" in data["phenotype_overview"]["paragraph"].split(";")[0]:
        return "essential"
    elif "Non-essential gene" in data["phenotype_overview"]["paragraph"].split(";")[0]:
        return "non-essential"
    else: 
        return "unknown"


def make_overview(directory: str):
    answer = {}
    global_dict = {}
    for filename in os.listdir(directory):
        filename = os.path.join(directory, filename)
        with open(filename) as f:
            data = list(csv.reader(f, delimiter = '\t'))[1:]
            for line in data:
                if(float(line[6]) == 0):
                    if(line[0] in global_dict):
                        key2 = global_dict[line[0]]
                    else:
                        key2 = get_essential(line[0])
                        global_dict[line[0]] = key2
                    answer = update_dictionary(answer, "0", key2)
                elif(float(line[6]) > 10):
                    if(line[0] in global_dict):
                        key2 = global_dict[line[0]]
                    else:
                        key2 = get_essential(line[0])
                        global_dict[line[0]] = key2
                    answer = update_dictionary(answer, ">10", key2)
                elif(float(line[6]) > 0 and float(line[6]) <= 0.5):
                    if(line[0] in global_dict):
                        key2 = global_dict[line[0]]
                    else:
                        key2 = get_essential(line[0])
                        global_dict[line[0]] = key2
                    answer = update_dictionary(answer, "0-0.5", key2)
                elif(float(line[6]) > 0.5 and float(line[6]) <= 1):
                    if(line[0] in global_dict):
                        key2 = global_dict[line[0]]
                    else:
                        key2 = get_essential(line[0])
                        global_dict[line[0]] = key2
                    answer = update_dictionary(answer, "0.5-1", key2)
                elif(float(line[6]) > 1 and float(line[6]) <= 2):
                    if(line[0] in global_dict):
                        key2 = global_dict[line[0]]
                    else:
                        key2 = get_essential(line[0])
                        global_dict[line[0]] = key2
                    answer = update_dictionary(answer, "1-2", key2)
                elif(float(line[6]) > 2 and float(line[6]) <= 3):
                    if(line[0] in global_dict):
                        key2 = global_dict[line[0]]
                    else:
                        key2 = get_essential(line[0])
                        global_dict[line[0]] = key2
                    answer = update_dictionary(answer, "2-3", key2)
                elif(float(line[6]) > 3 and float(line[6]) <= 4):
                    if(line[0] in global_dict):
                        key2 = global_dict[line[0]]
                    else:
                        key2 = get_essential(line[0])
                        global_dict[line[0]] = key2
                    answer = update_dictionary(answer, "3-4", key2)
                elif(float(line[6]) > 4 and float(line[6]) <= 5):
                    if(line[0] in global_dict):
                        key2 = global_dict[line[0]]
                    else:
                        key2 = get_essential(line[0])
                        global_dict[line[0]] = key2
                    answer = update_dictionary(answer, "4-5", key2)
                elif(float(line[6]) > 5 and float(line[6]) <= 10):
                    if(line[0] in global_dict):
                        key2 = global_dict[line[0]]
                    else:
                        key2 = get_essential(line[0])
                        global_dict[line[0]] = key2
                    answer = update_dictionary(answer, "5-10", key2)
        print("Chromosome: ", filename)
        with open("Domains/deleted_data/overview.json", "w") as f:
            json.dump(answer, f, indent=4)
                    

def update_dictionary(dictionary: dict, key: str, key2: str):
    if(key in dictionary):
        if(key2 in dictionary[key]):
            dictionary[key][key2] += 1
        else:
            dictionary[key][key2] = 1
    else:
        dictionary[key] = {key2: 1}
    return dictionary



def count_all_non_domains(directory: str):
    answer = {}
    for filename in os.listdir(directory):
        new_path = os.path.join(directory, filename)
        if os.path.isdir(new_path):
            domain_name = new_path[21:]
            answer[domain_name] = {"total": 0}
            for source in os.listdir(new_path):
                final_path = os.path.join(new_path, source)
                chromosome = final_path[(len(new_path)+19):-4]
                total = count_domainless_spaces(final_path, chromosome_sizes[chromosome])
                answer[domain_name][chromosome] = total / chromosome_sizes[chromosome]
                answer[domain_name]["total"] += total
            answer[domain_name]["total"] /= 12157105
    with open("Domains/deleted_data/outside_domain.json", "w") as json_file:
        json.dump(answer, json_file, indent=4)


def count_relative_non_domains(filename1: str, filename2: str):
    answer = {}
    with open(filename1) as f1:
        domainless = json.load(f1)
    with open(filename2) as f2:
        relative_insertions = json.load(f2)
    
    for domain in domainless:
        answer[domain] = {}
        for chromosome in domainless[domain]:
            answer[domain][chromosome] = float(relative_insertions[domain][chromosome][1]) / float(domainless[domain][chromosome]) 
    with open("Domains/deleted_data/final_thingys.json", "w") as json_file:
        json.dump(answer, json_file, indent=4)
            
                
    
# 565596 28025652


chromosome_sizes = {"chrI": 230218, "chrII": 813184, "chrIII": 316620, "chrIV": 1531933, "chrIX": 439888,
              "chrmt": 85779, "chrV": 576874, "chrVI": 270161, "chrVII": 1090940,
                "chrVIII": 562643, "chrX": 745751, "chrXI": 666816, "chrXII": 1078177, "chrXIII": 924431,
                "chrXIV": 784333, "chrXV": 1091291, "chrXVI": 948066}



if __name__ == '__main__':
    # check_smth("Domains/transposons/transposon_chrX.wig",count_domainless_spaces("Domains/domains_result/Pfam/transposon_result_chrX.txt", 60372))
    # average_transposons("Domains/deleted_data/relative_not_in_domain.json")
    # count_transposons("Domains/transposons/")
    # relative_not_in_domain("Domains/deleted_data/not_in_domain.json", "Domains/deleted_data/total_transposon_count.json")
    # count_make_bins("Domains/deleted_data/Pfam/")
    # make_overview("Domains/deleted_data/Pfam/")
    # count_domainless_spaces("Domains/deleted_data/Pfam/transposon_result_chrI.txt", 230218)
    # count_all_non_domains("Domains/deleted_data")
    # count_all_non_domains("Domains/deleted_data")
    # count_relative_non_domains("Domains/deleted_data/outside_domain.json", "Domains/deleted_data/relative_not_in_domain.json")
    average_transposons("Domains/deleted_data/final_thingys.json")