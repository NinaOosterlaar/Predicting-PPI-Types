import csv

def werner_count_activation_inhibition():
    # result {'activation': 28, 'inhibition': 32}
    count = {"activation": 0, "inhibition": 0}
    with open("interactions/Data/reference_result_werner.csv", 'r') as f:
        lines = list(csv.reader(f, delimiter=","))
    for line in lines:
        if("activating" in line):
            count["activation"] += 1
        elif("inhibiting" in line):
            count["inhibition"] += 1
    return count


if __name__ == "__main__":
    print(werner_count_activation_inhibition())