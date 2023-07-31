import sys
sys.path.append("interactions/Python/Algorithm")
sys.path.append("interactions/Python/data_scripts")
from determine_ppi import protein_interactions_and_go_file
from data_creation import create_csv_file, compare_csv_results


def main():
    result = protein_interactions_and_go_file("interactions/Data/protein-interactions.csv", "interactions/Data/genes_info_go.json")
    create_csv_file(result, "interactions/Data/interactions_result.csv")
    compare_csv_results("interactions/Data/interactions_result.csv", "interactions/Data/reference_result_werner.csv")
    
def main2():
    result = protein_interactions_and_go_file("interactions/Data/protein.activation_inhibition_input.csv", "interactions/Data/genes_info_go_big.json")
    create_csv_file(result, "interactions/Data/interactions_result_big.csv")
    compare_csv_results("interactions/Data/interactions_result_big.csv", "interactions/Data/protein.activation_inhibition_new.csv")
    

if __name__ == '__main__':    
    main2()
