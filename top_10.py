''' This is a script for selecting top 10 models '''
import os
import pandas as pd

def find_and_concatenate_csvs(root_dir, output_file):
    all_dfs = []  # List to store all dataframes

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'results.csv':
                path = os.path.join(root, file)
                print(path)
                df = pd.read_csv(path)
                df['Path'] = path  # Add a column with the path
                all_dfs.append(df)

    # Concatenate all dataframes
    concatenated_df = pd.concat(all_dfs, ignore_index=True)
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved as {output_file}")
    return concatenated_df

# Usage
root_directory = 'path_to_your_directory'  # Replace with your root directory path
output_csv_file = 'concatenated_results.csv'  # Output file name
df = find_and_concatenate_csvs("mlartifacts", output_csv_file)
df.columns = df.columns.str.replace(' ', '')
df = df.reset_index(drop=True)
sorted_df = df.sort_values('metrics/mAP50-95(B)', ascending=False)
sorted_df.to_csv("test.csv")
print(sorted_df.head())
#print("Top metrics/mAP50-95(B):  ")
print("    ")
print("    ")
print(list(sorted_df["metrics/mAP50-95(B)"][0:1])[0])
print("    ")
print("    ")
print("Obtained with artefact: ",list(sorted_df["Path"][0:1])[0])

