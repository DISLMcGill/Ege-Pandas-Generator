import pandas as pd

if __name__ == '__main__':
    def min_max_values(csv_file_path):
        # Read the CSV file into a pandas dataframe
        df = pd.read_csv(csv_file_path)

        # Initialize a dictionary to store min and max values
        min_max_dict = {}

        # Iterate over each column in the dataframe
        for column in df.columns:
            try:
                min_value = df[column].min()
                max_value = df[column].max()
            except Exception as e:
                print(f"Error processing column '{column}': {e}")
                continue

            # Store the values in the dictionary
            min_max_dict[column] = {'min': min_value, 'max': max_value}
        
        # Print the min and max values for each column
        for column, values in min_max_dict.items():
            print(f"Column: {column}\n")
            print(f"  Min: {values['min']}\n")
            print(f"  Max: {values['max']}\n")
        
    # Call the function with the path to your CSV file
    min_max_values("./benchmarks/customer.csv")
    min_max_values("./benchmarks/lineitem.csv")
    min_max_values("./benchmarks/nation.csv")
    min_max_values("./benchmarks/orders.csv")
    min_max_values("./benchmarks/part.csv")
    min_max_values("./benchmarks/partsupp.csv")
    min_max_values("./benchmarks/region.csv")
    min_max_values("./benchmarks/supplier.csv")

    