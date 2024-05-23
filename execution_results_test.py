import pandas as pd
from collections import Counter

if __name__ == '__main__':
    def min_max_values(csv_file_path, date_columns=[], substr_length=3):
        # Read the CSV file into a pandas dataframe
        df = pd.read_csv(csv_file_path, parse_dates=date_columns)

        # Initialize a dictionary to store min and max values
        min_max_dict = {}

        # Iterate over each column in the dataframe
        for column in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[column]):
                    min_value = df[column].min()
                    max_value = df[column].max()
                    min_max_dict[column] = {'min': min_value, 'max': max_value}
                elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
                    unique_values = df[column].dropna().unique().tolist()
                    min_max_dict[column] = {'possible_values': unique_values}
                     # Count starting characters/substrings
                    starting_substrs = df[column].dropna().apply(lambda x: x[:substr_length])
                    substr_counts = Counter(starting_substrs)
                    # Sort the counts in descending order
                    sorted_substr_counts = dict(sorted(substr_counts.items(), key=lambda item: item[1], reverse=True))
                    min_max_dict[column]['startswith'] = sorted_substr_counts
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    min_date = df[column].min()
                    max_date = df[column].max()
                    min_max_dict[column] = {'min_date': min_date, 'max_date': max_date}
            except Exception as e:
                print(f"Error processing column '{column}': {e}")
                continue
        
        # Print the min, max values, and possible values for enum type columns
        for column, values in min_max_dict.items():
            print(f"Column: {column}\n")
            if 'min' in values and 'max' in values:
                print(f"  Min: {values['min']}\n")
                print(f"  Max: {values['max']}\n")
            '''
            if 'possible_values' in values:
                print(f"  Possible values: {values['possible_values']}\n")
            '''
            if 'min_date' in values and 'max_date' in values:
                print(f"  Min Date: {values['min_date']}\n")
                print(f"  Max Date: {values['max_date']}\n")
            if 'startswith' in values:
                print(f"  Startswith frequencies: {values['startswith']}\n")
        
    # Call the function with the path to your CSV file
    min_max_values("./benchmarks/customer.csv")
    min_max_values("./benchmarks/lineitem.csv", date_columns=['SHIPDATE', 'COMMITDATE', 'RECEIPTDATE'])
    min_max_values("./benchmarks/nation.csv")
    min_max_values("./benchmarks/orders.csv", date_columns=['ORDERDATE'])
    min_max_values("./benchmarks/part.csv")
    min_max_values("./benchmarks/partsupp.csv")
    min_max_values("./benchmarks/region.csv")
    min_max_values("./benchmarks/supplier.csv")

    