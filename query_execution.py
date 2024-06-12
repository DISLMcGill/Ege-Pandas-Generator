 
import pandas as pd
import os
import re
import time

# Load datasets
customer = pd.read_csv("./benchmarks/customer.csv")
lineitem = pd.read_csv("./benchmarks/lineitem.csv")
nation = pd.read_csv("./benchmarks/nation.csv")
orders = pd.read_csv("./benchmarks/orders.csv")
part = pd.read_csv("./benchmarks/part.csv")
partsupp = pd.read_csv("./benchmarks/partsupp.csv")
region = pd.read_csv("./benchmarks/region.csv")
supplier = pd.read_csv("./benchmarks/supplier.csv")

# Local dictionary defining the context for pd.eval()
local_dict = {
    'customer': customer,
    'lineitem': lineitem,
    'nation': nation,
    'orders': orders,
    'part': part,
    'partsupp': partsupp,
    'region': region,
    'supplier': supplier
}

# Function to execute single-line queries
def execute_merged_queries(dir, filename):
    """execute merged queries in merged_queries_auto_sf0000.txt on the 
    datasets in benchmarks folder (customer.csv, lineitem.csv, etc.)"""
    # Read the merged queries file
    with open("results/merged_queries_auto_sf0000.txt", 'r') as file:
        merged_queries = file.readlines()

    # Store the query, whether or not it is valid, execution time, and cardinality of the result set in a text file
    try:
        f = open(f"{dir}/{filename}.txt", "a")
    except:
        filepath = f"{dir}/{filename}.txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        f = open(f"{dir}/{filename}.txt", "a")

    f.write("Query, Valid, Execution Time, Cardinality, Selections, Projections, Group by, Aggregations \n")

    # Regular expressions to match selections and projections
    selection_pattern = re.compile(r'\b\w+\[\((.*?)\)\]')
    projection_pattern = re.compile(r'\[\[.*?\]\]')

    # Iterate over each merged query and execute the query on the appropriate dataset
    for query in merged_queries:
        query_string = query.split('=', 1)[1].strip()

        start = time.time()
        result = pd.eval(query_string, local_dict=local_dict)
        end = time.time()

        print(result)
        f.write(f"{query_string}, ")

        # Query is valid if the result set is non-empty
        if len(result) == 0:
            f.write(f"{False}, ")
        else:
            f.write(f"{True}, ")

        # Write query execution time and cardinality of the result set
        f.write(f"{end - start}, ")
        f.write(f"{len(result)},")
        
        # Count the number of merges, group by, and aggregations
        num_merges = query_string.count("merge")
        num_groupby = query_string.count("groupby")
        num_agg = query_string.count("agg")

        # Count selections and projections using regular expressions
        num_selections = len(selection_pattern.findall(query_string))
        num_projections = len(projection_pattern.findall(query_string))

        # Write counts of operations
        f.write(f"{num_selections},")
        f.write(f"{num_projections},")
        f.write(f"{num_merges},")
        f.write(f"{num_groupby},")
        f.write(f"{num_agg}\n")

    f.close()
    file.close()

# Function to execute multi-line queries
def execute_merged_queries_multiline(dir, filename):
    """execute merged queries in merged_queries_auto_sf0000.txt on the 
    datasets in benchmarks folder (customer.csv, lineitem.csv, etc.)"""
    # Read the merged queries file
    with open("results/merged_queries_auto_sf0000.txt", 'r') as file:
        merged_queries = file.readlines()

    # Store the query, whether or not it is valid, execution time, and cardinality of the result set in a text file
    try:
        f = open(f"{dir}/{filename}.txt", "a")
    except:
        filepath = f"{dir}/{filename}.txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        f = open(f"{dir}/{filename}.txt", "a")

    f.write("Query, Valid, Execution Time, Cardinality, Selections, Projections, Group by, Aggregations \n")

    # Regular expressions to match selections and projections
    selection_pattern = re.compile(r'\b\w+\[\((.*?)\)\]')
    projection_pattern = re.compile(r'\[\[.*?\]\]')

    # Collect and execute each unmerged query block
    current_query_block = []
    for query in merged_queries:
        if query.strip() != "Next":
            current_query_block.append(query.strip())
        else:
            if current_query_block:
                # Combine the current query block into a single query string
                combined_query = ""
                last_df_name = ""
                skip_next = False
                for i, subquery in enumerate(current_query_block):
                    if skip_next:
                        skip_next = False
                        continue
                        
                    if "=" in subquery:
                        df_name, subquery_body = subquery.split('=', 1)
                        df_name = df_name.strip()
                        subquery_body = subquery_body.strip()

                        # Ensure column names in merge are quoted
                        if "merge" in subquery_body:
                            subquery_body = subquery_body.replace("left_on=", "left_on='").replace(", right_on=", "', right_on='").replace(")", "')")

                        if "groupby" in subquery_body:
                            combined_query += f"\n{df_name} = {subquery_body}\n"
                            last_df_name = df_name
                            if i + 1 < len(current_query_block) and "agg" in current_query_block[i + 1]:
                                agg_query = current_query_block[i + 1]
                                agg_df_name, agg_body = agg_query.split('=', 1)
                                combined_query += f"{agg_df_name.strip()} = {agg_body.strip()}"
                                last_df_name = agg_df_name.strip()
                                skip_next = True
                        else:
                            if i == 0:
                                combined_query = f"{df_name} = {subquery_body}"
                            else:
                                combined_query += f"\n{df_name} = {subquery_body}"
                            last_df_name = df_name

                try:
                    start = time.time()
                    exec(combined_query, local_dict)
                    result = local_dict[last_df_name]
                    end = time.time()

                    print(result)
                    f.write(f"{combined_query}, ")

                    # Query is valid if the result set is non-empty
                    if len(result) == 0:
                        f.write(f"{False}, ")
                    else:
                        f.write(f"{True}, ")

                    # Write query execution time and cardinality of the result set
                    f.write(f"{end - start}, ")
                    f.write(f"{len(result)},")

                    # Count the number of merges, group by, and aggregations
                    num_merges = combined_query.count("merge")
                    num_groupby = combined_query.count("groupby")
                    num_agg = combined_query.count("agg")

                    # Count selections and projections using regular expressions
                    num_selections = len(selection_pattern.findall(combined_query))
                    num_projections = len(projection_pattern.findall(combined_query))

                    # Write counts of operations
                    f.write(f"{num_selections},")
                    f.write(f"{num_projections},")
                    f.write(f"{num_merges},")
                    f.write(f"{num_groupby},")
                    f.write(f"{num_agg}\n")

                except Exception as e:
                    print(f"Error executing query block: {e}")
                    f.write(f"{combined_query}, {False}, 0, 0\n")
                
                # Reset for the next query block
                current_query_block = []
    
    f.close()
    file.close()
    

if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description='Execute queries')
    parser.add_argument('--params', type=str, required=True, help='Path to the user-defined parameters JSON file')

    args = parser.parse_args()

    with open(args.params, 'r') as pf:
        params = json.load(pf)

    multi_line = params.get('multi_line', False) == "True"

    #Execute queries
    if multi_line:
        execute_merged_queries_multiline(dir="./results", filename="merged_query_execution_results.csv")
    else:
        execute_merged_queries(dir="./results", filename="merged_query_execution_results.csv")

