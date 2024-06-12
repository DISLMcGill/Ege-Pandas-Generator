# Query Generator Prototype

## Overview
This query generator is designed to help users generate synthetic queries for training machine learning models that estimate query execution costs or predict cardinality.

## Quick start
### Prerequisites
- Python 3.6+

### Usage
- Create a JSON file storing the relational schema of the datasets with information about entities, their attributes, data types and ranges for each attribute, primary keys and foreign keys
- In the relational schema, do not include duplicate column names accross tables. For example, if you have customer and supplier entities with a PHONE attribute, rename the attributes to C_PHONE and S_PHONE

- Create another JSON file to store the following query generation parameters: 
    - num_selection: Maximum number of selection conditions per table, an integer from 0 to 3
    - projection: Whether to include projections, True or False
    - group by: Whether to include group by operations, True or False
    - aggregation: Whether to include aggregation operations, True or False
    - num_merges: Maximum number of merges in the generated queries, an integer from 0 to 5
    - num_queries: Number of queries to generate, accepts an integer from 1 to 5000
    - multi_line: Output format for the merged and unmerged queries. If set to "True", each output query is divided into multiple subqueries with one subquery on each line and the main queries are separated by a "Next" delimeter. If set to "False", the queries are output each on one line.
    - At least one of projections or selections must be included to be able to generate queries. 
    - If group by is set to True, then aggregation must also be set to True, since a groupby operation without an aggregation does not return a dataframe. 
    
- Run the query generator with the following command: `python query_generator.py --schema data_structure.json --params query_parameters.json`
- Replace data_structure.json and query_parameters.json with the actual file names for the relational schema and the query parameters

- The query generator first generates the queries using the specifications in the two input files and outputs a file called "merged_queries_auto_sf0000" with the resulting queries. 

### Example relational schema
{
    "entities": {
        "customer": {
            "properties": { "C_CUSTKEY": { "type": "int", "min": 1, "max": 100 }, "C_NAME": { "type": "string", "starting character": ["C"] }, "C_ADDRESS": { "type": "string", "starting character": ["I", "H", "X", "s", "9", "n", "z", "K", "T", "u", "Q", "O", "7", "o", "M", "c", "i", "3", "8", "L", "g", "4", "m", "S", "E", "x", "6", "P", "Y", "J", "j", "q", "a", "e", "b", "0", ",", "B", "F", "R", "r", "p", "D", "l", "U", "h", "w", "d", "v", "f"] }, "C_NATIONKEY": { "type": "int", "min": 0, "max": 23 }, "C_PHONE": { "type": "string", "starting character": ["1", "2", "3", "25-", "13-", "27-", "18-", "22-"] }, "C_ACCTBAL": { "type": "float", "min": -917.25, "max": 9983.38 }, "MKTSEGMENT": { "type": "enum", "values": ["BUILDING", "AUTOMOBILE", "MACHINERY", "HOUSEHOLD", "FURNITURE"] }, "C_COMMENT": { "type": "string", "starting character": ["i", " ", "s", "l", "r", "c", "t", "e", "o", "n", "a", "p", "h", "u", "k", "g", "y", ".", ",", "d", "f", "q", "w"] } },
            "primary_key": "C_CUSTKEY",
            "foreign_keys": { "C_NATIONKEY": ["N_NATIONKEY","nation"] }
        },
        "lineitem": {
            "properties": { "L_ORDERKEY": { "type": "int", "min": 1, "max": 194 }, "L_PARTKEY": { "type": "int", "min": 450, "max": 198344 }, "L_SUPPKEY": { "type": "int", "min": 22, "max": 9983 }, "LINENUMBER": { "type": "int", "min": 1, "max": 7 }, "QUANTITY": { "type": "int", "min": 1, "max": 50 }, "EXTENDEDPRICE": { "type": "float", "min": 1606.52, "max": 88089.08 }, "DISCOUNT": { "type": "float", "min": 0.00, "max": 0.10 }, "TAX": { "type": "float", "min": 0.00, "max": 0.08 }, "RETURNFLAG": { "type": "enum", "values": ["N", "R", "A"] }, "LINESTATUS": { "type": "enum", "values": ["O", "F"] }, "SHIPDATE": { "type": "date", "min": "1992-04-27", "max": "1998-10-30"  }, "COMMITDATE": { "type": "date", "min": "1992-05-15", "max": "1998-10-16" }, "RECEIPTDATE": { "type": "date", "min": "1992-05-02", "max": "1998-11-06" }, "SHIPINSTRUCT": { "type": "enum", "values": ["DELIVER IN PERSON", "TAKE BACK RETURN", "NONE", "COLLECT COD"] }, "SHIPMODE": { "type": "enum", "values": ["TRUCK", "MAIL", "REG AIR", "AIR", "FOB", "RAIL", "SHIP"] }, "L_COMMENT": { "type": "string", "starting character": [" ", "e", "s", "l", "t", "a", "n", "u", "y", "c", "i", ".", "r", "g", "p", "f", "o", "h", "q", "k", "j", ",", "b", "v", "-", "d", "ly ", " ca", " re", "s. ", "lit"] } },
            "primary_key": ["L_ORDERKEY", "L_PARTKEY", "L_SUPPKEY"],
            "foreign_keys": { "L_ORDERKEY": ["O_ORDERKEY","orders"], "L_PARTKEY": ["PS_PARTKEY","partsupp"], "L_SUPPKEY": ["PS_SUPPKEY","partsupp"] }
        },

        "nation": {
            "properties": { "N_NATIONKEY": { "type": "int", "min": 0, "max": 24 }, "N_NAME": { "type": "string", "starting character": ["I", "A", "C", "E", "J", "M", "R", "U", "B", "F", "G", "K", "P", "S", "V"] }, "N_REGIONKEY": { "type": "int", "min": 0, "max": 4 }, "N_COMMENT": { "type": "string", "starting character": [" ", "y", "e", "r", "s", "a", "v", "l", "n", "o", "i", "p", "c", "u", "t", "h"] } },
            "primary_key": "N_NATIONKEY",
            "foreign_keys": { "N_REGIONKEY": ["R_REGIONKEY","region"] }
        },
        "orders": {
            "properties": { "O_ORDERKEY": { "type": "int", "min": 1, "max": 800 }, "O_CUSTKEY": { "type": "int", "min": 302, "max": 149641 }, "ORDERSTATUS": { "type": "enum", "values": ["O", "F", "P"] }, "TOTALPRICE": { "type": "float", "min": 1156.67, "max": 355180.76 }, "ORDERDATE": { "type": "date", "min": "1992-01-13", "max": "1998-07-21" }, "ORDERPRIORITY": { "type": "enum", "values": ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"] }, "CLERK": { "type": "string", "starting character": ["C"] }, "SHIPPRIORITY": { "type": "int", "min": 0, "max": 0 }, "O_COMMENT": { "type": "string", "starting character" : [" ", "l", "e", "t", "s", "a", "i", "o", "n", "g", "u", "h", "c", "d", "r", "k", "y", "q", "b", ".", "f", "x", "z", "w", ",", "-", "j", "ly ", " re", "the", "egu", "uri"] } },
            "primary_key": "O_ORDERKEY",
            "foreign_keys": { "O_CUSTKEY": ["C_CUSTKEY","customer"] }
        },
        "part": {
            "properties": { "P_PARTKEY": { "type": "int", "min": 1, "max": 200 }, "P_NAME": { "type": "string", "starting character": ["b", "s", "l", "c", "m", "p", "g", "t", "a", "d", "h", "f", "i", "w", "n", "r", "o", "k", "v", "y", "cor", "bis", "blu", "lin", "lem"] }, "MFGR": { "type": "enum", "values": ["Manufacturer#1", "Manufacturer#2", "Manufacturer#3", "Manufacturer#4", "Manufacturer#5"] }, "BRAND": { "type": "enum", "values": ["Brand#13", "Brand#42", "Brand#34", "Brand#32", "Brand#24", "Brand#11", "Brand#44", "Brand#43", "Brand#54", "Brand#25", "Brand#33", "Brand#55", "Brand#15", "Brand#23", "Brand#12", "Brand#35", "Brand#52", "Brand#14", "Brand#53", "Brand#22", "Brand#45", "Brand#21", "Brand#41", "Brand#51", "Brand#31"] }, "TYPE": { "type": "string", "starting character": ["S", "M", "E", "P", "L", "STA", "SMA"] }, "SIZE": { "type": "int", "min": 1, "max": 49 }, "CONTAINER": { "type": "string", "starting character": ["JUMBO", "LG", "WRAP", "MED", "SM"] }, "RETAILPRICE": { "type": "float", "min": 901.00, "max": 1100.2 }, "PT_COMMENT": { "type": "string", "starting character": [" ", "e", "l", "s", "u", "i", "n", "o", "t", "a", "c", "p", "r", "k", "y", "h", "f", "m", "d", "b", "x", "!", "g", "w", "q", "ly ", "the", "kag", "ss ", " fi"] } },
            "primary_key": "P_PARTKEY"
        },
        "partsupp": {
            "properties": { "PS_PARTKEY": { "type": "int", "min": 1, "max": 50 }, "PS_SUPPKEY": { "type": "int", "min": 2, "max": 7551 }, "AVAILQTY": { "type": "int", "min": 43, "max": 9988 }, "SUPPLYCOST": { "type": "float", "min": 14.78, "max": 996.12 }, "P_COMMENT": { "type": "string", "starting character": [" ", "s", "l", "e", "r", "a", "t", "n", "i", "o", "u", "p", "b", "h", "y", "f", "g", "c", ",", "v", ".", "d", "x", "j", "k", "q", ";", "bli", "ly ", "are", " th", "the"] } },
            "primary_key": ["PS_PARTKEY", "PS_SUPPKEY"],
            "foreign_keys": { "PS_PARTKEY": ["P_PARTKEY","part"], "PS_SUPPKEY": ["S_SUPPKEY","supplier"] }
        },
        "region": {
            "properties": { "R_REGIONKEY": { "type": "int", "min": 0, "max": 4 }, "R_NAME": { "type": "string", "starting character": ["A", "E", "M", "AFR", "AME", "ASI"] }, "R_COMMENT": { "type": "string", "starting character": ["l", "h", "g", "u"] } },
            "primary_key": "R_REGIONKEY"
        },
        "supplier": {
            "properties": { "S_SUPPKEY": { "type": "int", "min": 1, "max": 200 }, "S_NAME": { "type": "string", "starting character": ["S"] }, "S_ADDRESS": { "type": "string", "starting character": ["N", "e", "f", "J", "o", "c", "b", "u", "p", "8", "q", "S", "Y", "i", "C", "g", "m", "L", "r", "W", "O", "7", "T", " ", "B", "G", "s", "9", "1", "H", "R", "y", "x", "Z", "z", "k", "j", "w", "I", "n", "M", "4", "5", "V", "F", "a", "l", "Q", "0", "U", "D", "h", "v", "2", "X", ",", "t", "E", "P", "6", "3", "d", "K"] }, "S_NATIONKEY": { "type": "int", "min": 0, "max": 24 }, "S_PHONE": { "type": "string", "starting character": ["1", "2", "3", "28-", "32-", "26-", "14-", "17-"] }, "S_ACCTBAL": { "type": "float", "min": -966.20, "max": 9915.24 }, "S_COMMENT": { "type": "string", "starting character": ["e", " ", "s", "a", "i", "r", "l", "u", "y", "n", "t", "c", "g", "h", "o", "d", "f", "x", "b", "k", ",", ".", "w", "!", "j", "v", "q", "the", "es ", " sl", "bli", "al "]  } },
            "primary_key": "S_SUPPKEY",
            "foreign_keys": { "S_NATIONKEY": ["N_NATIONKEY","nation"]}
        }
    }
}

### Example query parameters file
{
    "num_selections": 0,
    "projection": "True",
    "num_merges": 3,
    "group by": "True",
    "aggregation": "True",
    "num_queries": 1000,
    "multi_line": "True"
}

### Example program for query execution on TPC-H datasets

Below is an example program which executes queries generated for datasets in the TPC-H benchmark. The query execution can be replicated for any relational schema as follows:

1) Load the csv files for each dataframe and update the local dictionary defining the context for pd.eval

2) Loop through each of the output queries in "merged_queries_auto_sf0000". If the merged queries are output each on a single line, then we only need to extract the relevant part after the "=" sign (as in the execute_merged_queries() function). If each query is output on multiple lines, further processing is needed to combine the operations on each line into a single query (as in the execute_merged_queries_multiline() function). 

3) Execute the queries using pd.eval() and get the desired execution metrics

The program outputs execution metrics of merged and unmerged queries which include execution time, cardinality of the result set and number of each type of operation in a file called "merged_query_execution_results"

--------------------------------------------------------------------------------------------------------
 
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

---------------------------------------------------------------------------------------------------