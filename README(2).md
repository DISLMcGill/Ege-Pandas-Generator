# Query Generator Prototype

## Overview
This query generator is designed to help users generate synthetic queries for training machine learning models that estimate query execution costs or predict cardinality.

## Additional features
- **Configurable Query Generation:** Supports customization of query types, number of merges per query, number of queries and query complexity (query_parameters.json)
- **Sample Dataset Generation:** Automatically generates datasets to test the queries.
- **Execution Metrics Collection:** Executes generated queries to collect labels for ML models (merged/unmerged_query_exeuction_results.csv)
- **Integration with ML Frameworks:** Provides example pipelines for training and evaluation (scikit-learn, TensorFlow, or PyTorch)  
- **Build an interactive GUI:** Develop a graphical user interface (GUI) for: loading relational schema, setting query generation parameters interactively, previewing generated queries
- **Support various output file formats** Support various output formats for generated queries and execution metrics(queries, execution_time, cardinality, valid): JSON, CSV
-**API integration** Integrate API for dynamic schema retrivial and Cross-Platform Query Compatibility? (building a webpage)
-**Query generation to load data into dataframes** output queries that load data following a given distribution and relational schema into a dataframe so that users who want to test their ml models can use the query generator to get as much data as they want

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Function to generate pandas queries for synthetic data generation
def generate_pandas_queries(entity_name, entity_schema, num_rows):
    queries = []

    for column, properties in entity_schema['properties'].items():
        col_type = properties['type']
        if col_type == 'int':
            query = f"{entity_name}['{column}'] = np.random.randint({properties['min']}, {properties['max']} + 1, {num_rows})"
        elif col_type == 'float':
            query = f"{entity_name}['{column}'] = np.round(np.random.uniform({properties['min']}, {properties['max']}, {num_rows}), 2)"
        elif col_type == 'enum':
            values = ', '.join([f'"{v}"' for v in properties['values']])
            query = f"{entity_name}['{column}'] = np.random.choice([{values}], {num_rows})"
        elif col_type == 'string':
            start_chars = ', '.join([f'"{ch}"' for ch in properties['starting character']])
            query = (f"{entity_name}['{column}'] = [''.join([random.choice([{start_chars}]) + "
                     f"''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=10))]) "
                     f"for _ in range({num_rows})]")
        elif col_type == 'date':
            start_date = datetime.strptime(properties['min'], '%Y-%m-%d')
            end_date = datetime.strptime(properties['max'], '%Y-%m-%d')
            query = (f"{entity_name}['{column}'] = [(datetime.strptime('{properties['min']}', '%Y-%m-%d') + "
                     f"timedelta(days=random.randint(0, (datetime.strptime('{properties['max']}', '%Y-%m-%d') - "
                     f"datetime.strptime('{properties['min']}', '%Y-%m-%d')).days))) for _ in range({num_rows})]")
        else:
            raise ValueError(f"Unsupported type {col_type} for column {column}")
        
        queries.append(query)
    
    return queries

# Example schema for one of the entities (perhaps specify distribution of ints and floats)
schema = {
    "entities": {
        "customer": {
            "properties": { 
                "CUSTKEY": { "type": "int", "min": 1, "max": 100 },
                "C_NAME": { "type": "string", "starting character": ["C"] },
                "ADDRESS": { "type": "string", "starting character": ["I", "H", "X", "s", "9", "n", "z", "K", "T", "u", "Q", "O", "7", "o", "M", "c", "i", "3", "8", "L", "g", "4", "m", "S", "E", "x", "6", "P", "Y", "J", "j", "q", "a", "e", "b", "0", ",", "B", "F", "R", "r", "p", "D", "l", "U", "h", "w", "d", "v", "f"] },
                "NATIONKEY": { "type": "int", "min": 0, "max": 23 },
                "PHONE": { "type": "string", "starting character": ["1", "2", "3", "25-", "13-", "27-", "18-", "22-"] },
                "ACCTBAL": { "type": "float", "min": -917.25, "max": 9983.38 },
                "MKTSEGMENT": { "type": "enum", "values": ["BUILDING", "AUTOMOBILE", "MACHINERY", "HOUSEHOLD", "FURNITURE"] },
                "C_COMMENT": { "type": "string", "starting character": ["i", " ", "s", "l", "r", "c", "t", "e", "o", "n", "a", "p", "h", "u", "k", "g", "y", ".", ",", "d", "f", "q", "w"] }
            },
            "primary_key": "CUSTKEY",
            "foreign_keys": { "NATIONKEY": "nation" }
        },
        # Define other entities similarly...
    }
}

# Function to execute the generated queries
def execute_queries(entity_name, queries, num_rows):
    local_vars = {
        'np': np,
        'random': random,
        'datetime': datetime,
        'timedelta': timedelta,
        entity_name: pd.DataFrame(index=range(num_rows))
    }
    exec("\n".join(queries), {}, local_vars)
    return local_vars[entity_name]

# Generate and execute pandas queries for each entity
for entity_name, entity_schema in schema['entities'].items():
    queries = generate_pandas_queries(entity_name, entity_schema, num_rows=100)  # Generate queries for 100 rows of data
    df = execute_queries(entity_name, queries, num_rows=100)
    print(f"DataFrame for {entity_name}:\n", df.head(), "\n")



## Quick Start
### Prerequisites
- Python 3.6+
- Run the query generator with the following command:
- `python query_generator.py --schema data_structure.json --params query_parameters.json`

## Changes made from last prototype

May 9th meeting:
-Generating ideas on how to improve current prototype in README(2).md

May 13th meeting:
-Implement query_parameters.json configuration file (query types, number of merges per query, number of queries, query complexity)
-Update main function to load parameters from query_parameters.json 
-modified gen_base_queries to generate different base queries depending on query types
-updated generate_merge_operations to check number of queries and number of merges per query parameters 
-Implemented command line interface in main function
-In gen_queries, generate combinations ensuring they meet the complexity requirements (number of unmerged operations per query)

May 16th meeting:
-Tested executed queries on TPC-H datasets (customer, lineitem, nation, orders, …)
-Created data_structure_tpch.json file to store relational schema on TPC-H datasets
-Wrote execute_unmerged/merged_queries function in main which executes the generated queries on the TPC-H datasets and outputs execution metrics (valid, cardinality, execution time) in unmerged/merged_query_execution_results.csv

May 23rd meeting:
-make sure result set is non-empty:
	-for selections,  no == or != conditions on floats, only >, <, >=, <= (updated possible_selections and get_a_selection methods)
	-for selections, no > or >= max_value conditions and no < or <= min_value conditions on ints or floats (updated possible_selections and get_a_selection methods)
	-make sure selection conditions are logically consistent (is_logically_consistent method in operation(selection) class and is_consistent_with method in condition class)
	-for selection conditions on floats, round to 2 decimal places
	-do not generate merge queries on columns with different data ranges (add ranges_overlap method to pandas_query_pool and call it when checking merge columns)
	-66/1500 for unmerged, 49/500 for merged queries with empty result set
	-in relational schema, changed starting character range condition on strings to include a list of possible starting characters or substrings

-extend relational schema with date and enum type attributes with range constraints
-generate queries with date and enum type conditions:
	-updated selection with startswith condition on strings to choose from list of starting characters/substrings
	-include selection on dates with range conditions (e.g. SHIPDATE between ‘1994-01-01’ and ‘1994-12-31’): >,>=,<,<= operators
	-include selection on enums with IN condition (e.g. ORDERPRIORITY IN (‘1-URGENT’, ‘2-HIGH’)): ==, != and IN operators
	-75/1800 for unmerged, 148/500 for merged queries with empty result set due to equality and startswith selection conditions
	-large variation in number of False merged queries
-if possible, make the input format for the relational schema more convenient (perhaps use PySimpleGUI)

