# Query Generator Prototype

## Overview
This query generator is designed to help users generate synthetic queries for training machine learning models that estimate query execution costs or predict cardinality.

## Additional features
- **Configurable Query Generation:** Supports customization of query types, number of merges per query, number of queries and query complexity (query_parameters.json)
- **Sample Dataset Generation:** Automatically generates datasets to test the queries.
- **Execution Metrics Collection:** Executes generated queries to collect labels for ML models (merged/unmerged_query_exeuction_results.csv)
- **Integration with ML Frameworks:** Provides example pipelines for training and evaluation (scikit-learn, TensorFlow, or PyTorch)  (might be too complex for scope of this project)
- **Build an interactive GUI:** Develop a graphical user interface (GUI) for: loading relational schema, setting query generation parameters interactively, previewing generated queries
- **Support various output file formats** Support various output formats for generated queries and execution metrics(queries, execution_time, cardinality, valid): JSON, CSV
-**API integration** Integrate API for dynamic schema retrivial and Cross-Platform Query Compatibility? (building a webpage)

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
	-removed selection with startswith condition on strings because too restrictive (produces empty result sets)
	-do not generate merge queries on columns with different data ranges (add ranges_overlap method to pandas_query_pool and call it when checking merge columns)
	-66/1500 for unmerged, 49/500 for merged queries with empty result set
	-in relational schema, remove startswith condition on strings
    
-extend relational schema with date and enum type attributes with range constraints
-generate queries with date and enum type conditions:
	-include selection on dates with range conditions (e.g. SHIPDATE between ‘1994-01-01’ and ‘1994-12-31’)
	-include selection on enums with IN condition (e.g. ORDERPRIORITY IN (‘1-URGENT’, ‘2-HIGH’))
-if possible, make the input format for the relational schema more convenient (perhaps use PySimpleGUI)

