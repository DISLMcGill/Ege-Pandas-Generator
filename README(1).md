# query-generator

### The pandas query generator script is used to produce various examples of sql-like pandas queries by performing different types of table operations.

### Usage:

1. input the schema information including entity(dataframe) name, data types of each column, data ranages, foreign_keys, primary key and the directories that you want to save the results. Note that the names must match in foreign keys

2. Associate the dataframe with a variable with the same name, and create TBL_source object with the pd.DataFrame object with the same name of the table.
    eg: globals()[entity] = create_dataframe(entity_schema, num_rows=2)
        tbl = TBL_source(globals()[entity], entity)

3. add all the foreign key pairs from foreign_keys dictionary

4. generate base queries, the base queries generation can be customized in the function gen_base_queries(), line 1143

5. for all the queries you have, function get_new_pandas_queries() will give you queries without merging.

6. merge can be performed with a pool of different queries (line 1286).

7. generate_possible_merge_operations() will generate merged queries.

8. steps 5-7 can be repeated to generate more queries.



2024.05.01, Hongxin Huo


Bugs reported:

### exception raised in selection with empty dataframes
    # fixed in 2024.02.21, by creating two rows of random data

### unable to connect to mimi server
    # fixed in 2024.02.25

### exception raised in selection with string data types
    # fixed in 2024.02.26

### data schema format 
    # fixed in 2024.03.05

### dataframe associated with variables not having the same name will stop execute_query from running
    # fixed in 2024.03.27

### db2 database unable to connect 
    #2024.04.07 used PostgreSQl database instead

### projection, aggregation operation lead to few merge operations  
    # having trouble fixing the bug, depends on the complexity of the schema and foreign key pairs 



### proposal
    # Integrate API for dynamic schema retrivial and Cross-Platform Query Compatibility



