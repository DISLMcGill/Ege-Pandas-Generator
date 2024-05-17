# Query Generator Prototype

## Overview
This query generator is designed to help users generate synthetic queries for training machine learning models that estimate query execution costs or predict cardinality.

## Additional features
- **Configurable Query Generation:** Supports customization of query types, database schemas, and complexity levels.  
- **Sample Dataset Generation:** Automatically generates datasets to test the queries.
- **Execution Metrics Collection:** Executes generated queries to collect labels for ML models.
- **Integration with ML Frameworks:** Provides example pipelines for training and evaluation (scikit-learn, TensorFlow, or PyTorch)  (might be too complex for scope of this project)
- **Build an interactive GUI:** Develop a graphical user interface (GUI) for: loading relational schema, setting query generation parameters interactively, previewing generated queries
- **Support various output file formats** Support various output formats for generated queries: JSON (queries and metadata), CSV, Python files
CSV/JSON: output queries with metadata like labels and features (queries, execution_time, cardinality, num_tables, complexity)
Python file would contain a script that loads data and executes generated queries
-**API integration** Integrate API for dynamic schema retrivial and Cross-Platform Query Compatibility? (building a webpage)

## Quick Start
### Prerequisites
- Python 3.6+
- `pip install -r requirements.txt`

### Generating Queries
1. **Generate Sample Queries:**
   ```bash
   python query_generator.py --config config.json

# Generated Pandas Queries
import pandas as pd
import json

# Sample Data Loading (Replace with Actual Data)
coach = pd.DataFrame({
    'Role': [8, 10, 5, 15, 4, 9],
    'National_name': ['Brazil', 'Germany', 'France', 'Italy', 'Spain', 'Portugal']
})
association = pd.DataFrame({
    'Association_name': ['CONMEBOL', 'UEFA', 'CONCACAF'],
    'National_name': ['Brazil', 'Germany', 'Mexico']
})

# Query Execution
# Query 1
df0 = coach[coach['Role'] == 8]
df1 = df0[['Role', 'National_name']]

# Query 2
df2 = association[['Association_name', 'National_name']]
df3 = df1.merge(df2, left_on='National_name', right_on='National_name')

# Query 3
df4 = coach[coach['Role'] <= 15]
df5 = df4[['Role', 'National_name']]
df6 = association[['Association_name', 'National_name']]
df7 = df5.merge(df6, left_on='National_name', right_on='National_name')
df8 = df7[['Role', 'Association_name', 'National_name']]

# Query 4
df9 = coach[(coach['Role'] <= 7) | (coach['Role'] <= 4)]
df10 = df9[['Role', 'National_name']]

# Execute queries an

data = [
    {
        'query': "df0 = coach[coach['Role'] == 8]\ndf1 = df0[['Role', 'National_name']]",
        'execution_time': 0.000234,
        'cardinality': 1,
        'num_tables': 1,
        'complexity': "simple"
    },
    {
        'query': "df2 = association[['Association_name', 'National_name']]\ndf3 = df1.merge(df2, left_on='National_name', right_on='National_name')",
        'execution_time': 0.000543,
        'cardinality': 1,
        'num_tables': 2,
        'complexity': "medium"
    }
]

# Create DataFrame and write to CSV
df = pd.DataFrame(data)
df.to_csv('queries_pandas.csv', index=False)

# Write to JSON file
with open('queries.json', mode='w', encoding='utf-8') as file:
    json.dump(data, file, indent=4)
