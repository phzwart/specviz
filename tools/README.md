# Tools Package

This package contains utility functions and tools used throughout the SpecViz project.

## Contents

### dbtools.py
Database utility functions for working with DuckDB and SQLite databases:

- `check_table_exists()`: Check if a table exists in the database
- `read_df_from_db()`: Read a table into a pandas DataFrame
- `store_df_in_db()`: Store a pandas DataFrame in the database
- `append_df_to_table()`: Append data to an existing table
- `add_column_to_table()`: Add a new column to an existing table
- `store_dict_in_db()`: Store a dictionary in a key-value table
- `fetch_dict_from_db()`: Retrieve a dictionary from the key-value table
- `create_table_if_not_exists()`: Create the key-value table if it doesn't exist

## Usage

```python
from tools.dbtools import check_table_exists, read_df_from_db, store_df_in_db
import duckdb

# Connect to database
conn = duckdb.connect("my_database.db")

# Check if table exists
if check_table_exists(conn, "my_table"):
    # Read data
    df = read_df_from_db(conn, "my_table")
    
    # Store new data
    store_df_in_db(conn, new_df, "new_table", if_exists="replace")

conn.close()
```

## Dependencies

- `duckdb`: For database operations
- `pandas`: For DataFrame handling
- `sqlite3`: For SQLite support (optional) 