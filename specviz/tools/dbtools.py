from typing import Any, Dict, List, Union

import sqlite3

import duckdb
import numpy as np
import pandas as pd


def create_table_if_not_exists(
    conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection],
) -> None:
    """
    Creates a KeyValue table with a 'dict_name' column for both SQLite and DuckDB.

    Args:
        conn: Either a SQLite or DuckDB connection
    """
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)

    if is_duckdb:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS KeyValue (
                dict_name VARCHAR,
                key VARCHAR,
                value VARCHAR,
                PRIMARY KEY (dict_name, key)
            )
        """
        )
    else:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS KeyValue (
                    dict_name TEXT,
                    key TEXT,
                    value TEXT,
                    PRIMARY KEY (dict_name, key)
                )
            """
            )


def store_dict_in_db(
    conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection],
    dict_name: str,
    data_dict: dict[str, str],
) -> None:
    """
    Stores the given data_dict under the name 'dict_name'.

    Args:
        conn: Either a SQLite or DuckDB connection
        dict_name: Name of the dictionary to store
        data_dict: Dictionary to store
    """
    create_table_if_not_exists(conn)
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)

    # Convert the dictionary into a list of tuples
    dict_items = [(dict_name, str(k), str(v)) for k, v in data_dict.items()]

    if is_duckdb:
        # DuckDB uses different syntax for upsert
        for item in dict_items:
            conn.execute(
                """
                INSERT OR REPLACE INTO KeyValue (dict_name, key, value) 
                VALUES (?, ?, ?)
            """,
                item,
            )
    else:
        # SQLite version
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO KeyValue (dict_name, key, value) VALUES (?, ?, ?)",
                dict_items,
            )


def fetch_dict_from_db(
    conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection], dict_name: str
) -> dict[str, str]:
    """
    Fetches all key-value pairs for the specified 'dict_name'.

    Args:
        conn: Either a SQLite or DuckDB connection
        dict_name: Name of the dictionary to fetch

    Returns:
        Dict[str, str]: The fetched dictionary
    """
    create_table_if_not_exists(conn)
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)

    if is_duckdb:
        # DuckDB version
        result = conn.execute(
            """
            SELECT key, value
            FROM KeyValue
            WHERE dict_name = ?
        """,
            [dict_name],
        ).fetchall()
        return {k: v for (k, v) in result}
    else:
        # SQLite version
        cursor = conn.execute(
            """
            SELECT key, value
            FROM KeyValue
            WHERE dict_name = ?
        """,
            (dict_name,),
        )
        rows = cursor.fetchall()
        return {k: v for (k, v) in rows}


def store_df_in_db(
    conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection],
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "replace",
    index: bool = False,
) -> None:
    """
    Store a pandas DataFrame in either SQLite or DuckDB.

    Args:
        conn: Either a SQLite or DuckDB connection
        df: Pandas DataFrame to store
        table_name: Name of the table to create/update
        if_exists: How to behave if table exists ('fail', 'replace', or 'append')
        index: Whether to store the DataFrame index as a column
    """
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)

    if is_duckdb:
        # DuckDB version
        if if_exists == "replace":
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.register("temp_df", df)
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")
        conn.unregister("temp_df")
    else:
        # SQLite version
        df.to_sql(name=table_name, con=conn, if_exists=if_exists, index=index)


def read_df_from_db(
    conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection], table_name: str
) -> pd.DataFrame:
    """
    Read a table from either SQLite or DuckDB into a pandas DataFrame.

    Args:
        conn: Either a SQLite or DuckDB connection
        table_name: Name of the table to read

    Returns:
        pd.DataFrame: The table contents as a DataFrame
    """
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)

    if is_duckdb:
        # DuckDB version
        return conn.execute(f"SELECT * FROM {table_name}").df()
    else:
        # SQLite version
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)


def check_table_exists(
    conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection], table_name: str
) -> bool:
    """
    Check if a table exists in the database.

    Args:
        conn: Either a SQLite or DuckDB connection
        table_name: Name of the table to check

    Returns:
        bool: True if table exists, False otherwise
    """
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)

    if is_duckdb:
        try:
            # DuckDB's way of checking table existence
            result = conn.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name=?
            """,
                [table_name],
            ).fetchone()
        except Exception:
            # Fallback for older DuckDB versions
            result = (
                conn.execute(f"SELECT * FROM {table_name} LIMIT 0").fetchone()
                is not None
            )
    else:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (table_name,),
        )
        result = cursor.fetchone()

    return result is not None


def append_df_to_table(
    conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection],
    df: pd.DataFrame,
    table_name: str,
) -> None:
    """
    Append DataFrame to existing table or create new one if it doesn't exist.
    Only appends rows, ensuring columns match between existing and new data.

    Args:
        conn: Either a SQLite or DuckDB connection
        df: DataFrame to store
        table_name: Name of the table
    """
    if check_table_exists(conn, table_name):
        # Read existing data
        df_existing = read_df_from_db(conn, table_name)
        print("\nExisting table structure:")
        print(df_existing.info())
        print("\nExisting table content:")
        print(df_existing)

        print("\nNew data structure:")
        print(df.info())
        print("\nNew data content:")
        print(df)

        # Ensure at least one common column exists
        common_columns = df_existing.columns.intersection(df.columns)
        if len(common_columns) == 0:
            raise ValueError(f"No matching columns between existing table and new data")

        # Create a new DataFrame with all columns from both DataFrames
        all_columns = df_existing.columns.union(df.columns)

        # Prepare new data with same columns as existing
        df_to_append = pd.DataFrame(columns=all_columns)
        for col in all_columns:
            if col in df.columns:
                df_to_append[col] = df[col]
            else:
                df_to_append[col] = None

        # Combine with new data
        df_combined = pd.concat([df_existing, df_to_append], ignore_index=True)

        print("\nMerged table structure:")
        print(df_combined.info())
        print("\nMerged table content:")
        print(df_combined)

        # Store combined data
        store_df_in_db(conn, df_combined, table_name, if_exists="replace", index=False)
        print(f"\nAppended {len(df)} rows to existing table '{table_name}'")
    else:
        # Create new table
        store_df_in_db(conn, df, table_name, if_exists="fail", index=False)
        print(f"Created new table '{table_name}' with {len(df)} rows")


def add_column_to_table(
    conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection],
    table_name: str,
    column_name: str,
    default_value: Any = None,
    data_type: str = None,
    overwrite: bool = False,
) -> None:
    """
    Add a new column to an existing table.

    Args:
        conn: Either a SQLite or DuckDB connection
        table_name: Name of the table to modify
        column_name: Name of the new column to add
        default_value: Default value for the new column (optional)
        data_type: SQL data type for the new column (e.g., 'INTEGER', 'VARCHAR', 'DOUBLE').
                  If None, will be inferred from default_value
        overwrite: If True, will overwrite existing column if it exists
    """
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)

    # Infer data type if not provided
    if data_type is None:
        if isinstance(default_value, (int, np.integer)):
            data_type = "INTEGER"
        elif isinstance(default_value, (float, np.floating)):
            data_type = "DOUBLE"
        elif isinstance(default_value, (str, np.character)):
            data_type = "VARCHAR"
        elif isinstance(default_value, bool):
            data_type = "BOOLEAN"
        else:
            data_type = "VARCHAR"  # default to VARCHAR for unknown types

    print(f"\nAdding column '{column_name}' to table '{table_name}'")
    print(f"Default value: {default_value}")
    print(f"Using {'DuckDB' if is_duckdb else 'SQLite'}")

    # Print current table state
    current_df = read_df_from_db(conn, table_name)
    print("\nCurrent table structure:")
    print(current_df.info())
    print("\nCurrent table content:")
    print(current_df)

    if not check_table_exists(conn, table_name):
        raise ValueError(f"Table '{table_name}' does not exist")

    if is_duckdb:
        if isinstance(default_value, pd.Series):
            # For Series input, create a temporary table with the new column
            temp_df = current_df.copy()
            temp_df[column_name] = default_value.values
            store_df_in_db(conn, temp_df, table_name, if_exists="replace", index=False)
        else:
            # Add column with specified data type
            if default_value is None:
                conn.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {data_type}"
                )
            else:
                conn.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {data_type}"
                )
                # Convert default_value to the appropriate type before updating
                if data_type == "INTEGER":
                    default_value = int(default_value)
                elif data_type == "DOUBLE":
                    default_value = float(default_value)
                elif data_type == "BOOLEAN":
                    default_value = bool(default_value)
                else:
                    default_value = str(default_value)
                conn.execute(
                    f"UPDATE {table_name} SET {column_name} = ?", [default_value]
                )
    else:
        # SQLite version remains the same as it handles type conversion automatically
        df = read_df_from_db(conn, table_name)
        if isinstance(default_value, pd.Series):
            df[column_name] = default_value.values
        else:
            df[column_name] = default_value
        store_df_in_db(conn, df, table_name, if_exists="replace", index=False)

    # Print final table state
    final_df = read_df_from_db(conn, table_name)
    print("\nFinal table structure:")
    print(final_df.info())
    print("\nFinal table content:")
    print(final_df)

    print(f"\nSuccessfully added column '{column_name}' to table '{table_name}'")


def read_experiment_data(
    db_path: str,
    table_name: str = None,
    query: str = None,
    return_metadata: bool = True,
) -> Union[np.ndarray, Dict[str, Union[np.ndarray, pd.DataFrame]]]:
    """
    Read experiment data from DuckDB database and convert to numpy arrays for analysis.

    Args:
        db_path: Path to the DuckDB database file
        table_name: Name of the table to read (if None, will list available tables)
        query: Custom SQL query to execute (if provided, overrides table_name)
        return_metadata: If True, returns a dict with data arrays and metadata.
                        If False, returns just the numpy array.

    Returns:
        If return_metadata=True: Dict containing:
            - 'data': numpy array of the main data
            - 'columns': column names
            - 'metadata': pandas DataFrame with additional info
            - 'table_info': table structure information
        If return_metadata=False: Just the numpy array
    """
    import duckdb

    # Connect to the database
    conn = duckdb.connect(db_path)

    try:
        # If no table_name or query provided, list available tables
        if table_name is None and query is None:
            tables = conn.execute("SHOW TABLES").fetchall()
            print("Available tables:")
            for table in tables:
                print(f"  - {table[0]}")
            return None

        # Get table information
        if query is None:
            # Get table structure
            table_info = conn.execute(f"DESCRIBE {table_name}").fetchall()
            print(f"Table '{table_name}' structure:")
            for col in table_info:
                print(f"  {col[0]}: {col[1]}")

            # Read the data
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        else:
            # Execute custom query
            df = conn.execute(query).fetchdf()
            print(f"Executed custom query, returned {len(df)} rows")

        # Convert to numpy array
        data_array = df.to_numpy()

        if return_metadata:
            result = {
                "data": data_array,
                "columns": df.columns.tolist(),
                "metadata": df,
                "shape": data_array.shape,
                "dtypes": df.dtypes.to_dict(),
            }

            if query is None:
                result["table_info"] = table_info
                result["table_name"] = table_name

            return result
        else:
            return data_array

    finally:
        conn.close()


def list_experiment_tables(db_path: str) -> List[str]:
    """
    List all tables in the experiment database.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        List of table names
    """
    import duckdb

    conn = duckdb.connect(db_path)
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        return [table[0] for table in tables]
    finally:
        conn.close()


def get_experiment_summary(db_path: str, table_name: str = None) -> Dict[str, Any]:
    """
    Get a summary of experiment data including basic statistics.

    Args:
        db_path: Path to the DuckDB database file
        table_name: Name of the table to summarize (if None, summarizes all tables)

    Returns:
        Dictionary containing summary information
    """
    import duckdb

    conn = duckdb.connect(db_path)
    try:
        if table_name is None:
            # Summarize all tables
            tables = list_experiment_tables(db_path)
            summary = {}
            for table in tables:
                summary[table] = get_experiment_summary(db_path, table)
            return summary
        else:
            # Get table summary
            df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()

            summary = {
                "table_name": table_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "null_counts": df.isnull().sum().to_dict(),
            }

            # Add basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary["numeric_stats"] = df[numeric_cols].describe().to_dict()

            return summary

    finally:
        conn.close()


if __name__ == "__main__":
    conn = duckdb.connect(":memory:")

    # Create a test table
    df = pd.DataFrame({"id": [1, 2, 3]})
    store_df_in_db(conn, df, "test_table")

    # Test different data types
    add_column_to_table(conn, "test_table", "int_col", 42)  # Should infer INTEGER
    add_column_to_table(conn, "test_table", "float_col", 3.14, data_type="DOUBLE")
    add_column_to_table(conn, "test_table", "str_col", "hello")  # Should infer VARCHAR
    add_column_to_table(conn, "test_table", "bool_col", True)  # Should infer BOOLEAN

    # Verify the results
    result = read_df_from_db(conn, "test_table")
    print(result)
    print(result.dtypes)

    conn.close()
