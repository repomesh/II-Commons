import csv
import ast # For safely evaluating string representation of list
import os
import time
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row # To get results as dictionaries

db_pool = None

def _configure_db_connection(conn):
    """Helper function to configure each new connection in the pool."""
    register_vector(conn)

def init_db():
    """Initializes a database connection pool using settings from a .env file."""
    global db_pool
    if db_pool is not None:
        return db_pool

    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    
    if os.path.exists(dotenv_path):
        print(f"Loading .env file from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Relying on environment variables.", file=sys.stderr)
        # Still call load_dotenv without path to load from standard locations or existing env vars
        load_dotenv()


    POSTGRES_HOST = os.getenv('POSTGRES_HOST')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432') # Default port if not specified
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
    POSTGRES_DB = os.getenv('POSTGRES_DB')

    missing_vars = []
    if not POSTGRES_HOST: missing_vars.append("POSTGRES_HOST")
    if not POSTGRES_USER: missing_vars.append("POSTGRES_USER")
    if not POSTGRES_PASSWORD: missing_vars.append("POSTGRES_PASSWORD")
    if not POSTGRES_DB: missing_vars.append("POSTGRES_DB")

    if missing_vars:
        print(f"Error: Missing required database configuration: {', '.join(missing_vars)}. "
              "Please ensure they are set in .env file or environment variables.", file=sys.stderr)
        sys.exit(1)

    conn_info = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
    
    try:
        print(f"Attempting to connect to PostgreSQL at {POSTGRES_HOST}:{POSTGRES_PORT}, DB: {POSTGRES_DB}")
        db_pool = ConnectionPool(
            conninfo=conn_info,
            open=True, 
            configure=_configure_db_connection, 
            min_size=1, # Suitable for a benchmark script
            max_size=5  # Adjust as needed
        )
        # Test connection
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                if cur.fetchone()[0] == 1:
                    print("Database connection pool initialized and test query successful.")
                else:
                    print("Database connection pool initialized, but test query failed.", file=sys.stderr)
                    sys.exit(1)
    except Exception as e:
        print(f"Critical: Failed to initialize database pool or test connection: {e}", file=sys.stderr)
        sys.exit(1)
    return db_pool

def query_db(sql: str, values: tuple = None):
    """Executes a SQL query using the initialized connection pool and returns results as dicts."""
    global db_pool
    if db_pool is None:
        print("Error: Database pool is not initialized. Call init_db() first.", file=sys.stderr)
        sys.exit(1) # Critical error, cannot proceed
    
    results = []
    try:
        with db_pool.connection() as conn:
            # Using dict_row to get results as dictionaries
            with conn.cursor(row_factory=dict_row) as cur:
                # print(f"Executing SQL: {cur.mogrify(sql, values)}") # For debugging SQL
                cur.execute(sql, values)
                results = cur.fetchall()
    except Exception as e:
        print(f"Database query error: {e}", file=sys.stderr)
        # Optionally, re-raise or handle more gracefully depending on script's needs
        # For a benchmark, printing error and returning empty might be okay, or exiting.
    return results

def tempalte_vector_search_ts_wikipedia_en_embed(table_name: str, embedding_vector: list, top_k: int = 5) -> tuple:

    sql = f"""
    SELECT id, title, url, chunk_index, chunk_text,
           (vector <=> %s::vector) as distance,
           ((2 - (vector <=> %s::vector)) / 2) as similarity
    FROM {table_name}
    ORDER BY (vector <=> %s::vector) ASC
    OFFSET %s
    LIMIT %s;
    """
    # Parameters: embedding_vector (for distance, similarity, ordering), offset (0), limit (top_k)
    values = (embedding_vector, embedding_vector, embedding_vector, 0, top_k)
    return sql, values

def read_and_print_vector_info(input_file="vector.csv"):
    """
    Reads a CSV file containing text and vector strings.
    For each valid vector, performs a database search and prints results.
    Also records and reports query execution times.
    """
    query_execution_times = [] # List to store execution times of each query
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            csv_reader = csv.reader(infile)
            print("Processing vectors and performing database search...\n")

            for i, row in enumerate(csv_reader):
                if len(row) == 2:
                    text_content = row[0]
                    vector_str = row[1]
                    
                    print(f"--- Query Text: {text_content} ---")

                    vector_list = None
                    try:
                        vector_list = ast.literal_eval(vector_str)
                        if not isinstance(vector_list, list):
                            print(f"  Parsed vector is not a list for text: \"{text_content[:50]}...\"")
                            vector_list = None 
                        else:
                            print(f"  Vector Length: {len(vector_list)}")
                    except (ValueError, SyntaxError) as e:
                        print(f"  Error parsing vector string on line {i+1} for text \"{text_content[:50]}...\": {vector_str}. Error: {e}")
                    
                    if vector_list:
                        table_name_to_search = "ts_wikipedia_en_embed"
                        try:
                            sql_query, query_params = tempalte_vector_search_ts_wikipedia_en_embed(
                                table_name_to_search, vector_list, top_k=30
                            )
                            print(f"  Searching in '{table_name_to_search}'...")
                            
                            query_start_time = time.perf_counter()
                            db_results = query_db(sql_query, query_params)
                            query_end_time = time.perf_counter()
                            
                            execution_time = query_end_time - query_start_time
                            query_execution_times.append(execution_time)
                            print(f"  Query executed in: {execution_time:.4f} seconds")

                            if db_results and isinstance(db_results, list):
                                if not db_results: 
                                     print("  No results found in database.")
                                else:
                                     # print(db_results[0]) # Original print, can be kept or removed
                                     print(f"  Found {len(db_results)} results. First result example: {str(db_results[0])[:200]}...")
                            elif db_results: 
                                print(f"  Query executed, but results are not in the expected list format. Result: {db_results}")
                            else: 
                                print("  No results returned from query or an issue with psql.query.")
                        except Exception as db_e:
                            print(f"  Database search error for text \"{text_content[:50]}...\": {db_e}")
                    else:
                        print(f"  Skipping database search due to vector parsing/validation issue for text: \"{text_content[:50]}...\"")
                    print("-" * 50 + "\n")
                else:
                    print(f"Skipping malformed CSV row {i+1}: {row}\n")
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if query_execution_times:
            total_query_time = sum(query_execution_times)
            average_query_time = total_query_time / len(query_execution_times)
            num_queries = len(query_execution_times)
            print("\n--- Query Performance Summary ---")
            print(f"Total queries executed: {num_queries}")
            print(f"Total time spent on queries: {total_query_time:.4f} seconds")
            print(f"Average time per query: {average_query_time:.4f} seconds")
        else:
            print("\n--- Query Performance Summary ---")
            print("No queries were executed or timed.")

if __name__ == "__main__":
    init_db() # Initialize the database pool using .env settings

    # Default input file, can be changed if needed via command line argument
    csv_input_file = "vector.csv" 
    if len(sys.argv) > 1:
        csv_input_file = sys.argv[1]
        print(f"Using input file from argument: {csv_input_file}")
    else:
        print(f"Using default input file: {csv_input_file}")

    read_and_print_vector_info(input_file=csv_input_file)
    print("\nProcessing complete.")
