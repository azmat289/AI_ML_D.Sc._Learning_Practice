from dotenv import load_dotenv
import requests
import psycopg2
import matplotlib.pyplot as plt
import re
import os

load_dotenv(override=True)

from init_db import DB_CONFIG

# =========== Claude config ==========
LLM_API_URL = os.environ("ANTHROPIC_LLM_API_URL")
API_KEY = os.environ("ANTHROPIC_API_KEY")
headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY,
    "anthropic-version": "2023-06-01"
}

payload_data = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 512,
    "messages": [
        {
            "role": "user",
            "content": ""
        }
    ]
}

# Database schema context
SCHEMA_CONTEXT = """
Available tables and their structure:

departments:
- dept_id (SERIAL PRIMARY KEY)
- dept_name (VARCHAR(50) NOT NULL)

employees:
- emp_id (SERIAL PRIMARY KEY) 
- first_name (VARCHAR(50))
- last_name (VARCHAR(50))
- hire_date (DATE)
- dept_id (INT REFERENCES departments(dept_id))

salaries:
- salary_id (SERIAL PRIMARY KEY)
- emp_id (INT REFERENCES employees(emp_id))
- amount (NUMERIC(10,2))
- effective_date (DATE)

Relationships:
- employees.dept_id → departments.dept_id
- salaries.emp_id → employees.emp_id
"""
# ============================

def prompt(userQuery: str) -> str:
    return f"""
        You are an assistant that converts natural language into SQL for PostgreSQL.

        IMPORTANT: Use ONLY the exact table and column names provided in the schema above.
        - Always use proper JOINs when querying across tables
        - Use the correct column names (e.g., 'dept_name' not 'department_name')
        - Use the correct table names (e.g., 'departments' not 'department')

        User question: "{userQuery}"

        Return ONLY a valid SQL query that follows the schema. Format your response as:
        ```sql
        YOUR_SQL_QUERY_HERE
        ```
        """

def validate_sql_schema(sql: str) -> bool:
    """Basic validation that SQL uses correct table and column names."""
    sql_upper = sql.upper()
    
    # Valid table names
    valid_tables = ['DEPARTMENTS', 'EMPLOYEES', 'SALARIES']
    
    # Valid column mappings, Here D, E and S is alias of tables to identify columns with dot
    valid_columns = {
        'D': ['DEPT_ID', 'DEPT_NAME'],
        'E': ['EMP_ID', 'FIRST_NAME', 'LAST_NAME', 'HIRE_DATE', 'DEPT_ID'],
        'S': ['SALARY_ID', 'EMP_ID', 'AMOUNT', 'EFFECTIVE_DATE']
    }
    
    # Check if query references at least one valid table
    has_valid_table = any(table in sql_upper for table in valid_tables)
    if not has_valid_table:
        return False
    
    # Check for common column naming issues
    # Look for potential column references and validate them
    
    # Extract table.column patterns
    table_column_pattern = r'(\w+)\.(\w+)'
    matches = re.findall(table_column_pattern, sql_upper)
    
    for table, column in matches:
        if table in valid_columns:
            # print(column, table)
            # print(valid_columns[table])
            if column not in valid_columns[table]:
                return False
    
    return True


def run_llm(prompt: str, include_schema: bool = True) -> str:
    """Send prompt to llama.cpp server and extract SQL."""
    # Add schema context to prompt if requested
    if include_schema:
        schema_prompt = f"{SCHEMA_CONTEXT}\n\n{prompt}"
    else:
        schema_prompt = prompt

    payload_data["messages"][0]["content"] = schema_prompt;
    resp = requests.post(LLM_API_URL, headers=headers, json=payload_data)
    text = resp.json().get("content", "")

    if text and isinstance(text, list) and 'text' in text[0]:
        text = text[0]['text']

    text = text.strip()

    # Remove markdown fences (```sql ... ```)
    text = re.sub(r"```(?:sql|json)?", "", text, flags=re.IGNORECASE).strip()

    # Search for first SQL-looking statement
    match = re.search(r"(SELECT|WITH)[\s\S]+?;", text, re.I)

    if match:
        sql = match.group(0).strip()
        # Basic validation for schema-aware queries
        if validate_sql_schema(sql):
            return sql
        else:
            raise ValueError(f"Generated SQL doesn't match expected schema: {sql}")

    raise ValueError("No valid SQL found in model output")


def run_sql(query: str):
    """Execute SQL on PostgreSQL and return results."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(query)
    try:
        rows = cur.fetchall()
        print('Result :')
        print(rows)
        cols = [desc[0] for desc in cur.description]
    except:
        rows, cols = [], []
    conn.commit()
    cur.close()
    conn.close()
    return cols, rows


def visualize(cols, rows):
    """Quick bar chart if numeric results available."""
    if len(cols) >= 2 and len(rows) > 0:
        if 'first_name' in cols:
            xPlot_index = cols.index('first_name')
        elif 'dept_name' in cols:
            xPlot_index = cols.index('dept_name')

        if 'amount' in cols:
            yPlot_index = cols.index('amount')
        elif 'hire_date' in cols:
            yPlot_index = cols.index('hire_date')
        elif 'average_salary':
            yPlot_index = cols.index('average_salary')
        else:
            yPlot_index = 3

        x = [r[xPlot_index] for r in rows]
        y = [r[yPlot_index] for r in rows]
        plt.bar(x, y)
        plt.xlabel(cols[xPlot_index])
        plt.ylabel(cols[yPlot_index])
        plt.title("Query Result")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough data to plot.")


if __name__ == "__main__":
    nl_query = input("\nAsk your question: ")

    # Step 1: Convert NL → SQL with schema context
    prompt = prompt(nl_query)
    sql_query = run_llm(prompt)
    print("\n🔹 Generated SQL:\n", sql_query)

    # Step 2: Run SQL
    try:
        cols, rows = run_sql(sql_query)
        print("\n🔹 Query Results:")
        print(cols)
        for r in rows:
            print(r)

        # Step 3: Visualize
        visualize(cols, rows)

    except Exception as e:
        print("❌ Error running SQL:", e)



# Test with the following inputs one by one

# Show top 3 employees by salary
# List employees in Engineering department
# What is the average salary by department?