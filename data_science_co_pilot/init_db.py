import psycopg2
import os

DB_CONFIG = {
    "dbname": "ai_sql_agents",
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

schema_sql = """
DROP TABLE IF EXISTS salaries;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS departments;
"""

data_sql = """
INSERT INTO departments (dept_name) VALUES
('Engineering'),
('HR'),
('Sales'),
('Finance');

INSERT INTO employees (first_name, last_name, hire_date, dept_id) VALUES
('Alice', 'Johnson', '2020-01-15', 1),
('Bob', 'Smith', '2019-03-22', 1),
('Charlie', 'Brown', '2021-07-10', 2),
('Diana', 'Prince', '2018-11-05', 3),
('Ethan', 'Hunt', '2022-02-14', 4);

INSERT INTO salaries (emp_id, amount, effective_date) VALUES
(1, 90000, '2023-01-01'),
(2, 85000, '2023-01-01'),
(3, 60000, '2023-01-01'),
(4, 75000, '2023-01-01'),
(5, 95000, '2023-01-01');
"""

def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(schema_sql)
    cur.execute(data_sql)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database initialized with tables and sample data.")

if __name__ == "__main__":
    init_db()
