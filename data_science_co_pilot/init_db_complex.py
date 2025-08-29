import load_envs.load_envs as load_envs
import psycopg2
import os

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),   # your DB name
    "user": os.getenv("DB_USER"),          # your DB user
    "password": os.getenv("DB_PASSWORD"),          # change this
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

schema_sql = """
DROP TABLE IF EXISTS employee_projects;
DROP TABLE IF EXISTS projects;
DROP TABLE IF EXISTS salaries;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS departments;
DROP TABLE IF EXISTS locations;

-- Locations
CREATE TABLE locations (
    location_id SERIAL PRIMARY KEY,
    city VARCHAR(50),
    country VARCHAR(50)
);

-- Departments
CREATE TABLE departments (
    dept_id SERIAL PRIMARY KEY,
    dept_name VARCHAR(100) NOT NULL,
    manager_id INT,
    location_id INT REFERENCES locations(location_id)
);

-- Jobs
CREATE TABLE jobs (
    job_id SERIAL PRIMARY KEY,
    job_title VARCHAR(100) NOT NULL,
    min_salary NUMERIC(10,2),
    max_salary NUMERIC(10,2)
);

-- Employees
CREATE TABLE employees (
    emp_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    phone VARCHAR(20),
    dob DATE,
    hire_date DATE,
    dept_id INT REFERENCES departments(dept_id),
    job_id INT REFERENCES jobs(job_id),
    manager_id INT
);

-- Salaries
CREATE TABLE salaries (
    salary_id SERIAL PRIMARY KEY,
    emp_id INT REFERENCES employees(emp_id),
    amount NUMERIC(10,2),
    effective_date DATE
);

-- Projects
CREATE TABLE projects (
    project_id SERIAL PRIMARY KEY,
    project_name VARCHAR(100),
    start_date DATE,
    end_date DATE,
    budget NUMERIC(12,2)
);

-- Employee ↔ Project mapping
CREATE TABLE employee_projects (
    emp_id INT REFERENCES employees(emp_id),
    project_id INT REFERENCES projects(project_id),
    role VARCHAR(50),
    PRIMARY KEY (emp_id, project_id)
);
"""

data_sql = """
-- Locations
INSERT INTO locations (city, country) VALUES
('New York', 'USA'),
('London', 'UK'),
('Berlin', 'Germany'),
('Mumbai', 'India');

-- Departments
INSERT INTO departments (dept_name, manager_id, location_id) VALUES
('Engineering', NULL, 1),
('HR', NULL, 2),
('Sales', NULL, 3),
('Finance', NULL, 4);

-- Jobs
INSERT INTO jobs (job_title, min_salary, max_salary) VALUES
('Software Engineer', 60000, 120000),
('HR Specialist', 40000, 80000),
('Sales Executive', 45000, 90000),
('Finance Analyst', 55000, 110000),
('Project Manager', 70000, 140000);

-- Employees
INSERT INTO employees (first_name, last_name, email, phone, dob, hire_date, dept_id, job_id, manager_id) VALUES
('Alice', 'Johnson', 'alice.j@example.com', '123-456-7890', '1990-04-12', '2020-01-15', 1, 1, NULL),
('Bob', 'Smith', 'bob.s@example.com', '987-654-3210', '1988-09-23', '2019-03-22', 1, 1, 1),
('Charlie', 'Brown', 'charlie.b@example.com', '555-333-2222', '1992-07-10', '2021-07-10', 2, 2, 1),
('Diana', 'Prince', 'diana.p@example.com', '444-222-1111', '1985-11-05', '2018-11-05', 3, 3, 2),
('Ethan', 'Hunt', 'ethan.h@example.com', '222-333-4444', '1995-02-14', '2022-02-14', 4, 4, 2),
('Frank', 'Miller', 'frank.m@example.com', '333-444-5555', '1983-05-30', '2017-06-01', 1, 5, 1);

-- Salaries
INSERT INTO salaries (emp_id, amount, effective_date) VALUES
(1, 90000, '2023-01-01'),
(2, 85000, '2023-01-01'),
(3, 60000, '2023-01-01'),
(4, 75000, '2023-01-01'),
(5, 95000, '2023-01-01'),
(6, 120000, '2023-01-01');

-- Projects
INSERT INTO projects (project_name, start_date, end_date, budget) VALUES
('AI Platform', '2022-01-01', '2023-12-31', 500000),
('HR Revamp', '2021-06-01', '2022-12-31', 200000),
('Sales Expansion', '2022-03-01', '2023-09-30', 300000),
('Finance Automation', '2023-01-01', NULL, 400000);

-- Employee Projects
INSERT INTO employee_projects (emp_id, project_id, role) VALUES
(1, 1, 'Lead Developer'),
(2, 1, 'Developer'),
(3, 2, 'HR Coordinator'),
(4, 3, 'Sales Rep'),
(5, 4, 'Analyst'),
(6, 1, 'Project Manager');
"""

def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(schema_sql)
    cur.execute(data_sql)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database initialized with extended schema and sample data.")

if __name__ == "__main__":
    init_db()
