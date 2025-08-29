from dotenv import load_dotenv
import requests
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from decimal import Decimal
import os

load_dotenv(override=True)

from init_db_complex import DB_CONFIG

# =========== Local llama config ==========
LLM_API_URL = os.getenv("ANTHROPIC_API_KEY")

headers = {
    "Content-Type": "application/json"
}

payload_data = {
    "model": "llama3:8b",
    "messages": [
        {
            "role": "user",
            "content": ""
        }
    ],
    "stream": False
}

@dataclass
class Column:
    name: str
    data_type: str
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: Optional[Tuple[str, str]] = None  # (table, column)
    description: Optional[str] = None

@dataclass
class Table:
    name: str
    columns: List[Column]
    description: Optional[str] = None
    
    def get_column(self, column_name: str) -> Optional[Column]:
        return next((col for col in self.columns if col.name.lower() == column_name.lower()), None)

@dataclass
class Schema:
    name: str
    tables: List[Table]
    description: Optional[str] = None
    
    def get_table(self, table_name: str) -> Optional[Table]:
        return next((table for table in self.tables if table.name.lower() == table_name.lower()), None)

class ComplexSQLAgent:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.schemas: List[Schema] = []
        self.relationships: List[Dict[str, Any]] = []
        self.query_history: List[Dict[str, Any]] = []
        
    def discover_schema(self) -> None:
        """Automatically discover database schema including tables, columns, and relationships."""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Discover tables
        cur.execute("""
            SELECT table_schema, table_name, table_type
            FROM information_schema.tables 
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
        """)
        
        schema_tables = {}
        for schema_name, table_name, table_type in cur.fetchall():
            if schema_name not in schema_tables:
                schema_tables[schema_name] = []
            schema_tables[schema_name].append(table_name)
        
        # For each schema, discover columns and relationships
        for schema_name, table_names in schema_tables.items():
            tables = []
            
            for table_name in table_names:
                columns = self._discover_table_columns(cur, schema_name, table_name)
                tables.append(Table(name=table_name, columns=columns))
            
            self.schemas.append(Schema(name=schema_name, tables=tables))
        
        # Discover relationships
        self.relationships = self._discover_relationships(cur)
        
        cur.close()
        conn.close()
        
    def _discover_table_columns(self, cursor, schema_name: str, table_name: str) -> List[Column]:
        """Discover columns for a specific table."""
        cursor.execute("""
            SELECT 
                c.column_name,
                c.data_type,
                c.is_nullable::boolean,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key,
                CASE WHEN fk.column_name IS NOT NULL THEN true ELSE false END as is_foreign_key,
                fk.foreign_table_schema,
                fk.foreign_table_name,
                fk.foreign_column_name
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT ku.table_schema, ku.table_name, ku.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku 
                ON tc.constraint_name = ku.constraint_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
            ) pk ON c.table_schema = pk.table_schema 
                AND c.table_name = pk.table_name 
                AND c.column_name = pk.column_name
            LEFT JOIN (
                SELECT 
                    ku.table_schema, ku.table_name, ku.column_name,
                    ccu.table_schema as foreign_table_schema,
                    ccu.table_name as foreign_table_name,
                    ccu.column_name as foreign_column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku 
                ON tc.constraint_name = ku.constraint_name
                JOIN information_schema.constraint_column_usage ccu 
                ON tc.constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
            ) fk ON c.table_schema = fk.table_schema 
                AND c.table_name = fk.table_name 
                AND c.column_name = fk.column_name
            WHERE c.table_schema = %s AND c.table_name = %s
            ORDER BY c.ordinal_position
        """, (schema_name, table_name))
        
        columns = []
        for row in cursor.fetchall():
            col_name, data_type, is_nullable, is_pk, is_fk, fk_schema, fk_table, fk_column = row
            
            references = None
            if is_fk and fk_table and fk_column:
                references = (fk_table, fk_column)
            
            columns.append(Column(
                name=col_name,
                data_type=data_type,
                is_nullable=is_nullable,
                is_primary_key=is_pk,
                is_foreign_key=is_fk,
                references=references
            ))
        
        return columns
    
    def _discover_relationships(self, cursor) -> List[Dict[str, Any]]:
        """Discover foreign key relationships between tables."""
        cursor.execute("""
            SELECT 
                tc.table_schema,
                tc.table_name,
                ku.column_name,
                ccu.table_schema AS foreign_table_schema,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name,
                tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage ku 
            ON tc.constraint_name = ku.constraint_name
            JOIN information_schema.constraint_column_usage ccu 
            ON tc.constraint_name = ccu.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
            ORDER BY tc.table_schema, tc.table_name, ku.column_name
        """)
        
        relationships = []
        for row in cursor.fetchall():
            schema, table, column, fk_schema, fk_table, fk_column, constraint_name = row
            relationships.append({
                'from_table': table,
                'from_column': column,
                'to_table': fk_table,
                'to_column': fk_column,
                'constraint_name': constraint_name
            })
        
        return relationships
    
    def generate_schema_context(self) -> str:
        """Generate comprehensive schema context for the LLM."""
        context = "DATABASE SCHEMA INFORMATION:\n\n"
        
        for schema in self.schemas:
            context += f"Schema: {schema.name}\n"
            context += "=" * 50 + "\n\n"
            
            for table in schema.tables:
                context += f"Table: {table.name}\n"
                context += "-" * 30 + "\n"
                
                for col in table.columns:
                    pk_marker = " (PRIMARY KEY)" if col.is_primary_key else ""
                    fk_marker = f" (FK -> {col.references[0]}.{col.references[1]})" if col.is_foreign_key and col.references else ""
                    nullable_marker = "" if col.is_nullable else " NOT NULL"
                    
                    context += f"  - {col.name} ({col.data_type}){nullable_marker}{pk_marker}{fk_marker}\n"
                
                context += "\n"
        
        # Add relationship information
        if self.relationships:
            context += "RELATIONSHIPS:\n"
            context += "-" * 30 + "\n"
            for rel in self.relationships:
                context += f"  {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}\n"
            context += "\n"
        
        return context
    
    def create_complex_prompt(self, user_query: str) -> str:
        """Create sophisticated prompt for complex query generation."""
        schema_context = self.generate_schema_context()
        
        return f"""
{schema_context}

You are an expert PostgreSQL query generator. Generate SQL queries that:

QUERY CAPABILITIES:
- Use proper JOINs (INNER, LEFT, RIGHT, FULL OUTER) based on data requirements
- Implement Common Table Expressions (CTEs) for complex logic
- Use window functions for analytics (ROW_NUMBER, RANK, LAG, LEAD, etc.)
- Create subqueries and correlated subqueries when appropriate
- Use aggregate functions with proper GROUP BY and HAVING clauses
- Implement conditional logic with CASE statements
- Use set operations (UNION, INTERSECT, EXCEPT) when needed

ADVANCED FEATURES:
- Recursive CTEs for hierarchical data
- JSON operations if applicable
- Array operations and functions
- Date/time functions and intervals
- String manipulation functions
- Mathematical and statistical functions

OPTIMIZATION GUIDELINES:
- Use appropriate indexes (consider suggesting index creation)
- Minimize data transfer with selective columns
- Use EXISTS instead of IN for subqueries when appropriate
- Consider query execution order and join strategies
- Use LIMIT for large result sets when appropriate

QUERY ANALYSIS:
- Explain the query logic briefly
- Identify potential performance considerations
- Suggest alternative approaches if applicable

User Question: {user_query}

Respond with:
1. The SQL query in ```sql``` blocks
2. Brief explanation of the query logic
3. Performance considerations (if any)
4. Alternative approaches (if applicable)
"""

    def validate_complex_sql(self, sql: str) -> Dict[str, Any]:
        """Enhanced validation for complex SQL queries."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'complexity_score': 0,
            'features_used': []
        }
        
        sql_upper = sql.upper()
        
        # Check for valid table references
        table_names = [table.name.upper() for schema in self.schemas for table in schema.tables]
        
        # Basic table validation
        has_valid_table = any(table in sql_upper for table in table_names)
        if not has_valid_table:
            validation_result['errors'].append("No valid table references found")
            validation_result['is_valid'] = False
        
        # Detect SQL features and calculate complexity
        features = {
            'CTE': r'\bWITH\b',
            'Window Functions': r'\b(ROW_NUMBER|RANK|DENSE_RANK|LAG|LEAD|FIRST_VALUE|LAST_VALUE)\s*\(',
            'Subqueries': r'\(\s*SELECT\b',
            'Multiple JOINs': r'\bJOIN\b',
            'Aggregation': r'\b(COUNT|SUM|AVG|MIN|MAX|HAVING)\b',
            'Set Operations': r'\b(UNION|INTERSECT|EXCEPT)\b',
            'Conditional Logic': r'\bCASE\b'
        }
        
        for feature, pattern in features.items():
            if re.search(pattern, sql_upper):
                validation_result['features_used'].append(feature)
                validation_result['complexity_score'] += 1
        
        # Count JOINs for complexity
        join_count = len(re.findall(r'\bJOIN\b', sql_upper))
        validation_result['complexity_score'] += join_count * 0.5
        
        return validation_result
    
    def run_llm(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Enhanced LLM interaction with better parsing."""
        payload_data["messages"][0]["content"] = prompt

        try:
            resp = requests.post(LLM_API_URL, headers=headers, json=payload_data)
            resp.raise_for_status()
            
            response_data = resp.json()
            text = response_data["message"]["content"]


            if text and isinstance(text, list) and 'text' in text[0]:
                text = text[0]['text']
            
            # Extract SQL from response
            sql_match = re.search(r"```sql\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
            if not sql_match:
                # Fallback: look for any SQL-like statement
                sql_match = re.search(r"(SELECT|WITH)[\s\S]+?(?=\n\n|\n[A-Z]|$)", text, re.I)
            
            if sql_match:
                sql = sql_match.group(1).strip()
                
                # Validate the extracted SQL
                validation = self.validate_complex_sql(sql)
                
                if validation['is_valid']:
                    # Extract additional info from response
                    explanation = self._extract_explanation(text)
                    
                    return sql, {
                        'explanation': explanation,
                        'validation': validation,
                        'raw_response': text
                    }
                else:
                    raise ValueError(f"Generated SQL failed validation: {validation['errors']}")
            
            raise ValueError("No valid SQL found in model output")
            
        except requests.RequestException as e:
            raise ValueError(f"API request failed: {str(e)}")
    
    def _extract_explanation(self, text: str) -> str:
        """Extract explanation from LLM response."""
        lines = text.split('\n')
        explanation_lines = []
        
        capture = False
        for line in lines:
            if 'explanation' in line.lower() or 'logic' in line.lower():
                capture = True
                continue
            if capture and line.strip():
                if line.startswith('```') or line.startswith('Performance') or line.startswith('Alternative'):
                    break
                explanation_lines.append(line.strip())
        
        return ' '.join(explanation_lines) if explanation_lines else "No explanation provided"
    
    def execute_query(self, query: str) -> Tuple[List[str], List[Tuple], Dict[str, Any]]:
        """Execute query with enhanced error handling and performance monitoring."""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        execution_info = {
            'execution_time': 0,
            'row_count': 0,
            'error': None
        }
        
        try:
            import time
            start_time = time.time()
            
            cur.execute(query)
            
            execution_info['execution_time'] = time.time() - start_time
            
            try:
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
                execution_info['row_count'] = len(rows)
            except:
                rows, cols = [], []
                execution_info['row_count'] = cur.rowcount
                
        except Exception as e:
            execution_info['error'] = str(e)
            rows, cols = [], []
            
        finally:
            conn.commit()
            cur.close()
            conn.close()
        
        return cols, rows, execution_info

def main():
    agent = ComplexSQLAgent(DB_CONFIG)
    
    print("🔄 Discovering database schema...")
    agent.discover_schema()
    
    print(f"✅ Discovered {sum(len(schema.tables) for schema in agent.schemas)} tables across {len(agent.schemas)} schemas")
    
    while True:
        print("\n" + "="*60)
        nl_query = input("🤖 Ask your complex SQL question (or 'quit' to exit): ").strip()
        
        if nl_query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not nl_query:
            continue
        
        try:
            # Generate complex prompt
            prompt = agent.create_complex_prompt(nl_query)
            
            # Get SQL from LLM
            print("\n🔄 Generating complex SQL query...")
            sql_query, query_info = agent.run_llm(prompt)
            
            print(f"\n🔹 Generated SQL (Complexity: {query_info['validation']['complexity_score']:.1f}):")
            print("-" * 50)
            print(sql_query)
            
            if query_info['validation']['features_used']:
                print(f"\n🚀 Advanced Features: {', '.join(query_info['validation']['features_used'])}")
            
            if query_info['explanation']:
                print(f"\n💡 Explanation: {query_info['explanation']}")
            
            # Execute query
            print("\n🔄 Executing query...", sql_query)

            cols, rows, execution_info = agent.execute_query(sql_query)
            
            if execution_info['error']:
                print(f"❌ Execution Error: {execution_info['error']}")
                continue
            
            print(f"\n✅ Query executed successfully!")
            print(f"   Rows returned: {len(rows)}")
            print(f"   Execution time: {execution_info['execution_time']:.3f}s")
            
            if rows:
                # Display first few rows
                print(f"\n🔹 Results Preview:")
                print(cols)
                for i, row in enumerate(rows[:5]):  # Show first 5 rows
                    print(f"  {row}")
                if len(rows) > 5:
                    print(f"  ... and {len(rows) - 5} more rows")
                
                # Create visualization
                # print("\n📊 Creating visualization...")
                # agent.create_advanced_visualization(cols, rows, query_info, execution_info)
            else:
                print("\n📝 Query executed but returned no results")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()