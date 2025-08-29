from ai_sql_agent_v5_complex import ComplexSQLAgent
from init_db_complex import DB_CONFIG

def test_complex_queries():
    """Test suite for complex query generation scenarios."""
    
    agent = ComplexSQLAgent(DB_CONFIG)
    agent.discover_schema()
    
    print("🧪 COMPLEX SQL QUERY TEST SUITE")
    print("="*60)
    
    # Test cases from simple to very complex
    test_queries = [
        {
            "name": "Window Functions - Ranking",
            "query": "Show me the top 3 highest paid employees in each department with their salary rank"
        },
        {
            "name": "CTE with Aggregation", 
            "query": "Find departments where the average salary is above the company average, show department name and average salary"
        },
        {
            "name": "Complex Join with Subquery",
            "query": "List employees whose salary is higher than the average salary of their department, include department name"
        },
        {
            "name": "Time-based Analysis",
            "query": "Show salary trends by year - total salary cost per department for each year employees were hired"
        },
        {
            "name": "Advanced Analytics",
            "query": "Calculate running total of salary costs by department, ordered by hire date, with percentage of total company cost"
        },
        {
            "name": "Hierarchical Query Simulation",
            "query": "Find all employees hired in the same year as the highest paid employee, group by department"
        },
        {
            "name": "Complex Conditional Logic",
            "query": "Categorize employees as 'Senior' (>3 years), 'Mid-level' (1-3 years), 'Junior' (<1 year) based on hire date, show salary statistics for each category by department"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{'='*20} TEST {i}: {test_case['name']} {'='*20}")
        print(f"Query: {test_case['query']}")
        print("-" * 80)
        
        try:
            # Generate query
            prompt = agent.create_complex_prompt(test_case['query'])
            sql_query, query_info = agent.run_llm(prompt)
            
            print("🔹 Generated SQL:")
            print(sql_query)
            
            # Show complexity analysis
            validation = query_info['validation']
            print(f"\n📊 Complexity Score: {validation['complexity_score']:.1f}")
            print(f"🚀 Features Used: {', '.join(validation['features_used']) if validation['features_used'] else 'Basic SQL'}")
            
            if query_info.get('explanation'):
                print(f"💡 Logic: {query_info['explanation']}")
            
            # Execute (with error handling)
            try:
                cols, rows, exec_info = agent.execute_query(sql_query)
                if exec_info['error']:
                    print(f"❌ Execution Error: {exec_info['error']}")
                    results.append({'test': test_case['name'], 'status': 'failed', 'error': exec_info['error']})
                else:
                    print(f"✅ Executed successfully - {len(rows)} rows returned in {exec_info['execution_time']:.3f}s")
                    if rows and len(rows) <= 10:  # Show results for small result sets
                        print("📋 Results:")
                        print(f"   Columns: {cols}")
                        for row in rows[:5]:
                            print(f"   {row}")
                        if len(rows) > 5:
                            print(f"   ... and {len(rows) - 5} more rows")
                    results.append({'test': test_case['name'], 'status': 'success', 'rows': len(rows), 'complexity': validation['complexity_score']})
            except Exception as e:
                print(f"❌ Execution failed: {str(e)}")
                results.append({'test': test_case['name'], 'status': 'failed', 'error': str(e)})
                
        except Exception as e:
            print(f"❌ Query generation failed: {str(e)}")
            results.append({'test': test_case['name'], 'status': 'generation_failed', 'error': str(e)})
        
        print("\n" + "="*80)
    
    # Summary report
    print("\n🎯 TEST SUMMARY REPORT")
    print("="*60)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        avg_complexity = sum(r.get('complexity', 0) for r in successful) / len(successful)
        print(f"📊 Average Complexity Score: {avg_complexity:.2f}")
        
        print(f"\n🏆 Most Complex Query: {max(successful, key=lambda x: x.get('complexity', 0))['test']}")
    
    if failed:
        print(f"\n❌ Failed Tests:")
        for fail in failed:
            print(f"   - {fail['test']}: {fail.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_complex_queries()