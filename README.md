# AI_ML_D.Sc._Learning_Practice

This repository is for practicing and experimenting with AI/ML, SQL agents, and model inference.

---

## 📂 Project Structure

```text
AI_ML_D.Sc._Learning_Practice/
│
├── data_science_co_pilot/   # SQL agent scripts and utilities
├── inference_model/         # Placeholder for ML/AI model code
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore rules
└── requirements.txt         # Python dependencies

## ⚙️ Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd AI_ML_D.Sc._Learning_Practice

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Copy env variable and update with values
   ```base
   cp .env.example .env

5. Run SQL agent scripts inside the data_science_co_pilot/ folder
   ```base
   python data_science_co_pilot/init_db.py          # To set up the db
   python data_science_co_pilot/ai_sql_agent_v3.py  # generate sql queries using open source llama and plot visuals based on results
   python data_science_co_pilot/ai_sql_agent_v4.py  # generate sql queries using frontier llm model and plot visuals based on results

6. Run SQL agent scripts inside the data_science_co_pilot/ folder
   ```base
   python data_science_co_pilot/init_db_complex.py                     # To set up the db for complex query
   python data_science_co_pilot/ai_sql_agent_v5_complex_with_llama.py  # generate complex sql queries using open source llama
   python data_science_co_pilot/ai_sql_agent_v5_complex_with_llama.py  # generate complex sql queries using frontier llm model and plot visuals based on results

