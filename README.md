# NLP-to-SQL Query Generator
Overview
This project converts natural language questions into SQL queries using a fine-tuned T5 model on the WikiSQL dataset. Built with Flask, it features a web interface for query generation and a dashboard for performance metrics. 

Converts questions (e.g., "Show all employees in Sales") to SQL (SELECT * FROM employees WHERE department = 'Sales').
Web interface with Flask for user input and query output.


# Installation

Clone the repository:git clone https://github.com/manvithapula/my-sql-helper.git
cd my-sql-helper

Create and activate a virtual environment:python -m venv venv
source venv/bin/activate  
.\venv\Scripts\activate   


Install dependencies:pip install -r requirements.txt



# Dependencies
Listed in requirements.txt:
flask
flask-cors
transformers
torch
matplotlib
numpy
sentencepiece
tiktoken
protobuf
blobfile
datasets
nltk
tqdm

# Usage

Run the Flask app:python app.py

Example output:
Input: "Show all employees in Sales"
Output: SELECT * FROM employees WHERE department = 'Sales'
