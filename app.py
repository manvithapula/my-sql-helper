import logging
import json
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model
MODEL_NAME = "mrm8488/t5-small-finetuned-wikiSQL"
try:
    logging.info(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise RuntimeError("Model could not be loaded.")

# SQL generation logic
def generate_sql_query(question):
    try:
        logging.info(f"Generating SQL query for: {question}")
        input_text = f"translate English to SQL: {question}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        outputs = model.generate(input_ids, max_length=256)
        generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logging.info(f"Generated SQL: {generated_sql}")
        return generated_sql

    except Exception as e:
        logging.error(f"Error during SQL generation: {e}")
        return "ERROR: Failed to generate SQL"

# Route: home page
@app.route('/')
def home():
    return render_template('index.html')

# Route: translation API
@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    question = data.get('question')

    if not question:
        logging.warning("Received request without a question.")
        return jsonify({"error": "No question provided"}), 400

    logging.info(f"Received translation request: {question}")
    sql_query = generate_sql_query(question)

    logging.info(f"Returning SQL query: {sql_query}")
    return jsonify({"sql_query": sql_query})

# Route: evaluation dashboard
@app.route('/evaluation')
def evaluation():
    try:
        with open('evaluation_results.json') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Could not load evaluation data: {e}")
        return "Evaluation data not available.", 500

    accuracy = data.get("accuracy", 0)
    avg_similarity = data.get("avg_similarity", 0)
    scores = data.get("similarity_scores", [])

    fig, ax = plt.subplots()
    ax.hist(scores, bins=10, color='skyblue')
    ax.set_title("Similarity Score Distribution")
    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Frequency")

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return render_template("evaluation.html",
                           accuracy=round(accuracy, 3),
                           avg_similarity=round(avg_similarity, 3),
                           plot_image=plot_data)

# Start server
if __name__ == '__main__':
    logging.info("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
