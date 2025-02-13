import os
import json
import csv
import openai
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from crewai import Crew, Task, Agent
from crewai import LLM
from pinecone import Pinecone, ServerlessSpec

# Load API Keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "antibody_data")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_ENV:
    raise ValueError("Missing API keys. Set them as environment variables.")

openai_client = openai.OpenAI()

# Flask setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "../templates")
STATIC_DIR = os.path.join(BASE_DIR, "../static")

table_name = PINECONE_INDEX_NAME
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it does not exist
if table_name not in pc.list_indexes().names():
    pc.create_index(
        name=table_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

pinecone_index = pc.Index(table_name)

# CSV File Path
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(DATA_DIR, "mock_data.csv")

def get_embeddings(text):
    """Generate embeddings for given text."""
    return model.encode(text).tolist()

def load_data_into_pinecone():
    """Loads antibody troubleshooting data into Pinecone from mock_data.csv."""
    if not os.path.exists(CSV_FILE):
        print(f"⚠️ CSV file {CSV_FILE} not found. Skipping database population.")
        return

    with open(CSV_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_headers = {"Product Name", "Product Type", "Issue Description", "Resolution Suggestion"}
        
        if not required_headers.issubset(reader.fieldnames):
            raise ValueError(f"🚨 CSV headers mismatch! Expected: {required_headers}, Found: {reader.fieldnames}")

        vectors = []
        for idx, row in enumerate(reader):
            doc_id = str(idx + 1)
            product_name = row["Product Name"].strip()
            product_type = row["Product Type"].strip()
            issue = row["Issue Description"].strip()
            resolution = row["Resolution Suggestion"].strip()

            if not product_name or not issue or not resolution:
                print(f"⚠️ Skipping incomplete entry at line {idx + 2}")
                continue

            document_text = f"{product_name} ({product_type}): {issue} -> {resolution}"
            embedding = get_embeddings(document_text)

            vectors.append((doc_id, embedding, {"product_name": product_name, "product_type": product_type, "issue": issue, "resolution": resolution}))
        
        pinecone_index.upsert(vectors)
        print(f"✅ Successfully loaded {len(vectors)} entries into Pinecone!")

if pc.describe_index(table_name).total_vector_count == 0:
    load_data_into_pinecone()
else:
    print(f"📊 Total stored entries in Pinecone: {pc.describe_index(table_name).total_vector_count}")

THRESHOLD = 0.7

def search_pinecone(query):
    """Searches for relevant entries in Pinecone."""
    try:
        query_embedding = get_embeddings(query)
        search_results = pinecone_index.query(vector=query_embedding, top_k=5, include_metadata=True)

        for match in search_results.get("matches", []):
            if match["score"] >= THRESHOLD:
                return match["metadata"]
        return None
    except Exception as e:
        print(f"⚠️ Pinecone Query Error: {e}")
        return None

def generate_fallback_response(user_query):
    """Uses OpenAI GPT-3.5 to generate fallback responses if no match is found in Pinecone."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "Provide a brief, precise response relevant to the query."},
                {"role": "user", "content": user_query},
            ],
            max_tokens=50,
        )
        return response.choices[0].message.content if response.choices else "I couldn't process your request."
    except Exception as e:
        print(f"⚠️ OpenAI Fallback Error: {e}")
        return "I'm sorry, but I couldn't process your request."

resolution_agent = Agent(
    name="Resolution Finder",
    role="Antibody Troubleshooting Expert",
    goal="Find best resolution for antibody-related issues using AI.",
    backstory="An AI expert in troubleshooting antibody-related issues across various assays.",
    llm=LLM(model="gpt-3.5-turbo-0125"),
)

def resolve_issue(user_query):
    """Resolves antibody-related issues by searching Pinecone and using AI assistance if needed."""
    pinecone_result = search_pinecone(user_query)
    if pinecone_result:
        return pinecone_result
    
    resolution_task = Task(
        description=f"Find resolution for: {user_query}",
        agent=resolution_agent,
        expected_output="A resolution for the antibody issue."
    )
    crew = Crew(agents=[resolution_agent], tasks=[resolution_task])
    crew_result = crew.kickoff()
    return {"response": str(crew_result) if crew_result else "No resolution found."}

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    """Handles user queries and returns responses."""
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"response": "Please enter a query."})
    response_data = resolve_issue(user_query)
    return jsonify(response_data)

if __name__ == "__main__":
    print(f"🚀 Server running at http://127.0.0.1:5000")
    app.run(debug=True)
