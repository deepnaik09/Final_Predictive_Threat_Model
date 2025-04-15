from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import traceback
import os
import uuid
import re
import time
import markdown
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory

# Load .env for chatbot
load_dotenv()

app = Flask(__name__)
CORS(app)

# ====== Load Predictive Model ======
model = None
try:
    model_path = os.path.join(os.path.dirname(__file__), 'app/models/Predictive_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"⚠️ Model loading error: {str(e)}")

# ====== Predict Route ======
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
        marital_map = {'Single': 0, 'Married': 1, 'Divorced': 2}

        model_input = pd.DataFrame([{
            'AGE': data['age'],
            'AMT_INCOME_TOTAL': data['annual_income'],
            'AMT_CREDIT': data['loan_amount'],
            'AMT_ANNUITY': data['loan_amount'] / (data['tenure'] * 12),
            'AMT_GOODS_PRICE': data['loan_amount'],
            'GENDER': gender_map[data['gender']],
            'MARITAL_STATUS': marital_map[data['marital_status']]
        }])

        expected_columns = ['AGE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                            'AMT_GOODS_PRICE', 'GENDER', 'MARITAL_STATUS']
        model_input = model_input[expected_columns]

        if model_input.isnull().any().any() or (model_input < 0).any().any():
            return jsonify({'error': 'Invalid input'}), 400

        proba = model.predict_proba(model_input.to_numpy())[0][1]
        threshold = 0.25
        prediction = 1 if proba >= threshold else 0

        return jsonify({
            'result': 'Approved' if prediction else 'Denied',
            'score': round(proba, 4)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

# ====== Chatbot Config ======
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")

llm = ChatGroq(
    model=MODEL_NAME,
    groq_api_key=GROQ_API_KEY,
    temperature=0.3
)

FINANCIAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are FinBot, a certified financial assistant."),
    ("human", "{user_input}")
])

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.timeout = 3600

    def get_memory(self, session_id):
        self.cleanup_sessions()
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'memory': ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                ),
                'last_accessed': time.time()
            }
        else:
            self.sessions[session_id]['last_accessed'] = time.time()
        return self.sessions[session_id]['memory']

    def cleanup_sessions(self):
        current_time = time.time()
        self.sessions = {
            k: v for k, v in self.sessions.items()
            if current_time - v['last_accessed'] < self.timeout
        }

session_manager = SessionManager()

chat_chain = RunnableWithMessageHistory(
    FINANCIAL_PROMPT | llm,
    input_messages_key="user_input",
    get_session_history=lambda sid: session_manager.get_memory(sid).chat_memory,
    history_messages_key="chat_history"
)

@app.route("/chat", methods=["POST"])
def handle_chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    session_id = data.get("session_id", str(uuid.uuid4()))

    if not user_input:
        return jsonify({"response": "Empty message"}), 400

    try:
        memory = session_manager.get_memory(session_id)
        response = chat_chain.invoke(
            {"user_input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        response_text = response.content
        is_markdown = bool(re.search(r'(\*|\d+\.)\s', response_text))
        if is_markdown:
            response_text = markdown.markdown(response_text)

        return jsonify({
            "response": response_text,
            "session_id": session_id,
            "is_markdown_list": is_markdown
        })

    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"response": "Chat error"}), 500

@app.route("/welcome", methods=["GET"])
def welcome():
    return jsonify({
        "response": "Welcome! Ask me about anything finance-related.",
        "session_id": str(uuid.uuid4())
    })

# ====== Run Server ======
if __name__ == "__main__":
    app.run(debug=True, port=5000)
