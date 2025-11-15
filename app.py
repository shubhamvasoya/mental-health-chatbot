from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import sys

print("Starting SoulCare initialization...", file=sys.stderr)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

print("Loading environment variables...", file=sys.stderr)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY not found in .env file!", file=sys.stderr)
    sys.exit(1)

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file!", file=sys.stderr)
    sys.exit(1)

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

print("Loading embeddings...", file=sys.stderr)
try:
    embeddings = download_hugging_face_embeddings()
    print("✓ Embeddings loaded", file=sys.stderr)
except Exception as e:
    print(f"ERROR loading embeddings: {str(e)}", file=sys.stderr)
    sys.exit(1)

print("Connecting to Pinecone...", file=sys.stderr)
try:
    index_name = "mental-health-chatbot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    print("✓ Pinecone connected", file=sys.stderr)
except Exception as e:
    print(f"ERROR connecting to Pinecone: {str(e)}", file=sys.stderr)
    print("Make sure your Pinecone index 'mental-health-chatbot' exists!", file=sys.stderr)
    sys.exit(1)

print("Initializing retriever...", file=sys.stderr)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("✓ Retriever initialized", file=sys.stderr)

print("Initializing ChatGoogleGenerativeAI...", file=sys.stderr)
chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
print("✓ Chat model initialized", file=sys.stderr)

print("SoulCare initialization complete!", file=sys.stderr)


def get_system_prompt(age, gender, occupation):
    """Generate system prompt with user context"""
    occ_text = occupation if occupation else "Not specified"

    system_text = (
        "You are SoulCare — a warm, empathetic, and intelligent mental wellbeing assistant. "
        "Your purpose is to help users through emotional support and practical guidance. "
        "USER CONTEXT: "
        f"Age: {age}, Gender: {gender}, Occupation: {occ_text} "
        "Analyze what the user needs: "
        "1. EMOTIONAL - Provide empathy, validation, reassurance, coping strategies "
        "2. TECHNICAL/PRACTICAL - Provide step-by-step solutions and actionable strategies "
        "3. ADVICE - Present multiple perspectives, pros/cons, help clarify values "
        "4. STRESS - Validate pressure while breaking problems in manageable steps "
        "5. WELLNESS - Provide evidence-based health recommendations and stress factors "
        "6. VENTING - Listen empathetically, validate, normalize; don't over-solve "
        "TONE: Always validate first, match their energy, use specific examples, balance comfort with action. "
        "Be warm but effective. If serious mental health crisis, suggest professional help. "
    )

    return system_text


def analyze_user_intent(message):
    """Analyze user message to determine intent and sentiment"""
    message_lower = message.lower()

    emotional_keywords = [
        'feel', 'feeling', 'sad', 'depressed', 'anxious', 'worried', 'stress', 'stressed',
        'overwhelmed', 'scared', 'afraid', 'lonely', 'struggling', 'broken', 'hurt',
        'angry', 'frustrated', 'disappointed', 'devastated', 'heartbroken', 'numb', 'empty',
        'suicidal', 'harm', 'hurt myself', "can't take it", "can't handle", 'breaking down',
        'panic', 'panicking', 'crying', 'cry', 'exhausted', 'tired', 'depressing', 'terrible'
    ]

    technical_keywords = [
        'how to', 'how do i', 'help me', 'can you explain', 'debug', 'code', 'error',
        'problem', 'issue', 'fix', 'solution', 'advice on', 'tips for', 'steps', 'process',
        'way to', 'method', 'technique', 'approach', 'strategy', 'algorithm', 'implement',
        'build', 'create', 'develop', 'learn', 'study', 'understand', 'explain'
    ]

    venting_keywords = [
        'just', 'ugh', 'i hate', 'seriously', 'honestly', 'actually', 'literally',
        'can you imagine', 'no joke', 'can you believe', 'i mean', 'so annoying', 'ridiculous'
    ]

    urgent_keywords = [
        'crisis', 'emergency', 'immediate', 'right now', 'asap', 'urgent', 'please help',
        'dying', 'dead', 'kill myself', 'end it', 'give up', 'hopeless', 'no point'
    ]

    emotional_count = sum(1 for kw in emotional_keywords if kw in message_lower)
    technical_count = sum(1 for kw in technical_keywords if kw in message_lower)
    venting_count = sum(1 for kw in venting_keywords if kw in message_lower)
    urgent_count = sum(1 for kw in urgent_keywords if kw in message_lower)

    intent = 'venting'
    if urgent_count > 0:
        intent = 'emergency'
    elif technical_count > emotional_count and technical_count > 0:
        intent = 'technical'
    elif emotional_count > technical_count and emotional_count > 0:
        intent = 'emotional'
    elif 'advice' in message_lower or 'should i' in message_lower:
        intent = 'advice'
    elif any(kw in message_lower for kw in ['sleep', 'exercise', 'eat', 'diet', 'fitness']):
        intent = 'wellness'
    elif venting_count > 0:
        intent = 'venting'

    negative_indicators = ['not', 'no', "can't", "don't", "won't", 'fail', 'bad']
    negative_count = sum(1 for ind in negative_indicators if ind in message_lower)

    if urgent_count > 0:
        sentiment = 'urgent'
    elif emotional_count > 3 and negative_count > 0:
        sentiment = 'negative'
    elif any(pos in message_lower for pos in ['great', 'good', 'happy', 'better', 'amazing']):
        sentiment = 'positive'
    else:
        sentiment = 'neutral'

    emotional_level = 'low'
    if urgent_count > 0:
        emotional_level = 'high'
    elif emotional_count > 3:
        emotional_level = 'high'
    elif emotional_count > 1:
        emotional_level = 'medium'

    return {
        'intent': intent,
        'sentiment': sentiment,
        'emotional_level': emotional_level
    }


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/submit-intake", methods=["POST"])
def submit_intake():
    """Handle user intake form submission"""
    try:
        data = request.get_json()
        age = data.get("age")
        gender = data.get("gender")
        occupation = data.get("occupation", "")

        if not age or not gender:
            return jsonify({"success": False, "message": "Age and gender are required"}), 400

        session['user_age'] = age
        session['user_gender'] = gender
        session['user_occupation'] = occupation
        session['intake_complete'] = True

        return jsonify({"success": True, "message": "Welcome! We have your information. Let's get started."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/get", methods=["GET", "POST"])
def chat():
    """Handle chat messages with sentiment analysis and context-aware responses"""
    try:
        msg = request.form.get("msg")
        if not session.get('intake_complete'):
            return jsonify({"success": False, "message": "Please complete the intake form first"}), 400

        age = session.get('user_age')
        gender = session.get('user_gender')
        occupation = session.get('user_occupation', '')

        analysis = analyze_user_intent(msg)

        system_prompt_text = get_system_prompt(age, gender, occupation)

        intent_upper = analysis['intent'].upper()
        sentiment = analysis['sentiment']
        emotional_level = analysis['emotional_level']

        intent_context = (
            f"User Intent: {intent_upper} | "
            f"Sentiment: {sentiment} | "
            f"Emotional Level: {emotional_level} | "
            f"Personalize response accordingly."
        )

        full_system_prompt = f"""{system_prompt_text}

{intent_context}

Context from knowledge base:
{{context}}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", full_system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        print(f"User: {msg}", file=sys.stderr)
        print(f"Analysis: {analysis}", file=sys.stderr)

        response = rag_chain.invoke({"input": msg})

        return str(response['answer'])
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return str(e)


if __name__ == '__main__':
    print("Open your browser and go to: localhost:8080", file=sys.stderr)
    app.run(host="0.0.0.0", port=8080, debug=True)