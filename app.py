from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import logging
import os
import requests
import spacy
import networkx as nx
import json
from dotenv import load_dotenv
import textstat
import openai
import time
import re
from elevenlabs import ElevenLabs, VoiceSettings
import tempfile
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import stripe
from datetime import datetime, timedelta
import hashlib  # For password hashing

# Load environment variables first
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== JWT & STRIPE INITIALIZATION =====
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET", "fallback_secret_key")
jwt = JWTManager(app)
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# ===== USER MANAGEMENT =====
# In-memory storage (replace with DB in production)
users_db = {}
subscriptions_db = {}
plans = {
    "free": {"requests": 5, "price_id": None},
    "student": {"requests": 200, "price_id": os.getenv("STRIPE_STUDENT_PRICE_ID")},
    "pro": {"requests": 10000, "price_id": os.getenv("STRIPE_PRO_PRICE_ID")}
}

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ===== NLP & AI COMPONENTS =====
import en_core_web_sm
nlp = en_core_web_sm.load()
graph = nx.DiGraph()

# ElevenLabs Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICES = {"encouraging_female": "ZT9u07TYPVl83ejeLakq"}
DEFAULT_VOICE = "encouraging_female"
openai.api_key = os.getenv("OPENAI_API_KEY")

# API Endpoints
CONCEPTNET_API = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/RelatedTo&limit=20"
DBPEDIA_API = "http://dbpedia.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

concept_relations = {}
MAX_ATTEMPTS = 5
SUMMARY_MAX_WORDS = 120
READABILITY_THRESHOLD = 85
MAX_TEXT_LENGTH = 1000

# ===== HELPER FUNCTIONS =====
def fetch_conceptnet_relations(concept):
    url = CONCEPTNET_API.format(concept.replace(" ", "_").lower())
    response = requests.get(url).json()
    related = set()
    for edge in response.get("edges", []):
        end_node = edge["end"]["label"]
        related.add(end_node)
        graph.add_edge(concept, end_node)
    return list(related)

def fetch_dbpedia_relations(concept):
    query = f"""
    SELECT ?related WHERE {{
        <http://dbpedia.org/resource/{concept.replace(' ', '_')}> dbo:wikiPageWikiLink ?related .
    }} LIMIT 20
    """
    params = {"query": query, "format": "json"}
    response = requests.get(DBPEDIA_API, params=params).json()
    related = set()
    for result in response.get("results", {}).get("bindings", []):
        related_concept = result["related"]["value"].split("/")[-1].replace("_", " ")
        related.add(related_concept)
        graph.add_edge(concept, related_concept)
    return list(related)

def fetch_wikidata_relations(concept):
    params = {
        "action": "wbsearchentities",
        "search": concept,
        "language": "en",
        "format": "json"
    }
    response = requests.get(WIKIDATA_API, params=params).json()
    related = set()
    for entity in response.get("search", []):
        related.add(entity["label"])
        graph.add_edge(concept, entity["label"])
    return list(related)

def expand_concept_dataset(concept):
    if concept in concept_relations:
        return concept_relations[concept]
    
    related_concepts = (fetch_conceptnet_relations(concept) + 
                       fetch_dbpedia_relations(concept) + 
                       fetch_wikidata_relations(concept))
    
    concept_relations[concept] = related_concepts
    return related_concepts

def complete_sentence(text):
    if not text:
        return text
    if re.search(r'[.!?]$', text):
        return text
    last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_punct > 0:
        return text[:last_punct + 1]
    return text + '.'

# ===== AUTHENTICATION ENDPOINTS =====
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    
    if email in users_db:
        return jsonify({"error": "User already exists"}), 400
    
    try:
        customer = stripe.Customer.create(email=email)
        users_db[email] = {
            "password_hash": hash_password(password),
            "stripe_customer_id": customer.id,
            "subscription": "free",
            "requests_used": 0,
            "request_limit": plans["free"]["requests"],
            "reset_date": datetime.utcnow() + timedelta(days=7)
        }
        subscriptions_db[customer.id] = "free"
        return jsonify({"message": "User created"}), 201
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        return jsonify({"error": "Payment system error"}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    user = users_db.get(email)
    if not user or user["password_hash"] != hash_password(password):
        return jsonify({"error": "Invalid credentials"}), 401
    
    access_token = create_access_token(identity=email)
    return jsonify(access_token=access_token)

@app.route('/api/user', methods=['GET'])
@jwt_required()
def get_user():
    email = get_jwt_identity()
    user = users_db.get(email)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "email": email,
        "subscription": user["subscription"],
        "requests_used": user["requests_used"],
        "request_limit": user["request_limit"],
        "reset_date": user["reset_date"].isoformat()
    })

# ===== CORE FUNCTIONALITY ENDPOINTS =====
@app.route('/api/summarize', methods=['POST'])
@jwt_required()
def summarize():
    email = get_jwt_identity()
    user = users_db.get(email)
    
    # Check usage limits
    if user["requests_used"] >= user["request_limit"]:
        return jsonify({
            "error": "Request limit reached. Upgrade your plan.",
            "upgrade_url": f"{os.getenv('FRONTEND_URL')}/upgrade"
        }), 402
    
    data = request.get_json()
    input_text = data.get("text", "").strip()

    if not input_text:
        return jsonify({"error": "No text provided."}), 400

    try:
        # Truncate very long text
        if len(input_text) > 10000:
            input_text = input_text[:10000] + " [TEXT TRUNCATED]"
        
        prompt_template = (
            "Create a dyslexia-friendly summary with these rules:\n"
            "1. Use ultra-short sentences (max 8 words)\n"
            "2. Use simple vocabulary (grade 5 level)\n"
            "3. Break complex ideas into bullet points\n"
            "4. Avoid metaphors and idioms\n\n"
            "Text to summarize:\n"
            f"{input_text}\n\n"
            "Summary:"
        )

        # Generate summary
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt_template}],
                    temperature=0.5,
                    max_tokens=SUMMARY_MAX_WORDS
                )
                summary = response.choices[0].message.content.strip()
                summary = complete_sentence(summary)
                readability = textstat.flesch_reading_ease(summary)

                if readability >= READABILITY_THRESHOLD or attempt == MAX_ATTEMPTS - 1:
                    # Update usage after successful generation
                    user["requests_used"] += 1
                    return jsonify({
                        "summary_text": summary,
                        "readability": readability,
                        "requests_remaining": user["request_limit"] - user["requests_used"]
                    })
                time.sleep(1)
            except openai.error.OpenAIError as e:
                logger.error(f"OpenAI API error: {str(e)}")
                return jsonify({"error": f"AI service error: {str(e)}"}), 500

        return jsonify({"error": "Failed to generate readable summary"}), 500
    except Exception as e:
        logger.exception("Unexpected error in summarization")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/synthesize", methods=["POST"])
@jwt_required()
def synthesize():
    data = request.get_json()
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    if not ELEVENLABS_API_KEY:
        return jsonify({"error": "Voice service not configured"}), 500
    
    try:
        # Truncate long text
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
        
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=VOICES[DEFAULT_VOICE],
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.75,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True
            )
        )

        # Stream audio directly without temp file
        audio_bytes = b"".join(audio_stream)
        return send_file(
            io.BytesIO(audio_bytes),
            mimetype='audio/mpeg',
            as_attachment=False
        )
    except Exception as e:
        logger.error(f"TTS Error: {str(e)}")
        return jsonify({"error": "Voice generation failed"}), 500

# ===== SUBSCRIPTION MANAGEMENT =====
@app.route('/api/create-subscription', methods=['POST'])
@jwt_required()
def create_subscription():
    email = get_jwt_identity()
    user = users_db.get(email)
    data = request.get_json()
    plan_type = data.get('plan', 'student')
    
    if plan_type not in plans or not plans[plan_type]["price_id"]:
        return jsonify({"error": "Invalid plan selected"}), 400
    
    try:
        session = stripe.checkout.Session.create(
            customer=user["stripe_customer_id"],
            payment_method_types=['card'],
            line_items=[{
                'price': plans[plan_type]["price_id"],
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f"{os.getenv('FRONTEND_URL')}/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{os.getenv('FRONTEND_URL')}/cancel",
        )
        return jsonify(sessionId=session.id)
    except stripe.error.StripeError as e:
        logger.error(f"Stripe Checkout error: {str(e)}")
        return jsonify({"error": "Payment processing failed"}), 500

@app.route('/api/stripe-webhook', methods=['POST'])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    if not webhook_secret:
        return jsonify({"error": "Webhook secret not configured"}), 500
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except ValueError as e:
        return jsonify({"error": "Invalid payload"}), 400
    except stripe.error.SignatureVerificationError as e:
        return jsonify({"error": "Invalid signature"}), 400
    
    # Handle subscription events
    if event['type'] == 'customer.subscription.updated':
        subscription = event['data']['object']
        customer_id = subscription['customer']
        
        if subscription['status'] == 'active':
            # Determine plan from price ID
            price_id = subscription['items']['data'][0]['price']['id']
            plan_type = "student" if price_id == plans["student"]["price_id"] else "pro"
            
            # Update subscription status
            subscriptions_db[customer_id] = plan_type
            
            # Find user and update plan
            for email, user in users_db.items():
                if user["stripe_customer_id"] == customer_id:
                    user["subscription"] = plan_type
                    user["request_limit"] = plans[plan_type]["requests"]
                    user["reset_date"] = datetime.utcnow() + timedelta(days=30)
                    break
    
    return jsonify({"status": "success"}), 200

# ===== OTHER ENDPOINTS =====
@app.route('/api/related-concepts', methods=['POST'])
def related_concepts():
    try:
        data = request.get_json()
        concept = data.get('concept', '').strip()
        if not concept:
            return jsonify({'error': 'No concept provided'}), 400
        
        related = expand_concept_dataset(concept)[:10]
        return jsonify({'concept': concept, 'related_concepts': related})
    except Exception as e:
        logger.error(f"Related concepts error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/mindmap', methods=['GET'])
def get_mindmap():
    try:
        nodes = [{"id": node} for node in graph.nodes]
        edges = [{"source": source, "target": target} for source, target in graph.edges]
        return jsonify({"nodes": nodes, "edges": edges})
    except Exception as e:
        logger.error(f"Mindmap error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=os.environ.get('FLASK_DEBUG', False))