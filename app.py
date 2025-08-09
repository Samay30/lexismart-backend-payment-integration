import os
import io
import re
import json
import time
import hashlib
import logging
import tempfile
from datetime import datetime, timedelta

import requests
import networkx as nx
import stripe
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)

# -----------------------
# Bootstrap & Logging
# -----------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# App & CORS
# -----------------------
app = Flask(__name__)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
CORS(
    app,
    resources={r"/api/*": {"origins": [FRONTEND_URL]}},
    supports_credentials=True
)

# -----------------------
# JWT & Stripe
# -----------------------
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET", "fallback_secret_key")
jwt = JWTManager(app)

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
stripe.api_key = STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# -----------------------
# External APIs / Keys
# -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Try ElevenLabs import patterns
ELEVENLABS_AVAILABLE = False
try:
    from elevenlabs import ElevenLabs, VoiceSettings
    ELEVENLABS_AVAILABLE = True
except Exception:
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings
        ELEVENLABS_AVAILABLE = True
    except Exception:
        try:
            import elevenlabs  # legacy import path
            ELEVENLABS_AVAILABLE = True
        except Exception:
            ELEVENLABS_AVAILABLE = False
            logger.warning("ElevenLabs SDK not available; TTS endpoint will be disabled.")

# OpenAI (legacy SDK usage to match your earlier code)
import openai
openai.api_key = OPENAI_API_KEY

# -----------------------
# App State / “DB” (replace with real DB in prod)
# -----------------------
users_db = {}
subscriptions_db = {}  # stripe_customer_id -> plan name

# Product catalog — INR prices (adaptive pricing active in Stripe)
plans = {
    "free": {
        "requests": 5,
        "price_ids": {}  # no price for free
    },
    "LexiSmart Students": {
        "requests": 200,
        "price_ids": {
            "INR": os.getenv("PRICE_ID_STUDENT_INR")  # price_inr_...
        }
    },
    "LexiSmart Premium": {
        "requests": 10000,
        "price_ids": {
            "INR": os.getenv("PRICE_ID_PRO_INR")  # price_inr_...
        }
    }
}

# Allow shorthand names from frontend
PLAN_ALIASES = {
    "student": "LexiSmart Students",
    "pro": "LexiSmart Premium",
}

def resolve_plan_key(plan_type: str) -> str:
    if not plan_type:
        return ""
    return PLAN_ALIASES.get(plan_type.strip(), plan_type.strip())

def resolve_plan_by_price_id(price_id: str):
    for name, p in plans.items():
        for _, pid in p.get("price_ids", {}).items():
            if pid and pid == price_id:
                return name
    return None

# -----------------------
# AI / Knowledge helpers
# -----------------------
graph = nx.DiGraph()
CONCEPTNET_API = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/RelatedTo&limit=20"
DBPEDIA_API = "http://dbpedia.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

concept_relations = {}
MAX_ATTEMPTS = 5
SUMMARY_MAX_WORDS = 120
READABILITY_THRESHOLD = 85
MAX_TEXT_LENGTH = 1000

def simple_entity_extraction(text):
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    return list(set(entities))

def fetch_conceptnet_relations(concept):
    try:
        url = CONCEPTNET_API.format(concept.replace(" ", "_").lower())
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        related = set()
        for edge in data.get("edges", []):
            if "end" in edge and "label" in edge["end"]:
                end_node = edge["end"]["label"]
                related.add(end_node)
                graph.add_edge(concept, end_node)
        return list(related)
    except Exception as e:
        logger.error(f"ConceptNet error: {e}")
        return []

def fetch_dbpedia_relations(concept):
    try:
        query = f"""
        SELECT ?related WHERE {{
            <http://dbpedia.org/resource/{concept.replace(' ', '_')}> dbo:wikiPageWikiLink ?related .
        }} LIMIT 20
        """
        params = {"query": query, "format": "json"}
        r = requests.get(DBPEDIA_API, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        related = set()
        for res in data.get("results", {}).get("bindings", []):
            if "related" in res and "value" in res["related"]:
                related_concept = res["related"]["value"].split("/")[-1].replace("_", " ")
                related.add(related_concept)
                graph.add_edge(concept, related_concept)
        return list(related)
    except Exception as e:
        logger.error(f"DBPedia error: {e}")
        return []

def fetch_wikidata_relations(concept):
    try:
        params = {"action": "wbsearchentities", "search": concept, "language": "en", "format": "json"}
        r = requests.get(WIKIDATA_API, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        related = set()
        for ent in data.get("search", []):
            if "label" in ent:
                related.add(ent["label"])
                graph.add_edge(concept, ent["label"])
        return list(related)
    except Exception as e:
        logger.error(f"Wikidata error: {e}")
        return []

def expand_concept_dataset(concept):
    if concept in concept_relations:
        return concept_relations[concept]
    related = fetch_conceptnet_relations(concept) + fetch_dbpedia_relations(concept) + fetch_wikidata_relations(concept)
    concept_relations[concept] = related
    return related

def complete_sentence(text):
    if not text:
        return text
    if re.search(r'[.!?]$', text):
        return text
    last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_punct > 0:
        return text[:last_punct + 1]
    return text + '.'

# -----------------------
# Password hashing
# -----------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# -----------------------
# Health & Root
# -----------------------
@app.route("/", methods=["GET", "HEAD"])
def root():
    return jsonify(status="ok"), 200

@app.get("/api/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "openai_configured": bool(OPENAI_API_KEY),
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY),
        "elevenlabs_available": ELEVENLABS_AVAILABLE
    }), 200

# -----------------------
# Auth
# -----------------------
@app.post("/api/register")
def register():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    if email in users_db:
        return jsonify({"error": "User already exists"}), 400

    # Create Stripe customer
    try:
        customer = stripe.Customer.create(email=email)
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        return jsonify({"error": "Payment system error"}), 500

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

@app.post("/api/login")
def login():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    user = users_db.get(email)
    if not user or user["password_hash"] != hash_password(password):
        return jsonify({"error": "Invalid credentials"}), 401

    access_token = create_access_token(identity=email, expires_delta=timedelta(minutes=60))
    return jsonify(access_token=access_token), 200

@app.get("/api/user")
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
    }), 200

# -----------------------
# Payment Config & Health
# -----------------------
@app.get("/api/payment-health")
def payment_health():
    status = {
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "webhook_configured": bool(STRIPE_WEBHOOK_SECRET),
        "price_ids_configured": {
            "student_inr": bool(plans["LexiSmart Students"]["price_ids"].get("INR")),
            "pro_inr": bool(plans["LexiSmart Premium"]["price_ids"].get("INR")),
        },
        "frontend_url": FRONTEND_URL
    }
    try:
        acct = stripe.Account.retrieve()
        status["stripe_connection"] = "OK"
        status["account_country"] = acct.get("country")
        status["default_currency"] = acct.get("default_currency")
    except Exception as e:
        status["stripe_connection"] = f"Error: {e}"
    return jsonify(status), 200

@app.get("/api/payment-config")
def payment_config():
    """
    Returns live amounts & currency from Stripe for your two INR price IDs.
    Frontend can render these exactly as billed.
    """
    out = {"currency": "INR", "plans": {}}
    for key in ("LexiSmart Students", "LexiSmart Premium"):
        pid = plans[key]["price_ids"].get("INR")
        if not pid:
            out["plans"][key] = {"amount": None, "interval": "month", "price_id": None}
            continue
        try:
            price = stripe.Price.retrieve(pid)
            # Stripe returns amount in minor units (paise)
            amount_major = (price["unit_amount"] or 0) / 100.0
            interval = price.get("recurring", {}).get("interval", "month")
            out["plans"][key] = {
                "amount": amount_major,
                "interval": interval,
                "price_id": pid
            }
        except Exception as e:
            logger.error(f"Failed to fetch price {pid}: {e}")
            out["plans"][key] = {"amount": None, "interval": "month", "price_id": pid}
    return jsonify(out), 200

# -----------------------
# Create Subscription (INR)
# -----------------------
@app.post("/api/create-subscription")
@jwt_required()
def create_subscription():
    email = get_jwt_identity()
    user = users_db.get(email)
    if not user:
        return jsonify({"error": "User not found"}), 404

    data = request.get_json() or {}
    plan_input = data.get("plan")  # 'student', 'pro', or product names
    plan_key = resolve_plan_key(plan_input)

    logger.info(f"Creating subscription for {email}: plan_input={plan_input} -> plan_key={plan_key}")
    logger.info(f"Available plans: {list(plans.keys())}")

    if plan_key not in plans or plan_key == "free":
        return jsonify({"error": f"Invalid plan: {plan_input}"}), 400

    price_id = plans[plan_key]["price_ids"].get("INR")
    if not price_id:
        return jsonify({"error": f"Price ID (INR) not configured for plan {plan_key}"}), 400

    # Ensure Stripe customer exists/is valid
    customer_id = user.get("stripe_customer_id")
    try:
        stripe.Customer.retrieve(customer_id)
    except stripe.error.InvalidRequestError:
        cust = stripe.Customer.create(email=email)
        customer_id = cust.id
        user["stripe_customer_id"] = customer_id

    def create_session(pid: str):
        return stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": pid, "quantity": 1}],
            success_url=f"{FRONTEND_URL}/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_URL}/cancel",
            allow_promotion_codes=True,
            billing_address_collection="required",
            metadata={"user_email": email, "plan_type": plan_key},
            client_reference_id=email
        )

    try:
        session = create_session(price_id)
        logger.info(f"Checkout session created: {session.id} (INR)")
        return jsonify({"sessionId": session.id, "currency_used": "INR"}), 200

    except stripe.error.StripeError as e:
        msg = getattr(e, "user_message", "") or str(e)
        logger.error(f"Stripe error: {msg}")
        if isinstance(e, stripe.error.InvalidRequestError):
            return jsonify({"error": f"Invalid request: {msg}"}), 400
        elif isinstance(e, stripe.error.AuthenticationError):
            return jsonify({"error": "Payment system authentication failed"}), 500
        elif isinstance(e, stripe.error.APIConnectionError):
            return jsonify({"error": "Payment system connection failed"}), 503
        else:
            return jsonify({"error": f"Payment processing failed: {msg}"}), 500
    except Exception as e:
        logger.exception("Unexpected error in subscription creation")
        return jsonify({"error": "Internal server error"}), 500

# -----------------------
# Webhook: keep user plan in sync
# -----------------------
@app.post("/api/stripe-webhook")
def stripe_webhook():
    if not STRIPE_WEBHOOK_SECRET:
        logger.error("Webhook secret not configured")
        return jsonify({"error": "Webhook secret not configured"}), 500

    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        logger.info(f"Webhook event: {event['type']}")
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        return jsonify({"error": "Invalid payload"}), 400
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        return jsonify({"error": "Invalid signature"}), 400

    try:
        t = event["type"]
        obj = event["data"]["object"]

        if t == "checkout.session.completed":
            # You can mark a "pending activation" if needed; subscription changes land on invoice.paid
            pass

        if t == "invoice.paid":
            # Subscription is paid and active for the period
            customer_id = obj.get("customer")
            sub_id = obj.get("subscription")
            if customer_id and sub_id:
                sub = stripe.Subscription.retrieve(sub_id)
                _apply_subscription_state(customer_id, sub)

        if t == "customer.subscription.updated":
            sub = obj
            customer_id = sub.get("customer")
            _apply_subscription_state(customer_id, sub)

        if t == "customer.subscription.deleted":
            sub = obj
            customer_id = sub.get("customer")
            _downgrade_to_free(customer_id)

        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.exception(f"Webhook processing error: {e}")
        return jsonify({"error": "Webhook processing failed"}), 500

def _apply_subscription_state(customer_id: str, subscription: dict):
    try:
        price_id = subscription["items"]["data"][0]["price"]["id"]
        plan_name = resolve_plan_by_price_id(price_id)

        if not plan_name:
            logger.error(f"Unknown price ID in subscription: {price_id}")
            return

        # save mapping
        subscriptions_db[customer_id] = plan_name

        # update user
        for email, user in users_db.items():
            if user.get("stripe_customer_id") == customer_id:
                user["subscription"] = plan_name
                user["request_limit"] = plans[plan_name]["requests"]
                user["requests_used"] = 0
                # use current period end from Stripe if present
                cpe = subscription.get("current_period_end")
                if cpe:
                    user["reset_date"] = datetime.utcfromtimestamp(cpe)
                else:
                    user["reset_date"] = datetime.utcnow() + timedelta(days=30)
                logger.info(f"User {email} upgraded to {plan_name}")
                break
        else:
            logger.warning(f"No local user found for customer {customer_id}")
    except Exception as e:
        logger.exception(f"Apply subscription state error: {e}")

def _downgrade_to_free(customer_id: str):
    subscriptions_db[customer_id] = "free"
    for email, user in users_db.items():
        if user.get("stripe_customer_id") == customer_id:
            user["subscription"] = "free"
            user["request_limit"] = plans["free"]["requests"]
            user["requests_used"] = 0
            user["reset_date"] = datetime.utcnow() + timedelta(days=7)
            logger.info(f"User {email} downgraded to free")
            break

# -----------------------
# Core features
# -----------------------
@app.post("/api/summarize")
@jwt_required()
def summarize():
    email = get_jwt_identity()
    user = users_db.get(email)
    if not user:
        return jsonify({"error": "User not found"}), 404

    if user["requests_used"] >= user["request_limit"]:
        return jsonify({
            "error": "Request limit reached. Upgrade your plan.",
            "upgrade_url": f"{FRONTEND_URL}/upgrade"
        }), 402

    data = request.get_json() or {}
    input_text = (data.get("text") or "").strip()
    if not input_text:
        return jsonify({"error": "No text provided."}), 400

    try:
        if len(input_text) > 10000:
            input_text = input_text[:10000] + " [TEXT TRUNCATED]"

        prompt = (
            "Create a dyslexia-friendly summary with these rules:\n"
            "1. Use ultra-short sentences (max 8 words)\n"
            "2. Use simple vocabulary (grade 5 level)\n"
            "3. Break complex ideas into bullet points\n"
            "4. Avoid metaphors and idioms\n\n"
            "Text to summarize:\n"
            f"{input_text}\n\n"
            "Summary:"
        )

        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=SUMMARY_MAX_WORDS
                )
                summary = (resp.choices[0].message.content or "").strip()
                summary = complete_sentence(summary)

                import textstat
                readability = textstat.flesch_reading_ease(summary)

                # success criteria
                if readability >= READABILITY_THRESHOLD or attempt == MAX_ATTEMPTS - 1:
                    user["requests_used"] += 1
                    return jsonify({
                        "summary_text": summary,
                        "readability": readability,
                        "requests_remaining": user["request_limit"] - user["requests_used"]
                    }), 200
                time.sleep(1)
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
                return jsonify({"error": f"AI service error: {e}"}), 500

        return jsonify({"error": "Failed to generate readable summary"}), 500

    except Exception as e:
        logger.exception("Summarization server error")
        return jsonify({"error": "Internal server error"}), 500

@app.post("/api/synthesize")
@jwt_required()
def synthesize():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    if not ELEVENLABS_AVAILABLE:
        return jsonify({"error": "Voice service not available"}), 503
    if not ELEVENLABS_API_KEY:
        return jsonify({"error": "Voice service not configured"}), 500

    VOICES = {"encouraging_female": "ZT9u07TYPVl83ejeLakq"}
    DEFAULT_VOICE = "encouraging_female"

    try:
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        # New SDK pattern
        try:
            client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            stream = client.text_to_speech.convert(
                text=text,
                voice_id=VOICES[DEFAULT_VOICE],
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128",
                voice_settings=VoiceSettings(
                    stability=0.75, similarity_boost=0.75, style=0.0, use_speaker_boost=True
                )
            )
            audio_bytes = b"".join(stream)
            return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", as_attachment=False)
        except Exception:
            # Legacy fallback
            try:
                elevenlabs.set_api_key(ELEVENLABS_API_KEY)  # type: ignore
                audio_bytes = elevenlabs.generate(  # type: ignore
                    text=text,
                    voice=VOICES[DEFAULT_VOICE],
                    model="eleven_turbo_v2_5"
                )
                return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", as_attachment=False)
            except Exception as e2:
                logger.error(f"TTS fallback error: {e2}")
                return jsonify({"error": "Voice generation failed"}), 500

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({"error": "Voice generation failed"}), 500

# -----------------------
# Concept helpers
# -----------------------
@app.post("/api/related-concepts")
def related_concepts():
    try:
        data = request.get_json() or {}
        concept = (data.get("concept") or "").strip()
        if not concept:
            return jsonify({"error": "No concept provided"}), 400
        related = expand_concept_dataset(concept)[:10]
        return jsonify({"concept": concept, "related_concepts": related}), 200
    except Exception as e:
        logger.error(f"Related concepts error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.get("/api/mindmap")
def get_mindmap():
    try:
        nodes = [{"id": node} for node in graph.nodes]
        edges = [{"source": s, "target": t} for s, t in graph.edges]
        return jsonify({"nodes": nodes, "edges": edges}), 200
    except Exception as e:
        logger.error(f"Mindmap error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.post("/api/extract-entities")
def extract_entities():
    try:
        data = request.get_json() or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        entities = simple_entity_extraction(text)
        return jsonify({"entities": entities}), 200
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# -----------------------
# Startup Env Validation
# -----------------------
def validate_environment() -> bool:
    required = {
        "STRIPE_SECRET_KEY": STRIPE_SECRET_KEY,
        "PRICE_ID_STUDENT_INR": plans["LexiSmart Students"]["price_ids"].get("INR"),
        "PRICE_ID_PRO_INR": plans["LexiSmart Premium"]["price_ids"].get("INR"),
        "FRONTEND_URL": FRONTEND_URL,
        "JWT_SECRET": os.getenv("JWT_SECRET"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        return False
    try:
        stripe.Account.retrieve()
        logger.info("Stripe API key validated")
    except stripe.error.AuthenticationError:
        logger.error("Invalid Stripe API key")
        return False
    except Exception as e:
        logger.error(f"Stripe validation error: {e}")
        return False
    return True

if __name__ in ("__main__", "app"):
    if not validate_environment():
        logger.error("Environment validation failed. Please check your configuration.")

# -----------------------
# Local run
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
