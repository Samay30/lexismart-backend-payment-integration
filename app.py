# -*- coding: utf-8 -*-
"""
LexiSmart Backend (Render-friendly)
- Auth, Stripe subscriptions, Mindmaps, Analytics
- Dyslexia-friendly summarizer with iterative optimization
- TTS via ElevenLabs REST (no SDK)
- Low memory: no sentence-transformers / bert-score / networkx
"""

import os
import io
import re
import json
import math
import time
import hashlib
import logging
import traceback
from datetime import datetime, timedelta
from typing import Optional

import requests
import textstat
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
import stripe
import openai

# -----------------------
# Bootstrap & Logging
# -----------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("lexismart")

# -----------------------
# Flask app & CORS
# -----------------------
app = Flask(__name__)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
allowed_origins = [o for o in (os.getenv("ALLOWED_ORIGINS", "")).split(",") if o] or [
    FRONTEND_URL,
    "https://*.netlify.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
logger.info(f"Allowed CORS origins: {allowed_origins}")
CORS(app, resources={r"/api/*": {"origins": allowed_origins}}, supports_credentials=True)

# -----------------------
# Keys / Config
# -----------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is required")

app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET", "fallback_secret_key")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
jwt = JWTManager(app)

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
stripe.api_key = STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# -----------------------
# Plans
# -----------------------
plans = {
    "free": {"requests": 25, "price_ids": {}},
    "Student": {"requests": 200, "price_ids": {"USD": os.getenv("PRICE_ID_STUDENT_INR")}},
    "Pro": {"requests": 10000, "price_ids": {"USD": os.getenv("PRICE_ID_PRO_INR")}},
}
PLAN_ALIASES = {"student": "Student", "pro": "Pro"}

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
# Database (smaller pool)
# -----------------------
try:
    db_pool = ThreadedConnectionPool(
        minconn=1,
        maxconn=3,  # keep RAM low on Render free tier
        dsn=DATABASE_URL,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
    logger.info("Database connection pool initialized")
except Exception as e:
    logger.error(f"Failed to initialize database pool: {e}")
    raise

def get_db_connection():
    return db_pool.getconn()

def return_db_connection(conn):
    try:
        db_pool.putconn(conn)
    except Exception as e:
        logger.error(f"Failed to return DB connection: {e}")

def execute_query(query, params=None, fetch=False, fetch_one=False):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        if fetch_one:
            result = cursor.fetchone()
        elif fetch:
            result = cursor.fetchall()
        else:
            result = None
        conn.commit()
        return result
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"DB error: {e}\nQuery: {query}\nParams: {params}")
        raise
    finally:
        if conn:
            return_db_connection(conn)

def init_database():
    try:
        execute_query("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                stripe_customer_id VARCHAR(255),
                subscription VARCHAR(50) DEFAULT 'free',
                requests_used INTEGER DEFAULT 0,
                request_limit INTEGER DEFAULT 5,
                reset_date TIMESTAMP DEFAULT NOW() + INTERVAL '7 days',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        execute_query("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                stripe_subscription_id VARCHAR(255) UNIQUE,
                stripe_customer_id VARCHAR(255),
                plan_name VARCHAR(100),
                status VARCHAR(50),
                current_period_start TIMESTAMP,
                current_period_end TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        execute_query("""
            CREATE TABLE IF NOT EXISTS mindmaps (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                title VARCHAR(255),
                nodes_data JSONB,
                edges_data JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        execute_query("""
            CREATE TABLE IF NOT EXISTS usage_analytics (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                action_type VARCHAR(100),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        execute_query("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        execute_query("CREATE INDEX IF NOT EXISTS idx_subscriptions_customer_id ON subscriptions(stripe_customer_id);")
        execute_query("CREATE INDEX IF NOT EXISTS idx_mindmaps_user_id ON mindmaps(user_id);")
        logger.info("Database tables initialized")
    except Exception as e:
        logger.error(f"Failed to initialize DB: {e}")
        raise

# -----------------------
# Common helpers
# -----------------------
def get_user_by_email(email):
    return execute_query("SELECT * FROM users WHERE email = %s", (email,), fetch_one=True)

def get_user_by_id(user_id):
    return execute_query("SELECT * FROM users WHERE id = %s", (user_id,), fetch_one=True)

def create_user(email, password_hash, stripe_customer_id=None):
    return execute_query(
        """
        INSERT INTO users (email, password_hash, stripe_customer_id, subscription, requests_used, request_limit, reset_date)
        VALUES (%s, %s, %s, 'free', 0, %s, %s)
        RETURNING id
        """,
        (email, password_hash, stripe_customer_id, plans["free"]["requests"], datetime.utcnow() + timedelta(days=7)),
        fetch_one=True,
    )

def update_user_subscription(user_id, plan_name, request_limit, reset_date=None):
    if reset_date is None:
        reset_date = datetime.utcnow() + timedelta(days=30)
    execute_query(
        """
        UPDATE users
        SET subscription = %s, request_limit = %s, reset_date = %s,
            requests_used = 0, updated_at = NOW()
        WHERE id = %s
        """,
        (plan_name, request_limit, reset_date, user_id),
    )

def increment_user_requests(user_id):
    execute_query("UPDATE users SET requests_used = requests_used + 1, updated_at = NOW() WHERE id = %s", (user_id,))

def log_usage(user_id, action_type, metadata=None):
    try:
        execute_query(
            "INSERT INTO usage_analytics (user_id, action_type, metadata) VALUES (%s, %s, %s)",
            (user_id, action_type, json.dumps(metadata) if metadata else None),
        )
    except Exception as e:
        logger.warning(f"Failed to log usage: {e}")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    return hash_password(password) == password_hash

# -----------------------
# Health
# -----------------------
@app.route("/", methods=["GET", "HEAD"])
def root():
    return jsonify(status="ok"), 200

@app.get("/api/health")
def health_check():
    try:
        execute_query("SELECT 1", fetch_one=True)
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return jsonify({
        "status": "healthy",
        "database": db_status,
        "openai_configured": bool(OPENAI_API_KEY),
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY),
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "allowed_origins": allowed_origins
    }), 200

# -----------------------
# Auth
# -----------------------
@app.post("/api/register")
def register_post():
    try:
        data = request.get_json() or {}
        email = data.get("email", "").strip().lower()
        password = data.get("password", "").strip()
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters long"}), 400
        if "@" not in email or "." not in email:
            return jsonify({"error": "Please enter a valid email address"}), 400

        existing_user = get_user_by_email(email)
        if existing_user:
            return jsonify({"error": "An account with this email already exists"}), 400

        stripe_customer_id = None
        if STRIPE_SECRET_KEY:
            try:
                customer = stripe.Customer.create(email=email)
                stripe_customer_id = customer.id
                logger.info(f"Created Stripe customer {stripe_customer_id} for {email}")
            except stripe.error.StripeError as e:
                logger.error(f"Stripe customer creation failed: {e}")

        password_hash = hash_password(password)
        user_result = create_user(email, password_hash, stripe_customer_id)
        if not user_result:
            return jsonify({"error": "Failed to create user account"}), 500

        logger.info(f"User registered: {email}")
        return jsonify({"message": "Account created successfully! Please login."}), 201

    except Exception as e:
        logger.error(f"Registration error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Registration failed. Please try again later."}), 500

@app.post("/api/login")
def login():
    try:
        data = request.get_json() or {}
        email = data.get("email", "").strip().lower()
        password = data.get("password", "").strip()
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        user = get_user_by_email(email)
        if not user or not verify_password(password, user["password_hash"]):
            return jsonify({"error": "Invalid email or password"}), 401

        log_usage(user["id"], "login")
        access_token = create_access_token(identity=str(user["id"]), expires_delta=timedelta(hours=24))

        return jsonify({
            "access_token": access_token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "subscription": user["subscription"],
                "requests_used": user["requests_used"],
                "request_limit": user["request_limit"]
            }
        }), 200

    except Exception as e:
        logger.error(f"Login error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Login failed. Please try again later."}), 500

@app.get("/api/user")
@jwt_required()
def get_user():
    try:
        user_id = get_jwt_identity()
        user = get_user_by_id(int(user_id))
        if not user:
            return jsonify({"error": "User not found"}), 404

        return jsonify({
            "id": user["id"],
            "email": user["email"],
            "subscription": user["subscription"],
            "requests_used": user["requests_used"],
            "request_limit": user["request_limit"],
            "reset_date": user["reset_date"].isoformat() if user["reset_date"] else None
        }), 200
    except Exception as e:
        logger.error(f"Get user error: {e}")
        return jsonify({"error": "Failed to fetch user data"}), 500

# -----------------------
# Stripe
# -----------------------
@app.get("/api/payment-health")
def payment_health():
    status = {
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "webhook_configured": bool(STRIPE_WEBHOOK_SECRET),
        "price_ids_configured": {
            "student_inr": bool(plans["Student"]["price_ids"].get("USD")),
            "pro_inr": bool(plans["Pro"]["price_ids"].get("USD")),
        },
        "frontend_url": FRONTEND_URL
    }

    if STRIPE_SECRET_KEY:
        try:
            acct = stripe.Account.retrieve()
            status["stripe_connection"] = "OK"
            status["account_country"] = acct.get("country")
            status["default_currency"] = acct.get("default_currency")
        except Exception as e:
            status["stripe_connection"] = f"Error: {e}"
    else:
        status["stripe_connection"] = "Not configured"
    return jsonify(status), 200

@app.get("/api/payment-config")
def payment_config():
    out = {"currency": "USD", "plans": {}}
    for key in ("Student", "Pro"):
        pid = plans[key]["price_ids"].get("USD")
        if not pid:
            out["plans"][key] = {"amount": None, "interval": "month", "price_id": None}
            continue
        try:
            price = stripe.Price.retrieve(pid)
            amount_major = (price["unit_amount"] or 0) / 100.0
            interval = price.get("recurring", {}).get("interval", "month")
            out["plans"][key] = {"amount": amount_major, "interval": interval, "price_id": pid}
        except Exception as e:
            logger.error(f"Failed to fetch price {pid}: {e}")
            out["plans"][key] = {"amount": None, "interval": "month", "price_id": pid}
    return jsonify(out), 200

@app.post("/api/create-subscription")
@jwt_required()
def create_subscription():
    try:
        user_id = get_jwt_identity()
        user = get_user_by_id(int(user_id))
        if not user:
            return jsonify({"error": "User not found"}), 404

        data = request.get_json() or {}
        plan_input = data.get("plan")
        plan_key = resolve_plan_key(plan_input)
        logger.info(f"Creating subscription for user {user_id}: {plan_input} -> {plan_key}")

        if plan_key not in plans or plan_key == "free":
            return jsonify({"error": f"Invalid plan: {plan_input}"}), 400

        price_id = plans[plan_key]["price_ids"].get("USD")
        if not price_id:
            return jsonify({"error": f"Price ID (USD) not configured for plan {plan_key}"}), 500

        customer_id = user.get("stripe_customer_id")
        if not customer_id:
            try:
                customer = stripe.Customer.create(email=user["email"])
                customer_id = customer.id
                execute_query("UPDATE users SET stripe_customer_id = %s WHERE id = %s", (customer_id, user_id))
                logger.info(f"Created Stripe customer {customer_id} for existing user")
            except stripe.error.StripeError as e:
                logger.error(f"Failed to create Stripe customer: {e}")
                return jsonify({"error": "Payment system error"}), 500
        else:
            try:
                stripe.Customer.retrieve(customer_id)
            except stripe.error.InvalidRequestError:
                customer = stripe.Customer.create(email=user["email"])
                customer_id = customer.id
                execute_query("UPDATE users SET stripe_customer_id = %s WHERE id = %s", (customer_id, user_id))

        try:
            session = stripe.checkout.Session.create(
                customer=customer_id,
                mode="subscription",
                line_items=[{"price": price_id, "quantity": 1}],
                success_url=f"{FRONTEND_URL}/success?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{FRONTEND_URL}/cancel",
                allow_promotion_codes=True,
                billing_address_collection="required",
                metadata={"user_id": user_id, "user_email": user["email"], "plan_type": plan_key},
                client_reference_id=str(user_id),
            )
            log_usage(int(user_id), "subscription_checkout_created", {"plan": plan_key, "session_id": session.id})
            return jsonify({"sessionId": session.id, "currency_used": "USD"}), 200
        except stripe.error.StripeError as e:
            msg = getattr(e, "user_message", "") or str(e)
            logger.error(f"Stripe error creating session: {msg}")
            return jsonify({"error": f"Payment processing failed: {msg}"}), 500

    except Exception as e:
        logger.error(f"Subscription creation error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.post("/api/stripe-webhook")
def stripe_webhook():
    if not STRIPE_WEBHOOK_SECRET:
        logger.error("Webhook secret not configured")
        return jsonify({"error": "Webhook secret not configured"}), 500

    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        logger.info(f"Webhook event received: {event['type']}")
    except ValueError as e:
        logger.error(f"Invalid webhook payload: {e}")
        return jsonify({"error": "Invalid payload"}), 400
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid webhook signature: {e}")
        return jsonify({"error": "Invalid signature"}), 400

    try:
        event_type = event["type"]
        obj = event["data"]["object"]
        if event_type == "checkout.session.completed":
            handle_checkout_completed(obj)
        elif event_type == "invoice.paid":
            handle_invoice_paid(obj)
        elif event_type == "customer.subscription.updated":
            handle_subscription_updated(obj)
        elif event_type == "customer.subscription.deleted":
            handle_subscription_deleted(obj)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Webhook processing failed"}), 500

def handle_checkout_completed(session):
    try:
        customer_id = session.get("customer")
        user_id = session.get("client_reference_id")
        if user_id:
            log_usage(int(user_id), "checkout_completed", {"session_id": session["id"]})
        logger.info(f"Checkout completed for customer {customer_id}, user {user_id}")
    except Exception as e:
        logger.error(f"Error handling checkout completion: {e}")

def handle_invoice_paid(invoice):
    try:
        customer_id = invoice.get("customer")
        sub_id = invoice.get("subscription")
        if customer_id and sub_id:
            subscription = stripe.Subscription.retrieve(sub_id)
            apply_subscription_state(customer_id, subscription)
    except Exception as e:
        logger.error(f"Error handling invoice paid: {e}")

def handle_subscription_updated(subscription):
    try:
        customer_id = subscription.get("customer")
        apply_subscription_state(customer_id, subscription)
    except Exception as e:
        logger.error(f"Error handling subscription update: {e}")

def handle_subscription_deleted(subscription):
    try:
        customer_id = subscription.get("customer")
        downgrade_to_free(customer_id)
    except Exception as e:
        logger.error(f"Error handling subscription deletion: {e}")

def apply_subscription_state(customer_id: str, subscription: dict):
    try:
        price_id = subscription["items"]["data"][0]["price"]["id"]
        plan_name = resolve_plan_by_price_id(price_id)
        if not plan_name:
            logger.error(f"Unknown price ID in subscription: {price_id}")
            return
        user = execute_query("SELECT id FROM users WHERE stripe_customer_id = %s", (customer_id,), fetch_one=True)
        if not user:
            logger.warning(f"No user found for Stripe customer {customer_id}")
            return
        user_id = user["id"]
        current_period_end = subscription.get("current_period_end")
        reset_date = datetime.utcfromtimestamp(current_period_end) if current_period_end else None
        update_user_subscription(user_id, plan_name, plans[plan_name]["requests"], reset_date)
        execute_query(
            """
            INSERT INTO subscriptions 
            (user_id, stripe_subscription_id, stripe_customer_id, plan_name, status, current_period_start, current_period_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (stripe_subscription_id)
            DO UPDATE SET 
                plan_name = EXCLUDED.plan_name,
                status = EXCLUDED.status,
                current_period_end = EXCLUDED.current_period_end,
                updated_at = NOW()
            """,
            (
                user_id,
                subscription["id"],
                customer_id,
                plan_name,
                subscription["status"],
                datetime.utcfromtimestamp(subscription.get("current_period_start", 0)),
                reset_date,
            ),
        )
        log_usage(user_id, "subscription_activated", {"plan": plan_name})
        logger.info(f"User {user_id} upgraded to {plan_name}")
    except Exception as e:
        logger.error(f"Error applying subscription state: {e}")
        logger.error(traceback.format_exc())

def downgrade_to_free(customer_id: str):
    try:
        user = execute_query("SELECT id FROM users WHERE stripe_customer_id = %s", (customer_id,), fetch_one=True)
        if not user:
            logger.warning(f"No user found for Stripe customer {customer_id}")
            return
        user_id = user["id"]
        update_user_subscription(user_id, "free", plans["free"]["requests"])
        log_usage(user_id, "subscription_cancelled")
        logger.info(f"User {user_id} downgraded to free plan")
    except Exception as e:
        logger.error(f"Error downgrading user: {e}")

# -----------------------
# Dyslexia-friendly summarizer
# -----------------------
class DyslexiaFriendlyConfig:
    SUMMARY_MAX_TOKENS = 300

def flesch_reading_ease(text: str) -> float:
    try:
        fre = float(textstat.flesch_reading_ease(text))
        return max(0.0, min(100.0, fre))
    except Exception:
        return 0.0

def _cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

def _embed(text: str) -> list[float]:
    # Uses small/cheap model; truncate long inputs
    resp = openai.Embedding.create(model="text-embedding-3-small", input=text[:8000])
    return resp["data"][0]["embedding"]

def semantic_score(candidate: str, source: str) -> float:
    """Cosine similarity of OpenAI embeddings, mapped to [0,1]."""
    try:
        ec = _embed(candidate)
        es = _embed(source)
        sim = _cosine(ec, es)
        return float(max(0.0, min(1.0, sim)))
    except Exception:
        # light fallback: token overlap
        cand = set(re.findall(r"\b\w+\b", candidate.lower()))
        src = set(re.findall(r"\b\w+\b", source.lower()))
        inter = len(cand & src)
        return (inter / max(1, len(cand))) ** 0.5

def balance_score(fre: float, bert_like: float) -> float:
    return 0.5 * fre + 0.5 * (bert_like * 100.0)

def create_summary_prompt(text: str, target_fre: int, target_sim: float) -> str:
    return f"""
You are a skilled editor for dyslexic readers.

Write a concise, engaging summary with:
- Short, natural sentences (~6–12 words)
- Simple words (mostly 1–2 syllables)
- Clear flow with light connectors (so, but, then, because)
- No headings or bullet points
- 90–140 words total

Quality targets you should aim for:
- Flesch Reading Ease ≥ {target_fre}
- High meaning preservation (similarity ≥ {target_sim})

Structure:
1) What happened
2) Why it matters
3) What might happen next
4) End with one plain-english key takeaway.

Important:
- Keep it cohesive, not choppy.
- Avoid repetitive sentence starters.
- Output ONLY the summary text.

Article:
{text}
"""

@app.route("/api/summarize", methods=["GET", "POST", "OPTIONS"])
@jwt_required(optional=True)
def summarize():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    try:
        if request.method == "GET":
            input_text = (request.args.get("text") or "").strip()
            post_json = {}
        else:
            post_json = request.get_json() or {}
            input_text = (post_json.get("text") or "").strip()

        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        if len(input_text) > 20000:
            input_text = input_text[:20000] + " [TEXT TRUNCATED]"

        user_id = get_jwt_identity()
        user = None
        if user_id:
            user = get_user_by_id(int(user_id))
            if not user:
                return jsonify({"error": "User not found"}), 404
            if user["requests_used"] >= user["request_limit"]:
                return jsonify({
                    "error": "Request limit reached. Upgrade your plan for more requests.",
                    "upgrade_url": f"{FRONTEND_URL}/upgrade"
                }), 402

        target_fre = int(post_json.get("target_fre", 85))
        target_sim = float(post_json.get("target_bert", 0.85))  # name kept for compatibility
        max_attempts = int(post_json.get("max_attempts", 3))

        system_msg = {"role": "system", "content": "You are a careful editor optimizing readability and meaning."}
        base_user_prompt = create_summary_prompt(input_text, target_fre, target_sim)
        messages = [system_msg, {"role": "user", "content": base_user_prompt}]

        best = {"text": None, "fre": -1.0, "sim": -1.0, "balance": -math.inf}
        attempts_used = 0

        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.4,
                max_tokens=DyslexiaFriendlyConfig.SUMMARY_MAX_TOKENS,
            )
            candidate = (resp.choices[0].message.content or "").strip()

            fre = flesch_reading_ease(candidate)
            sim = semantic_score(candidate, input_text)
            bal = balance_score(fre, sim)

            if bal > best["balance"]:
                best.update({"text": candidate, "fre": fre, "sim": sim, "balance": bal})

            if fre >= target_fre and sim >= target_sim:
                break

            tweaks = []
            if fre < target_fre:
                tweaks.append(f"raise FRE to ≥ {target_fre} by using simpler words and slightly shorter sentences")
            if sim < target_sim:
                tweaks.append(f"increase meaning preservation to ≥ {target_sim} by keeping all key facts and names")
            tweak_text = "; ".join(tweaks) if tweaks else "refine flow to improve readability and meaning"

            feedback = (
                f"Scores now: FRE={fre:.1f}, SIM={sim:.3f}, BAL={bal:.1f}. "
                f"Revise to {tweak_text}. Keep 90–140 words, cohesive tone, no bullets, no headings."
            )
            messages.append({"role": "assistant", "content": candidate})
            messages.append({"role": "user", "content": feedback})

        final_text = best["text"] or "Summary unavailable."

        if user_id:
            increment_user_requests(int(user_id))
            log_usage(int(user_id), "summarize", {
                "text_length": len(input_text),
                "attempts": attempts_used,
                "fre": best["fre"],
                "similarity": best["sim"],
                "balance": best["balance"],
                "targets": {"fre": target_fre, "similarity": target_sim},
            })

        return jsonify({
            "summary_text": final_text,
            "scores": {
                "flesch_reading_ease": round(best["fre"], 2),
                "similarity": round(best["sim"], 4),
                "balance": round(best["balance"], 2),
                "targets": {"fre": target_fre, "similarity": target_sim}
            },
            "dyslexia_friendly": True,
            "attempts_used": attempts_used
        }), 200

    except Exception as e:
        logger.error(f"Summarization error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Summary generation failed"}), 500

# -----------------------
# TTS (ElevenLabs REST)
# -----------------------
@app.post("/api/synthesize")
@jwt_required()
def synthesize():
    try:
        user_id = get_jwt_identity()
        body = request.get_json() or {}
        text = (body.get("text") or "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if not ELEVENLABS_API_KEY:
            return jsonify({"error": "Voice service not configured"}), 500

        VOICE_ID = "ZT9u07TYPVl83ejeLakq"  # Rachel
        MAX_TEXT_LENGTH = 1000
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        DEFAULT_SPEED = 0.9
        speed = body.get("speed", DEFAULT_SPEED)
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = DEFAULT_SPEED
        speed = max(0.7, min(1.2, speed))

        add_pauses = bool(body.get("add_pauses", False))
        pause_seconds = float(body.get("pause_seconds", 0.6))

        if add_pauses:
            model_id = "eleven_turbo_v2_5"
            sent_split = re.sub(r'([.!?])(\s+)', r'\1 <break time="%.1fs" /> ' % pause_seconds, text)
            text_payload = f"<speak>{sent_split}</speak>"
        else:
            model_id = "eleven_monolingual_v1"
            text_payload = text

        headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
        payload = {
            "text": text_payload,
            "model_id": model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "speed": speed},
        }
        if add_pauses:
            payload["enable_ssml_parsing"] = True

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            audio_bytes = resp.content
            log_usage(int(user_id), "tts", {"text_length": len(text), "speed": speed, "add_pauses": add_pauses})
            return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", as_attachment=False, download_name="lexismart_audio.mp3")
        except Exception as e:
            logger.error(f"TTS API error: {e}")
            return jsonify({"error": "Voice generation failed. Please try again."}), 500

    except Exception as e:
        logger.error(f"TTS system error: {e}")
        return jsonify({"error": "Voice service unavailable"}), 500

# -----------------------
# Mind Map Storage
# -----------------------
@app.post("/api/save-mindmap")
@jwt_required()
def save_mindmap():
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        title = data.get("title", "Untitled Mind Map")
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        execute_query(
            "INSERT INTO mindmaps (user_id, title, nodes_data, edges_data) VALUES (%s, %s, %s, %s) RETURNING id",
            (int(user_id), title, json.dumps(nodes), json.dumps(edges)),
            fetch_one=True,
        )
        log_usage(int(user_id), "mindmap_saved", {"nodes_count": len(nodes)})
        return jsonify({"message": "Mind map saved successfully"}), 200
    except Exception as e:
        logger.error(f"Save mindmap error: {e}")
        return jsonify({"error": "Failed to save mind map"}), 500

@app.get("/api/mindmaps")
@jwt_required()
def get_mindmaps():
    try:
        user_id = get_jwt_identity()
        mindmaps = execute_query(
            """
            SELECT id, title, created_at, updated_at
            FROM mindmaps WHERE user_id = %s ORDER BY updated_at DESC
            """,
            (int(user_id),),
            fetch=True,
        )
        return jsonify({
            "mindmaps": [{
                "id": mm["id"],
                "title": mm["title"],
                "created_at": mm["created_at"].isoformat(),
                "updated_at": mm["updated_at"].isoformat()
            } for mm in mindmaps]
        }), 200
    except Exception as e:
        logger.error(f"Get mindmaps error: {e}")
        return jsonify({"error": "Failed to fetch mind maps"}), 500

@app.get("/api/mindmap/<int:mindmap_id>")
@jwt_required()
def get_mindmap(mindmap_id):
    try:
        user_id = get_jwt_identity()
        mindmap = execute_query(
            "SELECT nodes_data, edges_data, title FROM mindmaps WHERE id = %s AND user_id = %s",
            (mindmap_id, int(user_id)),
            fetch_one=True,
        )
        if not mindmap:
            return jsonify({"error": "Mind map not found"}), 404
        return jsonify({
            "title": mindmap["title"],
            "nodes": mindmap["nodes_data"],
            "edges": mindmap["edges_data"]
        }), 200
    except Exception as e:
        logger.error(f"Get mindmap error: {e}")
        return jsonify({"error": "Failed to fetch mind map"}), 500

# -----------------------
# Simple Entity Extraction
# -----------------------
def simple_entity_extraction(text):
    # Very light heuristic capitalized-phrase finder
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    return list(set(entities))

@app.post("/api/extract-entities")
@jwt_required()
def extract_entities():
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        entities = simple_entity_extraction(text)
        log_usage(int(user_id), "entity_extraction", {"entities_count": len(entities)})
        return jsonify({"entities": entities}), 200
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# -----------------------
# Analytics
# -----------------------
@app.get("/api/analytics")
@jwt_required()
def get_analytics():
    try:
        user_id = get_jwt_identity()
        stats = execute_query(
            """
            SELECT action_type, COUNT(*) as count, DATE(created_at) as date
            FROM usage_analytics
            WHERE user_id = %s AND created_at >= NOW() - INTERVAL '30 days'
            GROUP BY action_type, DATE(created_at)
            ORDER BY date DESC
            """,
            (int(user_id),),
            fetch=True,
        )
        return jsonify({"usage_stats": [{
            "action": s["action_type"],
            "count": s["count"],
            "date": s["date"].isoformat()
        } for s in stats]}), 200
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({"error": "Failed to fetch analytics"}), 500

# -----------------------
# Environment validation & startup
# -----------------------
def validate_environment() -> bool:
    required = {
        "DATABASE_URL": DATABASE_URL,
        "FRONTEND_URL": FRONTEND_URL,
        "JWT_SECRET": os.getenv("JWT_SECRET"),
        "OPENAI_API_KEY": OPENAI_API_KEY,
        # Stripe keys optional for non-payment flows:
        "STRIPE_SECRET_KEY": STRIPE_SECRET_KEY or "(optional)",
    }
    missing = [k for k, v in required.items() if not v or v == "(optional)"]
    if "OPENAI_API_KEY" in missing or "DATABASE_URL" in missing:
        logger.error(f"Missing required envs: {missing}")
        return False

    try:
        execute_query("SELECT 1", fetch_one=True)
        logger.info("Database connection validated")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

    if STRIPE_SECRET_KEY:
        try:
            stripe.Account.retrieve()
            logger.info("Stripe API key validated")
        except stripe.error.AuthenticationError:
            logger.error("Invalid Stripe API key")
            return False
        except Exception as e:
            logger.error(f"Stripe validation warning: {e}")

    return True

if __name__ in ("__main__", "app"):
    try:
        if not validate_environment():
            logger.error("Environment validation failed. Check configuration.")
            raise SystemExit(1)
        init_database()
        logger.info("LexiSmart backend started successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.error(traceback.format_exc())
        raise SystemExit(1)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") == "development"
    # If deploying with gunicorn on Render, also set WEB_CONCURRENCY=1
    app.run(host="0.0.0.0", port=port, debug=debug)
