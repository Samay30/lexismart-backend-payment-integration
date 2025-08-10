import os
import io
import re
import json
import time
import hashlib
import logging
import tempfile
import traceback
from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------
# Database Configuration
# -----------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is required")
    raise ValueError("DATABASE_URL is required")

# Initialize connection pool
try:
    db_pool = ThreadedConnectionPool(
        minconn=1,
        maxconn=10,  # Reduced for Render's free tier
        dsn=DATABASE_URL,
        cursor_factory=psycopg2.extras.RealDictCursor
    )
    logger.info("Database connection pool initialized")
except Exception as e:
    logger.error(f"Failed to initialize database pool: {e}")
    raise

# Database helper functions

def get_db_connection():
    """Get database connection from pool"""
    try:
        return db_pool.getconn()
    except Exception as e:
        logger.error(f"Failed to get database connection: {e}")
        raise

def return_db_connection(conn):
    """Return database connection to pool"""
    try:
        db_pool.putconn(conn)
    except Exception as e:
        logger.error(f"Failed to return database connection: {e}")


def execute_query(query, params=None, fetch=False, fetch_one=False):
    """Execute database query with proper connection handling"""
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
        logger.error(f"Database error: {e}")
        logger.error(f"Query: {query}")
        logger.error(f"Params: {params}")
        raise
    finally:
        if conn:
            return_db_connection(conn)


def init_database():
    """Initialize database tables"""
    try:
        # Users table
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

        # Subscriptions table
        execute_query("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                stripe_subscription_id VARCHAR(255),
                stripe_customer_id VARCHAR(255),
                plan_name VARCHAR(100),
                status VARCHAR(50),
                current_period_start TIMESTAMP,
                current_period_end TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)

        # Mind maps table
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

        # Usage analytics table
        execute_query("""
            CREATE TABLE IF NOT EXISTS usage_analytics (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                action_type VARCHAR(100),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        # Indexes
        execute_query("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        execute_query("CREATE INDEX IF NOT EXISTS idx_subscriptions_customer_id ON subscriptions(stripe_customer_id);")
        execute_query("CREATE INDEX IF NOT EXISTS idx_mindmaps_user_id ON mindmaps(user_id);")

        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

# -----------------------
# Flask app & CORS
# -----------------------
app = Flask(__name__)
# Get allowed origins from environment or use defaults
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
if not allowed_origins or allowed_origins[0] == "":
    allowed_origins = [
        FRONTEND_URL, 
        "https://*.netlify.app",  # Wildcard for Netlify
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]

# Add this to log CORS configuration
logger.info(f"Allowed CORS origins: {allowed_origins}")

CORS(
    app,
    resources={r"/api/*": {"origins": allowed_origins}},
    supports_credentials=True
)

# JWT & Stripe
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET", "fallback_secret_key")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
jwt = JWTManager(app)

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
stripe.api_key = STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# External keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# ElevenLabs import variations
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
            import elevenlabs
            ELEVENLABS_AVAILABLE = True
        except Exception:
            ELEVENLABS_AVAILABLE = False
            logger.warning("ElevenLabs SDK not available; TTS endpoint will be disabled.")

# OpenAI API
import openai
openai.api_key = OPENAI_API_KEY

# Plans
plans = {
    "free": {
        "requests": 25,
        "price_ids": {}
    },
    "Student": {
        "requests": 200,
        "price_ids": {
            "USD": os.getenv("PRICE_ID_STUDENT_INR")
        }
    },
    "Pro": {
        "requests": 10000,
        "price_ids": {
            "USD": os.getenv("PRICE_ID_PRO_INR")
        }
    }
}

PLAN_ALIASES = {
    "student": "Student",
    "pro": "Pro",
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

# More DB helpers
def get_user_by_email(email):
    return execute_query(
        "SELECT * FROM users WHERE email = %s",
        (email,),
        fetch_one=True
    )

def get_user_by_id(user_id):
    return execute_query(
        "SELECT * FROM users WHERE id = %s",
        (user_id,),
        fetch_one=True
    )

def create_user(email, password_hash, stripe_customer_id=None):
    return execute_query(
        """
        INSERT INTO users (email, password_hash, stripe_customer_id, subscription, requests_used, request_limit, reset_date)
        VALUES (%s, %s, %s, 'free', 0, %s, %s)
        RETURNING id
        """,
        (email, password_hash, stripe_customer_id, plans["free"]["requests"], datetime.utcnow() + timedelta(days=7)),
        fetch_one=True
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
        (plan_name, request_limit, reset_date, user_id)
    )

def increment_user_requests(user_id):
    execute_query(
        "UPDATE users SET requests_used = requests_used + 1, updated_at = NOW() WHERE id = %s",
        (user_id,)
    )

def log_usage(user_id, action_type, metadata=None):
    try:
        execute_query(
            "INSERT INTO usage_analytics (user_id, action_type, metadata) VALUES (%s, %s, %s)",
            (user_id, action_type, json.dumps(metadata) if metadata else None)
        )
    except Exception as e:
        logger.warning(f"Failed to log usage: {e}")

# Password hashing

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    return hash_password(password) == password_hash

# Health endpoints
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
        "elevenlabs_available": ELEVENLABS_AVAILABLE,
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "allowed_origins": allowed_origins
    }), 200

# Registration and authentication
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

        logger.info(f"User registered successfully: {email}")
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
        if not user:
            return jsonify({"error": "Invalid email or password"}), 401

        if not verify_password(password, user["password_hash"]):
            return jsonify({"error": "Invalid email or password"}), 401

        log_usage(user["id"], "login")

        access_token = create_access_token(
            identity=str(user["id"]),
            expires_delta=timedelta(hours=24)
        )

        logger.info(f"User logged in successfully: {email}")
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

# Payment health and config
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
            out["plans"][key] = {
                "amount": amount_major,
                "interval": interval,
                "price_id": pid
            }
        except Exception as e:
            logger.error(f"Failed to fetch price {pid}: {e}")
            out["plans"][key] = {"amount": None, "interval": "month", "price_id": pid}
    return jsonify(out), 200

# Subscription creation
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

        logger.info(f"Creating subscription for user {user_id}: plan_input={plan_input} -> plan_key={plan_key}")

        if plan_key not in plans or plan_key == "free":
            return jsonify({"error": f"Invalid plan: {plan_input}"}), 400

        price_id = plans[plan_key]["price_ids"].get("USD")
        if not price_id:
            return jsonify({"error": f"Price ID (INR) not configured for plan {plan_key}"}), 500

        customer_id = user["stripe_customer_id"]
        if not customer_id:
            try:
                customer = stripe.Customer.create(email=user["email"])
                customer_id = customer.id
                execute_query(
                    "UPDATE users SET stripe_customer_id = %s WHERE id = %s",
                    (customer_id, user_id)
                )
                logger.info(f"Created Stripe customer {customer_id} for existing user")
            except stripe.error.StripeError as e:
                logger.error(f"Failed to create Stripe customer: {e}")
                return jsonify({"error": "Payment system error"}), 500
        else:
            try:
                stripe.Customer.retrieve(customer_id)
            except stripe.error.InvalidRequestError:
                try:
                    customer = stripe.Customer.create(email=user["email"])
                    customer_id = customer.id
                    execute_query(
                        "UPDATE users SET stripe_customer_id = %s WHERE id = %s",
                        (customer_id, user_id)
                    )
                except stripe.error.StripeError as e:
                    logger.error(f"Failed to recreate Stripe customer: {e}")
                    return jsonify({"error": "Payment system error"}), 500

        try:
            session = stripe.checkout.Session.create(
                customer=customer_id,
                mode="subscription",
                line_items=[{"price": price_id, "quantity": 1}],
                success_url=f"{FRONTEND_URL}/success?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{FRONTEND_URL}/cancel",
                allow_promotion_codes=True,
                billing_address_collection="required",
                metadata={
                    "user_id": user_id,
                    "user_email": user["email"],
                    "plan_type": plan_key
                },
                client_reference_id=str(user_id)
            )
            logger.info(f"Checkout session created: {session.id} for user {user_id}")
            log_usage(int(user_id), "subscription_checkout_created", {"plan": plan_key, "session_id": session.id})
            return jsonify({"sessionId": session.id, "currency_used": "INR"}), 200
        except stripe.error.StripeError as e:
            msg = getattr(e, "user_message", "") or str(e)
            logger.error(f"Stripe error creating session: {msg}")
            if isinstance(e, stripe.error.InvalidRequestError):
                return jsonify({"error": f"Invalid payment request: {msg}"}), 400
            elif isinstance(e, stripe.error.AuthenticationError):
                return jsonify({"error": "Payment system authentication failed"}), 500
            elif isinstance(e, stripe.error.APIConnectionError):
                return jsonify({"error": "Payment system temporarily unavailable"}), 503
            else:
                return jsonify({"error": f"Payment processing failed: {msg}"}), 500
    except Exception as e:
        logger.error(f"Subscription creation error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

# Webhook handler
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

# Helper functions for Stripe events

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

# Subscription state application

def apply_subscription_state(customer_id: str, subscription: dict):
    try:
        price_id = subscription["items"]["data"][0]["price"]["id"]
        plan_name = resolve_plan_by_price_id(price_id)
        if not plan_name:
            logger.error(f"Unknown price ID in subscription: {price_id}")
            return
        user = execute_query(
            "SELECT id FROM users WHERE stripe_customer_id = %s",
            (customer_id,),
            fetch_one=True
        )
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
                reset_date
            )
        )
        log_usage(user_id, "subscription_activated", {"plan": plan_name})
        logger.info(f"User {user_id} upgraded to {plan_name}")
    except Exception as e:
        logger.error(f"Error applying subscription state: {e}")
        logger.error(traceback.format_exc())


def downgrade_to_free(customer_id: str):
    try:
        user = execute_query(
            "SELECT id FROM users WHERE stripe_customer_id = %s",
            (customer_id,),
            fetch_one=True
        )
        if not user:
            logger.warning(f"No user found for Stripe customer {customer_id}")
            return
        user_id = user["id"]
        update_user_subscription(user_id, "free", plans["free"]["requests"])
        log_usage(user_id, "subscription_cancelled")
        logger.info(f"User {user_id} downgraded to free plan")
    except Exception as e:
        logger.error(f"Error downgrading user: {e}")

# Summarization endpoint (GET + POST)
import re
import time
import logging
from typing import Dict, List, Tuple, Optional
from flask import jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

# Add these imports to your existing imports
try:
    import textstat
    from rouge_score import rouge_scorer
    TEXTSTAT_AVAILABLE = True
    ROUGE_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    ROUGE_AVAILABLE = False
    logging.warning("textstat and/or rouge-score not available. Install with: pip install textstat rouge-score")



# Enhanced configuration with narrative coherence focus
class NarrativeConfig:
    # Controller loop settings
    CONTROLLER_MAX_TRIES = 4
    
    # Readability targets (balanced for coherence)
    TARGET_FRE_MIN = 80.0  # Slightly lower to allow connecting words
    TARGET_AVG_SENTLEN_MAX = 8.0  # Slightly longer for flow
    TARGET_SYLL_PER_WORD_MAX = 1.4  # Allow some complexity for clarity
    
    # Semantic quality floor
    MIN_ROUGE_L = 0.20  # Higher to ensure meaning preservation
    
    # Narrative coherence requirements
    MIN_CAUSAL_CONNECTIONS = 2  # Minimum cause-effect relationships
    MIN_TEMPORAL_MARKERS = 1    # Words like "first", "then", "because"
    
    # Balance scoring weights (prioritize coherence)
    BALANCE_W_READABILITY = 0.4
    BALANCE_W_SEMANTICS = 0.3
    BALANCE_W_COHERENCE = 0.3  # New coherence weight
    
    # API settings
    SUMMARY_MAX_WORDS = 150  # Slightly longer for narrative flow
    REQUEST_DELAY = 0.5


class NarrativeEvaluator:
    """Enhanced evaluator with narrative coherence metrics"""
    
    def __init__(self):
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        except ImportError:
            self.rouge_scorer = None
        
        # Coherence indicators
        self.causal_words = {
            'because', 'since', 'due to', 'caused by', 'leads to', 'results in',
            'so', 'therefore', 'thus', 'hence', 'as a result'
        }
        
        self.temporal_words = {
            'first', 'then', 'next', 'after', 'before', 'while', 'during',
            'meanwhile', 'later', 'finally', 'now', 'today', 'yesterday'
        }
        
        self.contrast_words = {
            'but', 'however', 'although', 'despite', 'while', 'yet',
            'on the other hand', 'in contrast', 'nevertheless'
        }
        
        self.narrative_connectors = self.causal_words | self.temporal_words | self.contrast_words
    
    def calculate_narrative_coherence(self, text: str) -> Dict[str, float]:
        """Calculate narrative coherence metrics"""
        if not text or not text.strip():
            return {
                'coherence_score': 0.0,
                'causal_connections': 0,
                'temporal_markers': 0,
                'contrast_markers': 0,
                'connector_ratio': 0.0,
                'sentence_flow_score': 0.0
            }
        
        text_lower = text.lower()
        words = text_lower.split()
        sentences = self.split_sentences(text)
        
        # Count different types of connectors
        causal_count = sum(1 for word in self.causal_words if word in text_lower)
        temporal_count = sum(1 for word in self.temporal_words if word in text_lower)
        contrast_count = sum(1 for word in self.contrast_words if word in text_lower)
        
        total_connectors = causal_count + temporal_count + contrast_count
        connector_ratio = total_connectors / len(words) if words else 0
        
        # Sentence flow score (simple heuristic)
        flow_score = self._calculate_sentence_flow(sentences)
        
        # Overall coherence score
        coherence = min(100, (
            (causal_count * 20) +           # Reward cause-effect relationships
            (temporal_count * 15) +         # Reward temporal sequence
            (contrast_count * 10) +         # Reward contrasts/comparisons
            (flow_score * 30) +             # Reward smooth transitions
            (connector_ratio * 500)         # Reward overall connector usage
        ))
        
        return {
            'coherence_score': coherence,
            'causal_connections': causal_count,
            'temporal_markers': temporal_count,
            'contrast_markers': contrast_count,
            'connector_ratio': connector_ratio,
            'sentence_flow_score': flow_score
        }
    
    def _calculate_sentence_flow(self, sentences: List[str]) -> float:
        """Calculate how well sentences flow together"""
        if len(sentences) < 2:
            return 0.0
        
        flow_score = 0.0
        for i in range(len(sentences) - 1):
            current = sentences[i].lower().strip()
            next_sent = sentences[i + 1].lower().strip()
            
            # Check for pronouns referring to previous sentence subjects
            if any(word in next_sent.split()[:3] for word in ['he', 'she', 'it', 'they', 'this', 'that']):
                flow_score += 0.5
            
            # Check for topic continuity (shared keywords)
            current_words = set(current.split())
            next_words = set(next_sent.split()[:5])  # First few words of next sentence
            if current_words & next_words:
                flow_score += 0.3
        
        return min(1.0, flow_score / (len(sentences) - 1))
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Lightweight sentence splitting"""
        if not text or not text.strip():
            return []
        parts = [s.strip() for s in re.split(r'[.!?]+\s+', text) if s.strip()]
        return parts if parts else [text.strip()]
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics (from previous implementation)"""
        # ... (use the same implementation as before)
        if not text or not text.strip():
            return {
                'flesch_reading_ease': 0.0,
                'avg_sentence_length': 0.0,
                'syllable_density': 0.0,
                'short_word_ratio': 0.0,
                'total_words': 0,
                'total_sentences': 0,
            }
        
        words = text.split()
        sentences = self.split_sentences(text)
        total_words = len(words)
        total_sentences = max(1, len(sentences))
        
        # Simple syllable counting
        syllables = sum(self._count_word_syllables(word) for word in words)
        
        # Calculate FRE
        if total_words == 0:
            fre = 0.0
        else:
            avg_sentence_length = total_words / total_sentences
            avg_syllables_per_word = syllables / total_words
            fre = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        avg_sentence_length = total_words / total_sentences
        syllable_density = syllables / total_words if total_words > 0 else 0.0
        
        # Count short words
        short_words = sum(1 for word in words if self._count_word_syllables(word) <= 2)
        short_word_ratio = short_words / total_words if total_words > 0 else 0.0
        
        return {
            'flesch_reading_ease': fre,
            'avg_sentence_length': avg_sentence_length,
            'syllable_density': syllable_density,
            'short_word_ratio': short_word_ratio,
            'total_words': total_words,
            'total_sentences': total_sentences,
        }
    
    def _count_word_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower().strip()
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def calculate_rouge_l(self, summary: str, original: str) -> float:
        """Calculate ROUGE-L score"""
        if not self.rouge_scorer or not summary or not original:
            return 0.0
        try:
            score = self.rouge_scorer.score(original, summary)
            return score["rougeL"].fmeasure
        except:
            return 0.0


def meets_narrative_targets(readability: Dict[str, float], coherence: Dict[str, float], rouge_l: float) -> bool:
    """Check if summary meets narrative coherence + readability targets"""
    return (
        readability['flesch_reading_ease'] >= NarrativeConfig.TARGET_FRE_MIN and
        readability['avg_sentence_length'] <= NarrativeConfig.TARGET_AVG_SENTLEN_MAX and
        readability['syllable_density'] <= NarrativeConfig.TARGET_SYLL_PER_WORD_MAX and
        rouge_l >= NarrativeConfig.MIN_ROUGE_L and
        coherence['causal_connections'] >= NarrativeConfig.MIN_CAUSAL_CONNECTIONS and
        coherence['temporal_markers'] >= NarrativeConfig.MIN_TEMPORAL_MARKERS
    )


def calculate_narrative_balance_score(readability: Dict[str, float], coherence: Dict[str, float], rouge_l: float) -> float:
    """Calculate balanced score including narrative coherence"""
    fre = readability['flesch_reading_ease']
    semantics_scaled = 100.0 * rouge_l
    coherence_score = coherence['coherence_score']
    
    return (
        NarrativeConfig.BALANCE_W_READABILITY * fre + 
        NarrativeConfig.BALANCE_W_SEMANTICS * semantics_scaled +
        NarrativeConfig.BALANCE_W_COHERENCE * coherence_score
    )


def create_narrative_prompt(text: str, attempt: int = 0) -> str:
    """Create prompts that emphasize narrative flow and coherence"""
    
    base_instructions = [
        # Attempt 0: Narrative-focused approach
        (
            "Create a clear story summary that flows like a simple news report:\n"
            "- Tell what happened in order (first, then, finally)\n"
            "- Explain why things happened (because, so, due to)\n"
            "- Use short sentences (5-8 words each)\n"
            "- Connect ideas with simple words (but, and, so)\n"
            "- Keep the main story clear\n\n"
        ),
        # Attempt 1: Cause-effect focused
        (
            "Write a simple story that shows cause and effect:\n"
            "- Start with the main event\n"
            "- Explain what caused it (because, since)\n"
            "- Show what happened as a result (so, therefore)\n"
            "- Use connecting words between sentences\n"
            "- Keep sentences short but connected\n\n"
        ),
        # Attempt 2: Timeline approach
        (
            "Tell this story as a simple timeline:\n"
            "- First, explain the situation\n"
            "- Then, describe what Trump plans\n"
            "- Next, show the opposing view\n"
            "- Finally, give the real facts\n"
            "- Use timeline words (first, then, but, actually)\n\n"
        ),
        # Attempt 3: Contrast structure
        (
            "Write using a clear contrast pattern:\n"
            "- Trump says one thing, BUT data shows another\n"
            "- He claims problems exist, HOWEVER experts disagree\n"
            "- Use contrast words (but, however, actually, despite)\n"
            "- Keep the main conflict clear\n"
            "- Very simple words only\n\n"
        )
    ]
    
    instruction = base_instructions[min(attempt, len(base_instructions) - 1)]
    return f"{instruction}Article: {text[:1500]}\n\nConnected summary:"


@app.route("/api/summarize", methods=["GET", "POST", "OPTIONS"])
@jwt_required(optional=True)
def summarize():
    """Enhanced summarization endpoint with narrative coherence"""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    try:
        # Get input text (same as before)
        if request.method == "GET":
            input_text = (request.args.get("text") or "").strip()
        else:
            data = request.get_json() or {}
            input_text = (data.get("text") or "").strip()
        
        if not input_text:
            return jsonify({"error": "No text provided for summarization"}), 400
        
        if len(input_text) > 10000:
            input_text = input_text[:10000] + " [TEXT TRUNCATED]"
        
        # User authentication and limits (same as before)
        user_id = get_jwt_identity()
        if user_id:
            user = get_user_by_id(int(user_id))
            if not user:
                return jsonify({"error": "User not found"}), 404
            if user["requests_used"] >= user["request_limit"]:
                return jsonify({
                    "error": "Request limit reached. Upgrade your plan for more requests.",
                    "upgrade_url": f"{FRONTEND_URL}/upgrade"
                }), 402
        
        # Initialize narrative evaluator
        evaluator = NarrativeEvaluator()
        
        # Controller loop with narrative focus
        best_summary = None
        best_score = -float('inf')
        target_hit = False
        
        for attempt in range(NarrativeConfig.CONTROLLER_MAX_TRIES):
            try:
                # Create narrative-focused prompt
                prompt = create_narrative_prompt(input_text, attempt)
                
                # Generate summary
                resp = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,  # Slightly higher for more natural flow
                    max_tokens=NarrativeConfig.SUMMARY_MAX_WORDS
                )
                
                summary = (resp.choices[0].message.content or "").strip()
                if not summary:
                    continue
                
                # Evaluate all dimensions
                readability = evaluator.calculate_readability(summary)
                coherence = evaluator.calculate_narrative_coherence(summary)
                rouge_l = evaluator.calculate_rouge_l(summary, input_text)
                
                # Check if targets are met
                if meets_narrative_targets(readability, coherence, rouge_l):
                    target_hit = True
                    best_summary = summary
                    best_readability = readability
                    best_coherence = coherence
                    best_rouge = rouge_l
                    break
                
                # Calculate narrative balance score
                balance = calculate_narrative_balance_score(readability, coherence, rouge_l)
                if balance > best_score:
                    best_score = balance
                    best_summary = summary
                    best_readability = readability
                    best_coherence = coherence
                    best_rouge = rouge_l
                
                # Add delay between attempts
                if attempt < NarrativeConfig.CONTROLLER_MAX_TRIES - 1:
                    time.sleep(NarrativeConfig.REQUEST_DELAY)
                    
            except Exception as e:
                logger.error(f"Narrative summary attempt {attempt + 1} failed: {e}")
                if attempt == NarrativeConfig.CONTROLLER_MAX_TRIES - 1:
                    break
        
        # Check if we got any summary
        if not best_summary:
            return jsonify({"error": "Failed to generate coherent summary"}), 500
        
        # Update user request count and log usage
        if user_id:
            increment_user_requests(int(user_id))
            log_usage(int(user_id), "summarize_narrative", {
                "text_length": len(input_text),
                "target_hit": target_hit,
                "attempts_used": attempt + 1,
                "coherence_score": best_coherence['coherence_score']
            })
        
        # Calculate remaining requests
        remaining = None
        if user_id:
            user = get_user_by_id(int(user_id))
            remaining = user["request_limit"] - user["requests_used"] if user else None
        
        # Return enhanced response with narrative metrics
        response_data = {
            "summary_text": best_summary,
            "readability_score": round(best_readability['flesch_reading_ease'], 1),
            "avg_sentence_length": round(best_readability['avg_sentence_length'], 1),
            "syllable_density": round(best_readability['syllable_density'], 2),
            "semantic_quality": round(best_rouge * 100, 1),
            "coherence_score": round(best_coherence['coherence_score'], 1),
            "narrative_flow": {
                "causal_connections": best_coherence['causal_connections'],
                "temporal_markers": best_coherence['temporal_markers'],
                "contrast_markers": best_coherence['contrast_markers'],
                "sentence_flow_score": round(best_coherence['sentence_flow_score'], 2)
            },
            "targets_met": target_hit,
            "requests_remaining": remaining,
            "quality_metrics": {
                "total_words": best_readability['total_words'],
                "total_sentences": best_readability['total_sentences'],
                "short_word_ratio": round(best_readability['short_word_ratio'], 2),
                "connector_ratio": round(best_coherence['connector_ratio'], 3)
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Narrative summarization error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


# Example of how the improved system would handle your Trump article:
def demonstrate_narrative_improvement():
    """Example showing the difference between approaches"""
    
    original_fragmented = """
    1. Trump plans a news talk.
    2. He says it stops crime.
    3. He gives no details.
    4. A group plans a protest.
    5. There is no crime crisis.
    6. Many people sleep outside.
    7. Some use shelters or housing.
    8. Crime is lower than before.
    9. The mayor says crime is down.
    """
    
    improved_narrative = """
    Trump plans a news conference on Monday. He says it will stop crime in Washington DC. However, he gives no details about how.
    
    A protest group will meet at the same time. They disagree with Trump's plans.
    
    But data shows Trump is wrong about crime. DC crime is actually down 35% from last year. This is the lowest level in 30 years.
    
    The mayor says crime keeps falling. She says there is no crime crisis in DC.
    """
    
    return {
        "original_problems": [
            "No connecting words",
            "Facts seem random", 
            "No cause-effect relationships",
            "Hard to follow the story"
        ],
        "narrative_improvements": [
            "Uses connecting words (however, but, actually)",
            "Shows the conflict between claims and reality", 
            "Groups related information together",
            "Tells a coherent story with clear flow"
        ]
    }

# TTS endpoint
@app.post("/api/synthesize")
@jwt_required()
def synthesize():
    try:
        user_id = get_jwt_identity()
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
        MAX_TEXT_LENGTH = 1000

        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        try:
            # Try new SDK pattern first
            client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            stream = client.text_to_speech.convert(
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
            audio_bytes = b"".join(stream)
            
            log_usage(int(user_id), "tts", {"text_length": len(text)})
            return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", as_attachment=False)
            
        except Exception:
            # Legacy fallback
            try:
                elevenlabs.set_api_key(ELEVENLABS_API_KEY)
                audio_bytes = elevenlabs.generate(
                    text=text,
                    voice=VOICES[DEFAULT_VOICE],
                    model="eleven_turbo_v2_5"
                )
                
                log_usage(int(user_id), "tts", {"text_length": len(text)})
                return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", as_attachment=False)
                
            except Exception as e:
                logger.error(f"TTS fallback error: {e}")
                return jsonify({"error": "Voice generation failed"}), 500

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({"error": "Voice generation failed"}), 500

# Mind Map Storage
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
            """
            INSERT INTO mindmaps (user_id, title, nodes_data, edges_data)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (int(user_id), title, json.dumps(nodes), json.dumps(edges)),
            fetch_one=True
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
            FROM mindmaps 
            WHERE user_id = %s 
            ORDER BY updated_at DESC
            """,
            (int(user_id),),
            fetch=True
        )
        
        return jsonify({
            "mindmaps": [
                {
                    "id": mm["id"],
                    "title": mm["title"],
                    "created_at": mm["created_at"].isoformat(),
                    "updated_at": mm["updated_at"].isoformat()
                }
                for mm in mindmaps
            ]
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
            """
            SELECT nodes_data, edges_data, title
            FROM mindmaps 
            WHERE id = %s AND user_id = %s
            """,
            (mindmap_id, int(user_id)),
            fetch_one=True
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

# Related concepts
graph = nx.DiGraph()
CONCEPTNET_API = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/RelatedTo&limit=20"
DBPEDIA_API = "http://dbpedia.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

concept_relations = {}

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

@app.post("/api/related-concepts")
@jwt_required()
def related_concepts():
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        concept = (data.get("concept") or "").strip()
        
        if not concept:
            return jsonify({"error": "No concept provided"}), 400
            
        related = expand_concept_dataset(concept)[:10]
        log_usage(int(user_id), "related_concepts", {"concept": concept})
        
        return jsonify({"concept": concept, "related_concepts": related}), 200
        
    except Exception as e:
        logger.error(f"Related concepts error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.get("/api/mindmap-graph")
def get_mindmap_graph():
    try:
        nodes = [{"id": node} for node in graph.nodes]
        edges = [{"source": s, "target": t} for s, t in graph.edges]
        return jsonify({"nodes": nodes, "edges": edges}), 200
    except Exception as e:
        logger.error(f"Mindmap graph error: {e}")
        return jsonify({"error": "Internal server error"}), 500

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

# Analytics endpoint
@app.get("/api/analytics")
@jwt_required()
def get_analytics():
    try:
        user_id = get_jwt_identity()
        
        # Get usage stats for the user
        stats = execute_query(
            """
            SELECT 
                action_type,
                COUNT(*) as count,
                DATE(created_at) as date
            FROM usage_analytics 
            WHERE user_id = %s 
                AND created_at >= NOW() - INTERVAL '30 days'
            GROUP BY action_type, DATE(created_at)
            ORDER BY date DESC
            """,
            (int(user_id),),
            fetch=True
        )
        
        return jsonify({"usage_stats": [
            {
                "action": stat["action_type"],
                "count": stat["count"],
                "date": stat["date"].isoformat()
            }
            for stat in stats
        ]}), 200
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({"error": "Failed to fetch analytics"}), 500

# Environment Validation
def validate_environment() -> bool:
    """Validate required environment variables"""
    required = {
        "DATABASE_URL": DATABASE_URL,
        "STRIPE_SECRET_KEY": STRIPE_SECRET_KEY,
        "PRICE_ID_STUDENT_INR": plans["Student"]["price_ids"].get("USD"),
        "PRICE_ID_PRO_INR": plans["Pro"]["price_ids"].get("USD"),
        "FRONTEND_URL": FRONTEND_URL,
        "JWT_SECRET": os.getenv("JWT_SECRET"),
        "OPENAI_API_KEY": OPENAI_API_KEY
    }
    
    missing = [k for k, v in required.items() if not v]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        return False
    
    # Test database connection
    try:
        execute_query("SELECT 1", fetch_one=True)
        logger.info("Database connection validated")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
    
    # Test Stripe API
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

# -----------------------
# Application Startup
# -----------------------
if __name__ in ("__main__", "app"):
    try:
        if not validate_environment():
            logger.error("Environment validation failed. Please check your configuration.")
            exit(1)
        
        # Initialize database tables
        init_database()
        logger.info("Database initialized successfully")
        
        logger.info("LexiSmart backend started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        logger.error(traceback.format_exc())
        exit(1)

# -----------------------
# Local Development
# -----------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)