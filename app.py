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
from typing import Dict, List, Tuple, Optional

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
        maxconn=10,
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
        execute_query("CREATE INDEX IF NOT EXISTS idx_subscriptions_subscription_id ON subscriptions(stripe_subscription_id);")
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
        "https://*.netlify.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]

logger.info(f"Allowed CORS origins: {allowed_origins}")

CORS(
    app,
    resources={r"/api/*": {"origins": allowed_origins}},
    supports_credentials=True
)

# JWT & Stripe Configuration
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET", "fallback_secret_key")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
jwt = JWTManager(app)

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
stripe.api_key = STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# External API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# ElevenLabs SDK availability check
ELEVENLABS_AVAILABLE = False
try:
    from elevenlabs import ElevenLabs, VoiceSettings
    ELEVENLABS_AVAILABLE = True
except ImportError:
    try:
        import elevenlabs
        ELEVENLABS_AVAILABLE = True
    except ImportError:
        logger.warning("ElevenLabs SDK not available; TTS endpoint will be disabled.")

# OpenAI configuration
import openai
openai.api_key = OPENAI_API_KEY

# Subscription plans configuration
plans = {
    "free": {
        "requests": 25,
        "price_ids": {}
    },
    "Student": {
        "requests": 200,
        "price_ids": {
            "USD": os.getenv("PRICE_ID_STUDENT_USD")
        }
    },
    "Pro": {
        "requests": 10000,
        "price_ids": {
            "USD": os.getenv("PRICE_ID_PRO_USD")
        }
    }
}

PLAN_ALIASES = {
    "student": "Student",
    "pro": "Pro",
}

def resolve_plan_key(plan_type: str) -> str:
    """Resolve plan name from user input"""
    if not plan_type:
        return ""
    return PLAN_ALIASES.get(plan_type.strip().lower(), plan_type.strip())

def resolve_plan_by_price_id(price_id: str):
    """Find plan name by Stripe price ID"""
    for name, p in plans.items():
        for _, pid in p.get("price_ids", {}).items():
            if pid and pid == price_id:
                return name
    return None

# User management helper functions
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

# Password hashing utilities
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    return hash_password(password) == password_hash

# -----------------------
# Health & Status Endpoints
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
        "elevenlabs_available": ELEVENLABS_AVAILABLE,
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "allowed_origins": allowed_origins
    }), 200

# -----------------------
# Authentication Endpoints
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

        # Create Stripe customer
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

# -----------------------
# Payment Endpoints
# -----------------------
@app.get("/api/payment-health")
def payment_health():
    status = {
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "webhook_configured": bool(STRIPE_WEBHOOK_SECRET),
        "price_ids_configured": {
            "student_usd": bool(plans["Student"]["price_ids"].get("USD")),
            "pro_usd": bool(plans["Pro"]["price_ids"].get("USD")),
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
            return jsonify({"error": f"Price ID (USD) not configured for plan {plan_key}"}), 500

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
            # Verify customer exists
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
            return jsonify({"sessionId": session.id, "currency_used": "USD"}), 200
        except stripe.error.StripeError as e:
            msg = getattr(e, "user_message", "") or str(e)
            logger.error(f"Stripe error creating session: {msg}")
            return jsonify({"error": f"Payment processing failed: {msg}"}), 500
    except Exception as e:
        logger.error(f"Subscription creation error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

# -----------------------
# Stripe Webhook Handler
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

# Webhook event handlers
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

# -----------------------
# Summarization with Narrative Coherence
# -----------------------
class NarrativeConfig:
    """Configuration for narrative-focused summarization"""
    CONTROLLER_MAX_TRIES = 4
    TARGET_FRE_MIN = 80.0
    TARGET_AVG_SENTLEN_MAX = 8.0
    TARGET_SYLL_PER_WORD_MAX = 1.4
    MIN_ROUGE_L = 0.20
    MIN_CAUSAL_CONNECTIONS = 2
    MIN_TEMPORAL_MARKERS = 1
    BALANCE_W_READABILITY = 0.4
    BALANCE_W_SEMANTICS = 0.3
    BALANCE_W_COHERENCE = 0.3
    SUMMARY_MAX_WORDS = 150
    REQUEST_DELAY = 0.5

class NarrativeEvaluator:
    """Enhanced evaluator with narrative coherence metrics"""
    
    def __init__(self):
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
        sentences = self.split_sentences