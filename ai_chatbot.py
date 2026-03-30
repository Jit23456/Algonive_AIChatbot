"""
AI Chatbot — Customer Support Assistant
Algonive Python Programming Internship - Task 2

Features:
  - NLP-based intent detection
  - Predefined FAQ responses
  - AI-generated fallback responses
  - Conversation history / context tracking
  - Sentiment detection
  - API integration (weather example)
"""

import re
import random
import json
import urllib.request
from datetime import datetime


# ─────────────────────────────────────────────────────
# 1.  KNOWLEDGE BASE  (predefined FAQs)
# ─────────────────────────────────────────────────────
FAQ = {
    "greeting": {
        "patterns": ["hello", "hi", "hey", "good morning", "good evening", "howdy", "greetings"],
        "responses": [
            "Hello! Welcome to support. How can I help you today?",
            "Hi there! I'm your virtual assistant. What can I do for you?",
            "Hey! Great to see you. How may I assist you?",
        ],
    },
    "farewell": {
        "patterns": ["bye", "goodbye", "see you", "take care", "exit", "quit", "thanks bye"],
        "responses": [
            "Goodbye! Have a wonderful day!",
            "Take care! Feel free to reach out anytime.",
            "Bye! It was a pleasure assisting you.",
        ],
    },
    "order_status": {
        "patterns": ["where is my order", "order status", "track order", "order tracking",
                     "when will my order arrive", "shipping status", "delivery status"],
        "responses": [
            "To check your order status, please provide your Order ID and I'll look it up for you.",
            "You can track your order by visiting our website's 'Track Order' page with your order number.",
            "Please share your order ID and I'll fetch the latest status for you right away.",
        ],
    },
    "refund": {
        "patterns": ["refund", "money back", "return", "cancel order", "get my money", "return policy"],
        "responses": [
            "Our refund policy allows returns within 30 days of purchase. Please provide your order ID to initiate a refund.",
            "I can help with your refund request! Refunds are processed within 5-7 business days once approved.",
            "To process your return, please share your order ID and reason for return.",
        ],
    },
    "payment": {
        "patterns": ["payment", "pay", "credit card", "debit card", "upi", "net banking",
                     "payment failed", "charge", "bill", "invoice"],
        "responses": [
            "We accept Credit/Debit cards, UPI, Net Banking, and Wallets. Is there a specific payment issue?",
            "For payment issues, please ensure your card details are correct and try again. Need more help?",
            "Your invoice will be emailed within 24 hours of purchase. For payment failures, try a different method.",
        ],
    },
    "hours": {
        "patterns": ["working hours", "support hours", "open", "available", "timing", "when are you open"],
        "responses": [
            "Our support team is available Monday–Friday, 9 AM to 6 PM IST.",
            "We're open from 9 AM to 6 PM on weekdays. For urgent issues, use our 24/7 chat bot!",
        ],
    },
    "contact": {
        "patterns": ["contact", "phone number", "email", "reach you", "talk to agent", "human agent", "speak to someone"],
        "responses": [
            "You can reach us at support@example.com or call +91-9876543210 (Mon-Fri, 9AM-6PM).",
            "To speak with a human agent, call +91-9876543210 or email support@example.com.",
        ],
    },
    "pricing": {
        "patterns": ["price", "cost", "how much", "pricing", "plan", "subscription", "fee", "charges"],
        "responses": [
            "Our pricing plans start at ₹499/month. Visit our website for a full breakdown of features.",
            "We offer Basic (₹499), Pro (₹999), and Enterprise (custom) plans. Which are you interested in?",
        ],
    },
    "thanks": {
        "patterns": ["thank you", "thanks", "thank u", "thx", "appreciated", "helpful"],
        "responses": [
            "You're welcome! Happy to help anytime.",
            "Glad I could assist! Is there anything else you need?",
            "My pleasure! Don't hesitate to ask if you need more help.",
        ],
    },
    "about": {
        "patterns": ["who are you", "what are you", "your name", "are you a bot", "are you human", "about you"],
        "responses": [
            "I'm an AI-powered customer support chatbot built with Python. I'm here to help you 24/7!",
            "I'm a virtual assistant designed to answer your queries quickly and efficiently.",
        ],
    },
}


# ─────────────────────────────────────────────────────
# 2.  NLP ENGINE
# ─────────────────────────────────────────────────────
class NLPEngine:
    """Simple rule-based NLP: tokenize → intent match → sentiment."""

    STOPWORDS = {"i", "me", "my", "the", "a", "an", "is", "are", "was",
                 "do", "can", "could", "would", "please", "just", "want"}

    SENTIMENT = {
        "positive": ["great", "good", "happy", "love", "excellent", "awesome", "wonderful", "perfect"],
        "negative": ["bad", "terrible", "horrible", "angry", "upset", "frustrated", "worst", "hate", "useless"],
        "neutral":  [],
    }

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        return [t for t in tokens if t not in self.STOPWORDS]

    def detect_intent(self, text: str) -> tuple[str, float]:
        """Returns (intent_name, confidence_score)."""
        tokens = set(self.tokenize(text))
        text_lower = text.lower()

        best_intent = "unknown"
        best_score = 0.0

        for intent, data in FAQ.items():
            for pattern in data["patterns"]:
                pattern_words = set(pattern.lower().split())
                # substring match
                if pattern.lower() in text_lower:
                    score = len(pattern_words) / max(len(tokens), 1)
                    score = min(score + 0.5, 1.0)
                    if score > best_score:
                        best_score = score
                        best_intent = intent
                # word overlap
                overlap = len(tokens & pattern_words)
                if overlap > 0:
                    score = overlap / len(pattern_words)
                    if score > best_score:
                        best_score = score
                        best_intent = intent

        return best_intent, round(best_score, 2)

    def detect_sentiment(self, text: str) -> str:
        text_lower = text.lower()
        for word in self.SENTIMENT["positive"]:
            if word in text_lower:
                return "positive"
        for word in self.SENTIMENT["negative"]:
            if word in text_lower:
                return "negative"
        return "neutral"

    def extract_order_id(self, text: str) -> str | None:
        match = re.search(r"\b([A-Z]{2,3}[-]?\d{4,8})\b", text.upper())
        return match.group(1) if match else None


# ─────────────────────────────────────────────────────
# 3.  API INTEGRATION  (weather example)
# ─────────────────────────────────────────────────────
def fetch_weather(city: str = "Mumbai") -> str:
    """Fetch weather using Open-Meteo (free, no API key needed)."""
    coords = {
        "mumbai":   (19.076, 72.877),
        "delhi":    (28.704, 77.102),
        "kolkata":  (22.572, 88.363),
        "bangalore":(12.971, 77.594),
        "chennai":  (13.083, 80.270),
    }
    city_key = city.lower().strip()
    lat, lon = coords.get(city_key, (19.076, 72.877))
    url = (f"https://api.open-meteo.com/v1/forecast?"
           f"latitude={lat}&longitude={lon}"
           f"&current_weather=true")
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        cw = data["current_weather"]
        temp = cw["temperature"]
        wind = cw["windspeed"]
        return f"Current weather in {city.title()}: {temp}°C, wind {wind} km/h."
    except Exception:
        return f"Weather data for {city.title()} is temporarily unavailable."


# ─────────────────────────────────────────────────────
# 4.  CHATBOT CORE
# ─────────────────────────────────────────────────────
class AIChatbot:
    def __init__(self, name: str = "Aria"):
        self.name = name
        self.nlp = NLPEngine()
        self.history: list[dict] = []
        self.context: dict = {}
        self.session_start = datetime.now()

    # ---------- response generation ----------
    def _faq_response(self, intent: str) -> str:
        return random.choice(FAQ[intent]["responses"])

    def _ai_fallback(self, user_input: str) -> str:
        """Rule-based fallback when no intent is matched."""
        lower = user_input.lower()

        # weather query
        for city in ["mumbai", "delhi", "kolkata", "bangalore", "chennai"]:
            if city in lower or "weather" in lower:
                city_name = city if city in lower else "Mumbai"
                return fetch_weather(city_name)

        # order ID detected
        oid = self.nlp.extract_order_id(user_input)
        if oid:
            return (f"I found Order ID **{oid}** in your message. "
                    f"Let me check — your order is currently being processed "
                    f"and will be delivered within 3-5 business days.")

        # question words
        if any(w in lower for w in ["how", "what", "why", "when", "where", "which"]):
            return ("That's a great question! I'm still learning about this topic. "
                    "For detailed help, contact our support team at support@example.com.")

        return ("I'm not sure I understood that. Could you rephrase? "
                "Or type 'help' to see what I can assist with.")

    def _handle_special(self, text: str) -> str | None:
        lower = text.lower().strip()
        if lower in {"help", "menu", "options"}:
            return (
                "Here's what I can help with:\n"
                "  • Order status & tracking\n"
                "  • Refunds & returns\n"
                "  • Payment issues\n"
                "  • Pricing & plans\n"
                "  • Contact & support hours\n"
                "  • Weather info (e.g. 'weather in Mumbai')\n"
                "Just type your question!"
            )
        if lower in {"history", "show history"}:
            if not self.history:
                return "No conversation history yet."
            lines = []
            for h in self.history[-6:]:
                lines.append(f"  You: {h['user']}")
                lines.append(f"  {self.name}: {h['bot']}")
            return "\n".join(lines)
        return None

    def respond(self, user_input: str) -> dict:
        """Main entry: returns response dict with metadata."""
        user_input = user_input.strip()
        if not user_input:
            return {"response": "Please type something!", "intent": "empty", "sentiment": "neutral"}

        # special commands
        special = self._handle_special(user_input)
        if special:
            self._log(user_input, special, "special", "neutral")
            return {"response": special, "intent": "special", "sentiment": "neutral"}

        intent, confidence = self.nlp.detect_intent(user_input)
        sentiment = self.nlp.detect_sentiment(user_input)

        # sentiment-aware prefix
        prefix = ""
        if sentiment == "negative":
            prefix = "I'm sorry to hear that. "
        elif sentiment == "positive":
            prefix = "That's great! "

        if intent != "unknown" and confidence >= 0.3:
            response = prefix + self._faq_response(intent)
        else:
            response = prefix + self._ai_fallback(user_input)

        # farewell → exit flag
        exit_flag = intent == "farewell"

        self._log(user_input, response, intent, sentiment)
        return {
            "response": response,
            "intent": intent,
            "confidence": confidence,
            "sentiment": sentiment,
            "exit": exit_flag,
        }

    def _log(self, user: str, bot: str, intent: str, sentiment: str):
        self.history.append({
            "user": user, "bot": bot,
            "intent": intent, "sentiment": sentiment,
            "time": datetime.now().strftime("%H:%M:%S"),
        })

    def stats(self) -> dict:
        total = len(self.history)
        intents = {}
        for h in self.history:
            intents[h["intent"]] = intents.get(h["intent"], 0) + 1
        duration = (datetime.now() - self.session_start).seconds
        return {"messages": total, "intents": intents, "duration_sec": duration}


# ─────────────────────────────────────────────────────
# 5.  CLI  DEMO
# ─────────────────────────────────────────────────────
def run_demo():
    bot = AIChatbot(name="Aria")

    print("=" * 60)
    print("   AI CHATBOT — ARIA  |  Algonive Task 2")
    print("=" * 60)
    print(f"  {bot.name}: Hello! I'm Aria, your AI support assistant.")
    print(f"  {bot.name}: Type 'help' for options, 'bye' to exit.\n")

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {bot.name}: Goodbye! Have a great day!")
            break

        if not user_input:
            continue

        result = bot.respond(user_input)
        print(f"  {bot.name}: {result['response']}")
        print(f"  [intent={result['intent']} | sentiment={result['sentiment']}]\n")

        if result.get("exit"):
            s = bot.stats()
            print(f"  ── Session Summary ──")
            print(f"  Messages exchanged : {s['messages']}")
            print(f"  Duration           : {s['duration_sec']}s")
            print(f"  Top intents        : {s['intents']}")
            break


if __name__ == "__main__":
    run_demo()