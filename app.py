from flask import Flask, render_template, request
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import session
from datetime import datetime
from fpdf import FPDF
from flask import send_file
import io

# Setup
nltk.download("punkt")
nltk.download("stopwords")

# Tools
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")

# Load trained model
model = load_model("fine_tuned_spam_detection_model.keras")

def clean_text_for_pdf(text):
    emoji_pattern = re.compile("["                 
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\u2022"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r"[“”‘’]", "'", text)
    text = text.encode('latin-1', 'ignore').decode('latin-1')  # NEW fix
    return text

# Load tokenizer
try:
    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
except FileNotFoundError:
    print("Error: tokenizer.pkl not found. Please generate it first.")
    exit()

def analyze_email_features(text):
    reasons = []

    # Check number of links
    link_count = len(re.findall(r'https?://|www\.', text))
    if link_count >= 2:
        reasons.append(f"Contains {link_count} links")

    # Spammy words
    spammy_keywords = ["free", "winner", "limited offer", "act now", "prize", "guarantee", "urgent", "click here"]
    matches = [word for word in spammy_keywords if word in text.lower()]
    if matches:
        reasons.append(f"Spam keywords: {', '.join(matches)}")

    # Excessive uppercase words
    upper_words = sum(1 for word in text.split() if word.isupper() and len(word) > 3)
    if upper_words >= 3:
        reasons.append("Excessive uppercase words")

    # Marketing tone indicators
    if "unsubscribe" in text.lower() or "order now" in text.lower():
        reasons.append("Marketing/Promotional tone detected")

    if not reasons:
        reasons.append("No strong spam signals detected")

    return reasons

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = word_tokenize(text)
    words = [ps.stem(w) for w in words if w not in stop_words]
    cleaned_text = " ".join(words)

    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=500)
    return padded

def get_flagged_tokens(text):
    text_lower = text.lower()
    flagged = {
        "links": re.findall(r'https?://\S+|www\.\S+', text),
        "spam_keywords": [kw for kw in [
            "free", "winner", "limited offer", "act now", "prize", "guarantee", "urgent", "click here"
        ] if kw in text_lower],
        "shouting_words": [word for word in text.split() if word.isupper() and len(word) >= 4]
    }
    return flagged

def classify_email_category(text):
    lowered = text.lower()

    if any(x in lowered for x in ["dear friend", "let’s catch up", "how are you", "family", "birthday"]):
        return "Personal"
    elif any(x in lowered for x in ["invoice", "meeting", "project", "client", "deadline", "@company.com"]):
        return "Work"
    elif any(x in lowered for x in ["offer", "deal", "discount", "limited time", "buy now", "promo"]):
        return "Promotional"
    elif any(x in lowered for x in ["account", "bank", "login", "security", "verify", "payment", "@bank", "@secure"]):
        return "Security"
    elif any(x in lowered for x in ["newsletter", "update", "subscribe", "edition", "monthly"]):
        return "Newsletter"
    else:
        return "Unknown"

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    reasons = []
    flagged_tokens = {"links": [], "spam_keywords": [], "shouting_words": []}
    history = session.get("scan_history", [])
    dashboard = {}
    category = ""

    if request.method == "POST":
        full_email = request.form["email_text"]
        processed_text = preprocess_text(full_email)
        prediction = float(model.predict(processed_text)[0][0])
        print(f"⚡ Prediction score: {prediction:.4f}")

        score_percentage = prediction * 100

        # Classification
        if prediction >= 0.7:
            result = f"Spam (Score: {score_percentage:.2f}%)"
        elif prediction > 0.3:
            result = f"Likely Spam (Score: {score_percentage:.2f}%)"
        else:
            result = f"Clean Email (Score: {score_percentage:.2f}%)"

        # Reason extraction
        reasons = analyze_email_features(full_email)
        category = classify_email_category(full_email)

        # Flag tokens
        flagged_tokens = get_flagged_tokens(full_email)

        # Add to scan history
        if "scan_history" not in session:
            session["scan_history"] = []

        session["scan_history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result": result,
            "score": round(float(score_percentage), 2),
            "preview": full_email[:100] + ("..." if len(full_email) > 100 else ""),
            "flagged": flagged_tokens
        })

        session["scan_history"] = session["scan_history"][-10:]
        history = session["scan_history"]

        # Dashboard stats
        from collections import Counter
        total_scans = len(history)
        spam_count = sum(1 for h in history if "Spam" in h["result"])
        clean_count = total_scans - spam_count

        top_links, top_keywords, top_uppercase = [], [], []
        for h in history:
            flagged = h.get("flagged", {})
            top_links.extend(flagged.get("links", []))
            top_keywords.extend(flagged.get("spam_keywords", []))
            top_uppercase.extend(flagged.get("shouting_words", []))

        dashboard = {
            "total_scans": total_scans,
            "spam_percentage": round((spam_count / total_scans) * 100, 2) if total_scans else 0,
            "clean_percentage": round((clean_count / total_scans) * 100, 2) if total_scans else 0,
            "top_links": Counter(top_links).most_common(5),
            "top_keywords": Counter(top_keywords).most_common(5),
            "top_uppercase": Counter(top_uppercase).most_common(5)
        }

    return render_template("index.html", result=result, reasons=reasons, flagged=flagged_tokens, history=history, dashboard=dashboard, category=category)

@app.route("/download-report", methods=["POST"])
def download_report():
    data = request.form
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Email Spam Detection Report", ln=True, align='C')
    pdf.ln(10)

    # Metadata
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Scanned At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, f"Spam Score: {data.get('score', 'N/A')}%", ln=True)
    pdf.cell(200, 10, f"Category: {clean_text_for_pdf(data.get('category', 'N/A'))}", ln=True)
    pdf.ln(5)

    # Reasons
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Flagged Reasons:", ln=True)
    pdf.set_font("Arial", size=12)
    reasons_raw = data.get("reasons", "")
    reasons = reasons_raw.split("|") if reasons_raw else []
    for reason in reasons:
        pdf.multi_cell(0, 10, f"- {clean_text_for_pdf(reason)}")

    pdf.ln(5)

    # Tokens
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Flagged Tokens:", ln=True)
    pdf.set_font("Arial", size=12)
    for key in ['links', 'keywords', 'uppercase']:
        tokens = data.get(key, "")
        if tokens:
            label = {
                'links': 'Links',
                'keywords': 'Spam Keywords',
                'uppercase': 'Uppercase Words'
            }[key]
            pdf.multi_cell(0, 10, f"{label}: {clean_text_for_pdf(tokens)}")

    # Full Email
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Full Email Content:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, clean_text_for_pdf(data.get("email", "")))

    # Encode and stream
    pdf_data = bytes(pdf.output(dest='S'))
    pdf_bytes = io.BytesIO(pdf_data)
    pdf_bytes.seek(0)

    return send_file(pdf_bytes, as_attachment=True, download_name="spam_report.pdf")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
