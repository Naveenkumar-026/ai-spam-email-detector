# AI Spam Email Detector

A Flask-based spam email detection project that combines a **trained deep learning model** with **lightweight heuristic analysis** to classify pasted email content as:

- **Clean Email**
- **Likely Spam**
- **Spam**

In addition to prediction, the app highlights suspicious signals such as links, spam keywords, shouting/uppercase words, category labels, session-based scan history, and downloadable PDF reports.

---

## Overview

This repository contains **two sides of the project**:

1. **Inference / Web App**
   - a Flask interface for scanning pasted email content
   - live spam scoring using a saved Keras model
   - indicator extraction, category tagging, session history, and PDF export

2. **Training Pipeline**
   - dataset loading and preprocessing
   - tokenization and padding
   - CNN + BiLSTM-based model training
   - evaluation, model export, and tokenizer generation

This makes the repository useful both as:

- a portfolio project
- a machine learning classification demo
- a cybersecurity / phishing-awareness prototype
- a base for future secure-email analysis work

---

## Features

- Flask web interface for pasting and scanning full email text
- AI-based spam scoring using a saved `.keras` model
- heuristic analysis of suspicious links and spam keywords
- uppercase / shouting detection
- basic email category classification
- recent scan history stored in Flask session
- recent activity dashboard
- downloadable PDF scan report
- training script for rebuilding the model pipeline

---

## Repository structure

```text
ai-spam-email-detector/
├── app.py
├── spam_detection.py
├── save_tokenizer.py
├── setup_nltk.py
├── fine_tuned_spam_detection_model.keras
├── tokenizer.pkl
├── data/
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech stack

- **Python**
- **Flask**
- **TensorFlow / Keras**
- **NLTK**
- **NumPy**
- **FPDF2**

---

## How the web app works

1. Paste email content into the web form
2. The text is cleaned, tokenized, and padded
3. The saved model produces a spam score
4. Heuristic checks extract:
   - suspicious links
   - spam keywords
   - uppercase/shouting words
5. The app assigns a coarse category such as:
   - Personal
   - Work
   - Promotional
   - Security
   - Newsletter
6. The result is stored in recent session history
7. A PDF report can be downloaded for the scan

---

## Installation

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python setup_nltk.py
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python setup_nltk.py
```

> The app also downloads required NLTK resources on startup, but running `setup_nltk.py` once is cleaner and avoids first-run delays.

---

## Run the web app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## Runtime requirements

The web app expects these files to be present in the project root:

- `fine_tuned_spam_detection_model.keras`
- `tokenizer.pkl`

Without them, the app cannot load the trained model/tokenizer pipeline correctly.

---

## Optional environment variables

For safer local deployment, you can set:

### Linux / macOS

```bash
export FLASK_SECRET_KEY="replace-this-with-a-strong-secret"
export PORT=5000
export FLASK_DEBUG=false
python app.py
```

### Windows PowerShell

```powershell
$env:FLASK_SECRET_KEY="replace-this-with-a-strong-secret"
$env:PORT="5000"
$env:FLASK_DEBUG="false"
python app.py
```

---

## Model behavior

The application currently classifies results into three practical output bands:

- **Clean Email**
- **Likely Spam**
- **Spam**

The final score is presented as a percentage, and the UI supplements the prediction with explainable lightweight indicators.

---

## Training pipeline

The training side of the repository lives in:

```text
spam_detection.py
save_tokenizer.py
```

### Training flow

- load email data from directories and CSV datasets
- combine subject, sender, and body into a single text field
- preprocess text using tokenization, stopword removal, and stemming
- tokenize with a vocabulary cap of 5000 words
- pad sequences to length 500
- split into training and testing sets
- apply class weighting
- train a CNN + BiLSTM-style model
- evaluate with accuracy / precision / recall / F1 / confusion matrix
- save the trained model as `fine_tuned_spam_detection_model.keras`
- save the tokenizer as `tokenizer.pkl`

---

## Training data notes

The training script expects a local data setup that includes:

### Directory-based mail corpora

```text
data/
├── easy_ham/
├── hard_ham/
├── spam/
└── spam_2/
```

### Additional CSV datasets

- `real_emails_dataset.csv`
- `realistic_spam_ham_dataset.csv`
- `real_world_ham_dataset.csv`

If you want the repository to be reproducible for others, document or include those datasets separately where licensing allows.

---

## Generate tokenizer after training

If needed, regenerate the tokenizer with:

```bash
python save_tokenizer.py
```

---

## Example use cases

- quick phishing / spam demo for presentations
- academic mini-project in AI + cybersecurity
- email text risk scoring prototype
- base system for a larger anti-phishing platform

---

## Current limitations

- works on pasted email content, not full mailbox ingestion
- does not currently analyze:
  - SPF / DKIM / DMARC
  - full email headers
  - sender infrastructure
  - attachment payloads
  - URL reputation feeds
- recent history is session-based, not database-backed
- explanation logic is lightweight and rule-based alongside the model

---

## Suggested future improvements

- header analysis (SPF, DKIM, Return-Path, Received chain)
- malicious URL reputation checks
- attachment scanning
- persistent database-backed history
- user authentication
- API version of the scanner
- explainable AI output for prediction confidence
- rate limiting and production hardening

---

## Security note

Do not use the default Flask secret key in production. Set your own `FLASK_SECRET_KEY` before deployment.

---

## License

Add your preferred license here.
