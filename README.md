# AI Spam Email Detector

A Flask-based spam email detection application that uses a trained deep learning model to classify pasted email content as **Clean Email** or **Spam**. The app also extracts lightweight indicators such as suspicious links, spam keywords, uppercase/shouting patterns, category labels, scan history, and downloadable PDF reports.

## Features

- Web interface for pasting full email content
- AI-based spam scoring using a saved Keras model
- Lightweight heuristic signal extraction
- Category tagging such as Work, Promotional, and Security
- Scan history stored in Flask session
- PDF report export
- Simple dashboard view for recent scan statistics

## Project Structure

```text
.
├── app.py
├── spam_detection.py
├── save_tokenizer.py
├── setup_nltk.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── requirements.txt
├── .gitignore
└── README.md
```

## Important Files

This project may also include:
- `fine_tuned_spam_detection_model.keras`
- `tokenizer.pkl`

You can publish the repository in one of two ways:

### Option A — code-first public repo
Commit the source code, templates, styling, documentation, and helper scripts only. Exclude the model, tokenizer, and large datasets.

### Option B — reproducible repo
Commit code plus model/tokenizer/datasets only if you have the right to redistribute them. Use Git LFS for large files when needed.

## Recommended Environment

Tested successfully with **Python 3.11**.

## Installation

```bash
python -m venv .venv311
source .venv311/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python setup_nltk.py
```

## Run the App

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## NLTK Setup

The application requires these NLTK resources:
- `stopwords`
- `punkt`
- `punkt_tab`

The included `setup_nltk.py` downloads all required resources.

## Example Test Emails

### 1. Clean email
**Subject:** Meeting rescheduled to Tuesday

**Body:**
Hi Arun,  
The project review has been moved from Monday to Tuesday at 3:00 PM in Lab 2. Please bring the updated presentation and the intrusion detection summary.  
Thanks,  
Naveen

### 2. Spam email
**Subject:** URGENT!!! You won a FREE iPhone — claim NOW

**Body:**
Congratulations!!! You have been selected as the lucky winner of a FREE iPhone 15 Pro Max. Click here immediately to claim your reward before your account is deleted: http://free-gift-claim-now.ru

Do not miss this LIMITED TIME OFFER!!! SEND YOUR DETAILS NOW!!!

### 3. Phishing-style email
**Subject:** Password expiry notice

**Body:**
Dear User,  
Your email account password will expire within 24 hours. To avoid suspension, verify your account immediately using the secure link below:  
http://mail-verification-login-security.com

Failure to verify may result in temporary access loss.  
IT Support Team

## Notes Before Publishing

- Keep `debug=False` for public/demo-ready code.
- Do not commit secrets or local environment values.
- Add a license before making the repository public.
- Confirm you have redistribution rights for every dataset included.
- If you keep the trained model in GitHub, prefer Git LFS.

## Suggested Future Improvements

- Email header analysis (SPF, DKIM, Return-Path, Received chain)
- URL reputation / blacklist checks
- Attachment analysis
- Explainable model decisions
- Persistent database-backed history instead of session-only history
- Authentication and rate limiting
