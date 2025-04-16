# Spam Email Detection

A machine learning model to classify emails as spam or ham (non-spam) using Naive Bayes classification.

## Features
- TF-IDF vectorization of email content
- Keyword-based feature extraction
- Link detection
- Naive Bayes classification

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage
1. Place your email dataset in the `data` directory
2. Run the model:
```bash
python spam_detector.py
```

## Data Format
The model expects a CSV file with the following columns:
- subject: Email subject line
- body: Email content
- label: 'spam' or 'ham'
- ![Screenshot 2025-04-16 092729](https://github.com/user-attachments/assets/a8deaef7-b50b-494c-8750-8960be7a9095)
- 
![Screenshot 2025-04-16 093130](https://github.com/user-attachments/assets/4510450d-df7f-4452-a2fa-66a946232879)
