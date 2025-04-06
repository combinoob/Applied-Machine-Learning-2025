# import libraries
from sklearn.base import BaseEstimator as estimator
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set of English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    '''
    Preprocess the input text:
    - Tokenize
    - Remove stopwords
    - Lemmatize
    - Lowercase
    Returns a string (joined tokens).
    '''
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha()]  # Keep only words
    tokens = [token for token in tokens if token.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(tokens)

def score(text: str, model: estimator, threshold: float = 0.5) -> tuple[bool, float]:
    '''
    Predict if a given message is spam or ham using the trained model.
    Args:
        text (str): The message to classify
        model (estimator): Trained model (pipeline with vectorizer)
        threshold (float): Threshold for classification (default 0.5)
    Returns:
        (bool, float): (prediction, propensity)
    '''
    if not isinstance(text, str):
        raise ValueError("Text must be a string")

    if not isinstance(model, estimator):
        raise ValueError("Model must be a scikit-learn estimator")

    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1")

    # Preprocess and predict
    processed = preprocess_text(text)
    propensity = model.predict_proba([processed])[0][1]
    prediction = propensity > threshold

    return bool(prediction), float(propensity)
