
import pickle
from score import score
import requests
import subprocess
import os

# Test Cases
MESSAGES = [
    r"Just a friendly reminder that the deadline for our current project is next Friday, March 24th. Please make sure all your work is submitted by 5:00 PM on that day.",
    r"This is to confirm our meeting scheduled for tomorrow, March 13th, at 10:00 AM in the conference room. We'll be discussing the new marketing strategy.",
    r"Welcome to this week's edition of our newsletter! Inside, you'll find updates on recent company achievements, upcoming events, and employee spotlights.",
    r"We value your feedback! Please take a moment to complete our customer service survey and let us know how we're doing.",
    r"Please note that our offices will be closed on April 7th in observance of Good Friday. Regular business hours will resume on Monday, April 10th.",
    r"We appreciate your recent purchase and hope you're satisfied with your new product. If you have any questions or concerns, please don't hesitate to contact our support team.",
    r"Join us for an informative webinar on the latest trends in digital marketing, scheduled for March 20th at 2:00 PM. Register now to secure your spot!",
    r"Attached is the monthly performance report for your review. Please take a look and let me know if you have any questions or require further clarification.",
    r"Mark your calendars for our upcoming Staff Appreciation Day on April 14th! We have a fun-filled day planned to celebrate our amazing team.",
    r"Please be advised that our IT department will be performing system maintenance this Saturday, March 18th, from 8:00 AM to 12:00 PM. During this time, access to certain online services may be temporarily unavailable.",
    r"Congratulations! You've been selected to receive a free vacation to the Bahamas. Click here to claim your prize now!",
    r"Limited time offer! Get 50% off on all electronics. Hurry, sale ends soon. Visit our website to shop now.",
    r"You have won $1,000,000 in our lottery! To claim your prize, please send us your personal details and bank information.",
    r"Attention! Your computer has been infected with a virus. Download our antivirus software immediately to protect your device.",
    r"Exclusive deal for you! Buy one, get one free on all our products. Don't miss out on this amazing offer.",
    r"Warning! Your account will be suspended if you do not update your payment information within 24 hours. Click here to update now.",
    r"You're invited to join our exclusive investment club. Invest with us and earn guaranteed high returns. Contact us for more details.",
    r"Flash sale! All items must go. Up to 70% off on selected products. Shop now before it's too late.",
    r"Your email has been randomly selected to receive a $500 gift card. Click here to claim your reward.",
    r"Special announcement! We're giving away free samples of our new product. Sign up now to receive yours.",
]

# function to load the pre-trained model
def load_test_model():
    """Loads a pre-trained test model from a joblib file.
    
    Returns:
        object: The pre-trained model
    """
    filename = 'C:\Users\Nirjhar Nath\OneDrive\Desktop\AML2025-Assignment3/model/best_model.pkl'
    return pickle.load(open(filename, 'rb'))

# function to test the score function
def test_score():
    '''
    Function to test the score function

    This function tests the score function by loading a pre-trained model and running a smoke test, a format test, and a value test.

    Parameters: None

    Returns: None
    '''
    model = load_test_model()

    for message in MESSAGES:
        # Smoke test: Does it run without crashing?
        prediction, propensity = score(message, model, 0.5)
        assert prediction is not None
        assert propensity is not None
        # Format test: Are outputs in expected format?
        assert isinstance(prediction, int)
        assert isinstance(propensity, float)
        # Value tests
        assert 0 <= propensity <= 1  # Propensity should be between 0 and 1
        assert prediction in [0, 1]  # Prediction should be 0 or 1
        # Threshold boundary tests
        assert score(message, model, 0.0)[0] == 1  # Threshold 0 -> always 1
        assert score(message, model, 1.0)[0] == 0  # Threshold 1 -> always 0
        
    # Obvious spam and non-spam cases
    assert score("Congratulations! You won $1000!!!", model, 0.5)[0] == 1  # Likely spam
    assert score("You have won $1,000,000 in our lottery! To claim your prize, please send us your personal details and bank information.", model, 0.5)[0] == 1  # Likely spam
    assert score("Hello, how are you today?", model, 0.5)[0] == 0  # Likely non-spam
    assert score("This is to confirm our meeting scheduled for tomorrow, March 13th, at 10:00 AM in the conference room. We'll be discussing the new marketing strategy.", model, 0.5)[0] == 0  # Likely non-spam


def test_flask():
    """
    Integration test for the Flask application.

    This test starts the Flask app in a separate process, sends requests to the app,
    and checks the responses to ensure that the app is working correctly.
    """

    # Start the Flask app in a separate process
    process = subprocess.Popen(["python", "app.py"])
    
    # Test the /home endpoint
    response = requests.get("http://127.0.0.1:5000/")
    assert response.status_code == 200
    assert "Spam Classifier" in response.text

    # Test the /score endpoint
    response = requests.post("http://127.0.0.1:5000/score", data={"text": "Test message"})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data and "propensity" in data
    assert isinstance(data["prediction"], bool)
    assert isinstance(data["propensity"], float)
    assert 0 <= data["propensity"] <= 1
    assert data["prediction"] in [0, 1]
    
    # Stop the Flask app
    process.kill()
