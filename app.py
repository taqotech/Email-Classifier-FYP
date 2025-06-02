# Import Flask for creating the web server
from flask import Flask, request, jsonify

# Import joblib for loading the trained model
import joblib

# Initialize Flask application
app = Flask(__name__, static_folder='static')

# Load the trained pipeline (model + TF-IDF vectorizer)
pipeline = joblib.load('spam_classifier_pipeline.pkl')

# Define route for serving the index.html file
@app.route('/')
def home():
    # Return the index.html file from the static folder
    return app.send_static_file('index.html')

# Define route for handling prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Extract JSON data from the request
    data = request.json
    # Get the email text from the request, default to empty string if not provided
    email_text = data.get('email', '')
    
    # Check if email text is empty
    if not email_text.strip():
        # Return error response if input is empty
        return jsonify({'error': 'Email text cannot be empty'}), 400
    
    # Make prediction using the pipeline
    prediction = pipeline.predict([email_text])[0]
    # Get prediction probabilities
    probabilities = pipeline.predict_proba([email_text])[0]
    
    # Prepare response dictionary
    result = {
        'prediction': 'Spam' if prediction == 1 else 'Not Spam',
        'probability_spam': float(probabilities[1]),
        'probability_not_spam': float(probabilities[0])
    }
    
    # Return JSON response with prediction results
    return jsonify(result)

# Run the Flask app if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)