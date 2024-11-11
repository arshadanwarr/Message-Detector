from flask import Flask, request, render_template
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Ensure you have the necessary NLTK resources downloaded (run this once)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stemmer
ps = PorterStemmer()

def transform(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text
    y = []

    # Remove non-alphanumeric tokens
    for i in text:
        if i.isalnum():
            y.append(i)
    
    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Stem the remaining words
    y = [ps.stem(i) for i in y]

    return " ".join(y)  # Return the processed text as a single string

# Load the trained model and vectorizer
model_path = 'model.pkl'
vectorizer_path = 'victorizer.pkl'  # Use the correct filename here

with open(model_path, 'rb') as file:  # Use 'rb' for reading binary files
    model = pickle.load(file)

with open(vectorizer_path, 'rb') as file:  # Use 'rb' for reading binary files
    victorizer = pickle.load(file)  # Fixed variable name

# Specify template folder as 'template'
app = Flask(__name__, template_folder='templates')  # Ensure this matches your structure

@app.route('/')
def home():
    return render_template('index.html')  # Render index.html from the 'template' folder

@app.route('/predict', methods=['POST'])
def predict():
    print("Received POST request")
    
    # Extract data from form
    sms_text = request.form['SMS']  # Get the SMS text from the form
    print("SMS Text:", sms_text)
    
    try:
        # Transform the input text using your transform function
        transformed_text = transform(sms_text)  # Call the transform function
        
        # Transform the input text using the vectorizer
        final_features = victorizer.transform([transformed_text])  # Use transformed text here
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'Spam' if prediction[0] == 1 else 'Not Spam'
        
        print("Prediction:", output)

        return render_template('index.html', prediction_text='Prediction: {}'.format(output))
    
    except Exception as e:
        print("Error during prediction:", e)
        return render_template('index.html', prediction_text='Error during prediction.')

if __name__ == "__main__":
    app.run(debug=True)