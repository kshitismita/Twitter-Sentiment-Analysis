from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load your saved model and vectorizer
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure templates/index.html exists with form

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form.get('tweet', '')
    if not tweet.strip():
        return render_template('index.html', prediction_text="Please enter a tweet to analyze.")
    
    vect = vectorizer.transform([tweet])
    prediction = model.predict(vect)[0]
    labels = {0: "Not Hate Speech", 1: "Hate Speech"}
    result = f"Prediction: {labels.get(prediction, 'Unknown')}"
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
