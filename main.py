from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__)

# Load datasets
sym_des = pd.read_csv("datasets/symptoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")
training_data = pd.read_csv('datasets/minimal_symptoms.csv')
severity_data = pd.read_csv('datasets/minimal_severity.csv')

# Load model
svc = pickle.load(open('models/svc.pkl','rb'))

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Create symptoms dictionary
all_symptoms = []
for col in training_data.columns[1:]:
    all_symptoms.extend(training_data[col].unique())
symptoms_dict = {symptom: idx for idx, symptom in enumerate(set(all_symptoms))}

# Create severity dictionary
severity_dict = dict(zip(severity_data.iloc[:, 0], severity_data.iloc[:, 1]))

def calculate_severity_score(symptoms):
    """Calculate severity score based on symptoms"""
    score = 0
    for symptom in symptoms:
        if symptom in severity_dict:
            score += severity_dict[symptom]
    return score

def get_severity_level(score):
    """Determine severity level based on score"""
    if score <= 3:
        return "Low"
    elif score <= 6:
        return "Moderate"
    else:
        return "High"

def get_predicted_value(user_symptoms):
    """Match user symptoms with diseases"""
    try:
        # Convert user symptoms to lowercase for matching
        user_symptoms = [s.strip().lower() for s in user_symptoms]
        
        # Create a dictionary to store disease matches
        disease_matches = {}
        
        # Iterate through each row in training data
        for _, row in training_data.iterrows():
            disease = row['Disease']
            disease_symptoms = [
                str(row['Symptom_1']).lower(),
                str(row['Symptom_2']).lower(),
                str(row['Symptom_3']).lower(),
                str(row['Symptom_4']).lower()
            ]
            
            # Count matching symptoms
            matches = len(set(user_symptoms) & set(disease_symptoms))
            if matches > 0:
                disease_matches[disease] = matches
        
        # Get disease with most matching symptoms
        if disease_matches:
            predicted_disease = max(disease_matches.items(), key=lambda x: x[1])[0]
            return predicted_disease
        
        return None

    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def helper(dis):
    """Get disease information"""
    try:
        # Get description - case insensitive matching
        dis_des = description[description['Disease'].str.lower() == dis.lower()]['Description'].iloc[0] \
            if not description[description['Disease'].str.lower() == dis.lower()].empty \
            else "No description available"
        
        # Get precautions
        prec_df = precautions[precautions['Disease'] == dis]
        my_precautions = []
        if not prec_df.empty:
            # Get all precautions from columns Precaution_1 to Precaution_4
            for i in range(1, 5):
                prec = prec_df[f'Precaution_{i}'].iloc[0]
                if isinstance(prec, str) and prec.strip():
                    my_precautions.append(prec.strip().capitalize())
        
        # Get medications
        med_df = medications[medications['Disease'] == dis]
        if not med_df.empty:
            meds_str = med_df['Medication'].iloc[0]
            if isinstance(meds_str, str):
                # Split by semicolon and clean each medication
                meds = [med.strip().capitalize() for med in meds_str.split(';') if med.strip()]
            else:
                meds = ["Consult a doctor for appropriate medications"]
        else:
            meds = ["Consult a doctor for appropriate medications"]
        
        # Get diet recommendations
        diet_df = diets[diets['Disease'] == dis]
        if not diet_df.empty:
            diet_str = diet_df['Diet'].iloc[0]
            if isinstance(diet_str, str):
                # Split by semicolon and clean each diet item
                diet = [d.strip().capitalize() for d in diet_str.split(';') if d.strip()]
            else:
                diet = ["Follow a balanced diet as recommended by your healthcare provider"]
        else:
            diet = ["Follow a balanced diet as recommended by your healthcare provider"]
        
        # Get workout
        workout_df = workout[workout['disease'] == dis]
        if not workout_df.empty:
            work = [w.strip().capitalize() for w in workout_df['workout'].tolist() if w.strip()]
        else:
            work = ["Consult your healthcare provider for appropriate exercise recommendations"]
        
        return dis_des, my_precautions, meds, diet, work

    except Exception as e:
        print(f"Helper function detailed error:", e)
        return ("No description available", 
                ["Consult a healthcare provider"], 
                ["Consult a doctor for medications"], 
                ["Follow a balanced diet"], 
                ["Consult for exercise recommendations"])

def generate_gemini_response(user_input, symptoms):
    """Generate AI response based on medical symptoms"""
    if not symptoms:
        return "Please describe your specific health symptoms first."
    
    # Convert symptoms list to string
    symptoms_str = ', '.join(symptoms)
    
    prompt = f"""As a medical assistant, help the user understand their health condition based on these symptoms: {symptoms_str}

User question: {user_input}

Guidelines for response:
1. Focus on the specific symptoms mentioned
2. Provide clear, concise medical information
3. Avoid making definitive diagnoses
4. Suggest general precautions if appropriate
5. Recommend consulting a healthcare professional for serious concerns

Please respond in a helpful and informative way."""

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text.strip() or "I apologize, but I need more specific information about your symptoms to provide a helpful response."
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return "I apologize, but I'm having trouble processing your request at the moment. Please try again."

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if not symptoms or symptoms == "Symptoms":
            message = "Please enter valid symptoms"
            return render_template('index.html', message=message)
        
        # Clean symptoms
        user_symptoms = [s.strip().lower() for s in symptoms.split(',')]
        
        # Predict disease
        predicted_disease = get_predicted_value(user_symptoms)
        
        if not predicted_disease:
            return render_template('index.html', message="Unable to predict disease from symptoms")
        
        # Get disease details
        try:
            dis_des, precautions, medications, rec_diet, workout_list = helper(predicted_disease)
            
            # Calculate severity
            severity_score = calculate_severity_score(user_symptoms)
            severity_level = get_severity_level(severity_score)
            
            return render_template('index.html', 
                               predicted_disease=predicted_disease.title(), 
                               dis_des=dis_des,
                               my_precautions=precautions, 
                               medications=medications, 
                               my_diet=rec_diet,
                               workout=workout_list,
                               severity_level=severity_level,
                               severity_score=severity_score)
        
        except Exception as e:
            print(f"Error processing disease details: {e}")
            return render_template('index.html', message="Unable to process disease information")

    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.form.get('message', '')
        symptoms_str = request.form.get('symptoms', '')
        
        # Handle empty symptoms
        if not symptoms_str:
            return jsonify({'response': 'Please enter your symptoms in the main input field first.'})
        
        # Clean and validate symptoms
        symptoms = [sym.strip().lower() for sym in symptoms_str.split(',') if sym.strip()]
        
        # Generate response
        response = generate_gemini_response(user_input, symptoms)
        
        # Return JSON response
        return jsonify({'response': response or 'I apologize, but I could not process your request.'})
        
    except Exception as e:
        print(f"Chat error: {str(e)}")  # Log the error
        return jsonify({'response': 'An error occurred while processing your request. Please try again.'}), 500

# Additional routes
@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

if __name__ == '__main__':
    app.run(debug=True, port=5001)