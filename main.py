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
        # Get description (working correctly - keep as reference)
        dis_des = description[description['Disease'] == dis]['Description'].iloc[0] if not description[description['Disease'] == dis].empty else ""
        
        # Get precautions - fix format
        prec_df = precautions[precautions['Disease'] == dis]
        my_precautions = []
        if not prec_df.empty:
            # Get all precautions from columns Precaution_1 to Precaution_4
            for i in range(1, 5):
                prec = prec_df[f'Precaution_{i}'].iloc[0]
                if isinstance(prec, str) and prec.strip():
                    my_precautions.append(prec.strip())
        
        # Get medications - fix format
        med_df = medications[medications['Disease'] == dis]
        if not med_df.empty:
            meds = med_df['Medication'].iloc[0]
            if isinstance(meds, str):
                meds = [m.strip() for m in meds.split(';') if m.strip()]
            else:
                meds = []
        else:
            meds = []
        
        # Get diet - fix format
        diet_df = diets[diets['Disease'] == dis]
        if not diet_df.empty:
            diet = diet_df['Diet'].iloc[0]
            if isinstance(diet, str):
                diet = [d.strip() for d in diet.split(';') if d.strip()]
            else:
                diet = []
        else:
            diet = []
        
        # Get workout (working correctly - keep as reference)
        work = workout[workout['disease'] == dis]['workout'].tolist()
        
        return dis_des, my_precautions, meds, diet, work

    except Exception as e:
        print(f"Helper function detailed error:", e)
        return "", [], [], [], []

def generate_gemini_response(user_input, symptoms):
    """Generate AI response based on medical symptoms"""
    if not symptoms:
        return "Please describe your specific health symptoms first."
    
    allowed_symptoms_str = ', '.join(set(all_symptoms))
    
    prompt = f"""Medical Symptom Assistant Guidelines:
    - Only discuss symptoms related to the following: {allowed_symptoms_str}
    - User's symptoms: {', '.join(symptoms)}
    - User's query: {user_input}
    
    Provide a concise, informative medical response focusing strictly on the symptoms and potential health conditions."""

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text.strip() or "I can only discuss medical symptoms based on our predefined list."
    except Exception as e:
        return "Sorry, I can only discuss medical symptoms at the moment."

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
            result = helper(predicted_disease)
            if not result:
                return render_template('index.html', message="No information found for this disease")
            
            dis_des, precautions, medications, rec_diet, workout = result
            
            # Prepare precautions
            my_precautions = []
            for precaution_list in precautions:
                my_precautions.extend(precaution_list)

            # Calculate severity
            severity_score = calculate_severity_score(user_symptoms)
            severity_level = get_severity_level(severity_score)

            return render_template('index.html', 
                                   predicted_disease=predicted_disease, 
                                   dis_des=dis_des,
                                   my_precautions=my_precautions, 
                                   medications=medications, 
                                   my_diet=rec_diet,
                                   workout=workout,
                                   severity_level=severity_level,
                                   severity_score=severity_score)
        
        except Exception as e:
            print(f"Error processing disease details: {e}")
            return render_template('index.html', message="Unable to process disease information")

    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('message', '')
    symptoms = request.form.get('symptoms', '').split(',')
    
    # Clean and validate symptoms
    symptoms = [sym.strip().lower() for sym in symptoms if sym.strip()]
    
    response = generate_gemini_response(user_input, symptoms)
    return jsonify({'response': response})

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