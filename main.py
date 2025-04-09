from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import google.generativeai as genai
import os
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

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
        
        # Log the user symptoms
        print(f"User symptoms: {user_symptoms}")
        
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
            
            # Log the disease and its symptoms
            print(f"Checking disease: {disease}, Symptoms: {disease_symptoms}")
            
            # Count matching symptoms
            matches = len(set(user_symptoms) & set(disease_symptoms))
            if matches > 0:
                disease_matches[disease] = matches
                print(f"Match found: {disease} with {matches} matching symptoms")
        
        # Get disease with most matching symptoms
        if disease_matches:
            predicted_disease = max(disease_matches.items(), key=lambda x: x[1])[0]
            print(f"Predicted disease: {predicted_disease}")
            return predicted_disease
        
        # If no matches found, use Gemini API to predict
        try:
            symptoms_str = ', '.join(user_symptoms)
            prompt = f"""Based on these symptoms: {symptoms_str}, what could be the possible medical condition?
            Provide only the name of the most likely disease or condition in one word.
            If uncertain, return 'Unknown Condition'."""
            
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            if response and response.text:
                predicted_disease = response.text.strip()
                # Clean up the response
                predicted_disease = predicted_disease.replace('.', '').replace('"', '').strip()
                print(f"Gemini API predicted disease: {predicted_disease}")
                return predicted_disease
            else:
                return "Unknown Condition"
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            return "Unknown Condition"

    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Unknown Condition"

def generate_disease_description(disease_name):
    """Generate detailed disease description using Gemini"""
    prompt = f"""Provide a brief, clear description of {disease_name} in 2-3 sentences. Include:
1. What it is
2. Most common symptoms
3. Basic cause

Keep it simple and concise, around 50-60 words total. Write in paragraph form, not bullet points."""

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Check if response is valid
        if not response or not response.text:
            return f"{disease_name} is a medical condition that affects the body. Common symptoms may vary depending on the severity and type. It's important to consult with a healthcare professional for proper diagnosis and treatment."
        
        # Clean up the response
        description = response.text.strip()
        description = description.replace('•', '').replace('*', '').replace('\n', ' ')
        
        # Ensure description is not empty
        if not description:
            return f"{disease_name} is a medical condition that affects the body. Common symptoms may vary depending on the severity and type. It's important to consult with a healthcare professional for proper diagnosis and treatment."
            
        return description
    except Exception as e:
        print(f"Error generating disease description: {e}")
        return f"{disease_name} is a medical condition that affects the body. Common symptoms may vary depending on the severity and type. It's important to consult with a healthcare professional for proper diagnosis and treatment."

def helper(dis):
    """Get disease information"""
    try:
        print(f"Original disease name: {dis}")
        
        # Always generate description using Gemini for consistency and detail
        dis_des = generate_disease_description(dis)
        
        # Initialize data containers
        my_precautions = []
        meds = []
        diet = []
        work = []
        
        # Create disease name variations for matching
        disease_variations = [
            dis,  # Original
            dis.lower(),  # lowercase
            dis.title(),  # Title Case
            ' '.join(word.capitalize() for word in dis.split()),  # Each Word Capitalized
            dis.replace('-', ' '),  # Replace hyphens with spaces
            dis.replace('(', '').replace(')', '')  # Remove parentheses
        ]
        
        print(f"Trying to match with variations: {disease_variations}")
        
        # -------------------- 1. Get precautions from dataset --------------------
        # Try to match with any of the disease variations
        matched_precautions = False
        for disease_var in disease_variations:
            # Try both exact and case-insensitive matches
            prec_df = precautions[precautions['Disease'] == disease_var]
            if prec_df.empty:
                prec_df = precautions[precautions['Disease'].str.lower() == disease_var.lower()]
            
            if not prec_df.empty:
                matched_precautions = True
                print(f"Matched precautions with: {disease_var}")
                
                # Extract precautions
                for i in range(1, 5):
                    col_name = f'Precaution_{i}'
                    if col_name in prec_df.columns:
                        prec = prec_df[col_name].iloc[0]
                        if isinstance(prec, str) and prec.strip():
                            my_precautions.append(prec.strip().capitalize())
                
                print(f"Precautions found in dataset: {my_precautions}")
                break
        
        if not matched_precautions:
            print(f"No precautions found in dataset for disease: {dis}")
            my_precautions = ["Consult a healthcare professional for precautions"]
        
        # -------------------- 2. Get medications from dataset --------------------
        # Try to match with any of the disease variations
        matched_medications = False
        for disease_var in disease_variations:
            # Try both exact and case-insensitive matches
            med_df = medications[medications['Disease'] == disease_var]
            if med_df.empty:
                med_df = medications[medications['Disease'].str.lower() == disease_var.lower()]
            
            if not med_df.empty:
                matched_medications = True
                print(f"Matched medications with: {disease_var}")
                
                # Extract medications
                if 'Medication' in med_df.columns:
                    med_str = med_df['Medication'].iloc[0]
                    print(f"Raw medication data: {med_str}")
                    
                    # Handle the case where medication is stored as a list in string format
                    if isinstance(med_str, str):
                        if med_str.startswith('[') and med_str.endswith(']'):
                            try:
                                # Try to safely evaluate the string as a list
                                med_list = eval(med_str)
                                if isinstance(med_list, list):
                                    meds = [m.strip() for m in med_list if m and isinstance(m, str) and m.strip()]
                            except:
                                # If eval fails, try a simpler parsing approach
                                med_str = med_str.strip('[]').replace("'", "").replace('"', '')
                                meds = [m.strip() for m in med_str.split(',') if m.strip()]
                        else:
                            # If not a list format, split by semicolons or commas
                            if ';' in med_str:
                                meds = [m.strip().capitalize() for m in med_str.split(';') if m.strip()]
                            else:
                                meds = [m.strip().capitalize() for m in med_str.split(',') if m.strip()]
                
                print(f"Medications found in dataset: {meds}")
                break
        
        if not matched_medications:
            print(f"No medications found in dataset for disease: {dis}")
            meds = ["Consult a healthcare professional for medications"]
        
        # -------------------- 3. Get diet recommendations from dataset --------------------
        # Try to match with any of the disease variations
        matched_diet = False
        for disease_var in disease_variations:
            # Try both exact and case-insensitive matches
            diet_df = diets[diets['Disease'] == disease_var]
            if diet_df.empty:
                diet_df = diets[diets['Disease'].str.lower() == disease_var.lower()]
            
            if not diet_df.empty:
                matched_diet = True
                print(f"Matched diet with: {disease_var}")
                
                # Extract diet recommendations
                if 'Diet' in diet_df.columns:
                    diet_str = diet_df['Diet'].iloc[0]
                    print(f"Raw diet data: {diet_str}")
                    
                    # Handle the case where diet is stored as a list in string format
                    if isinstance(diet_str, str):
                        if diet_str.startswith('[') and diet_str.endswith(']'):
                            try:
                                # Try to safely evaluate the string as a list
                                diet_list = eval(diet_str)
                                if isinstance(diet_list, list):
                                    diet = [d.strip() for d in diet_list if d and isinstance(d, str) and d.strip()]
                            except:
                                # If eval fails, try a simpler parsing approach
                                diet_str = diet_str.strip('[]').replace("'", "").replace('"', '')
                                diet = [d.strip() for d in diet_str.split(',') if d.strip()]
                        else:
                            # If not a list format, split by semicolons or commas
                            if ';' in diet_str:
                                diet = [d.strip().capitalize() for d in diet_str.split(';') if d.strip()]
                            else:
                                diet = [d.strip().capitalize() for d in diet_str.split(',') if d.strip()]
                
                print(f"Diet recommendations found in dataset: {diet}")
                break
        
        if not matched_diet:
            print(f"No diet recommendations found in dataset for disease: {dis}")
            diet = ["Consult a nutritionist for dietary recommendations"]
        
        # -------------------- 4. Get workout recommendations from dataset --------------------
        # First check if we have specific workout data in our dataset
        matched_workout = False
        workout_recs = []
        
        for disease_var in disease_variations:
            # Try both exact and case-insensitive matches
            workout_df_filtered = workout[workout['disease'] == disease_var]
            if workout_df_filtered.empty:
                workout_df_filtered = workout[workout['disease'].str.lower() == disease_var.lower()]
            
            if not workout_df_filtered.empty:
                matched_workout = True
                print(f"Matched workout with: {disease_var}")
                
                # Extract workout recommendations (unique values)
                if 'workout' in workout_df_filtered.columns:
                    workout_recs = list(set([w.strip().capitalize() for w in workout_df_filtered['workout'].tolist() 
                                         if isinstance(w, str) and w.strip()]))
                    # Only take first 5 workout suggestions to avoid too many
                    if len(workout_recs) > 5:
                        workout_recs = workout_recs[:5]
                
                print(f"Workout recommendations found in dataset: {workout_recs}")
                break
        
        # Use workout recommendations if found, otherwise suggest consulting a professional
        if matched_workout and workout_recs:
            work = workout_recs
        else:
            print(f"No workout recommendations found in dataset for disease: {dis}")
            work = ["Consult a physical therapist for exercise recommendations"]
        
        # Print final recommendations for debugging
        print(f"Final Precautions: {my_precautions}")
        print(f"Final Medications: {meds}")
        print(f"Final Diet: {diet}")
        print(f"Final Workout: {work}")
        
        return dis_des, my_precautions, meds, diet, work

    except Exception as e:
        print(f"Helper function detailed error:", e)
        traceback.print_exc()
        
        # Provide fallback information
        try:
            error_description = generate_disease_description(dis)
        except:
            error_description = f"{dis} is a medical condition that requires professional medical attention."
        
        return (error_description, 
                ["Consult a healthcare professional for precautions"], 
                ["Consult a healthcare professional for medications"], 
                ["Consult a nutritionist for dietary recommendations"], 
                ["Consult a physical therapist for exercise recommendations"])

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