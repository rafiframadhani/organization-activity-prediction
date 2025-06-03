from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_super_secret_key' # Perlu untuk flash messages
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize global variables to sensible defaults to prevent errors if loading fails
model = None
oe_position = None
oe_chat = None
feature_importances = pd.Series()
max_submitted_works = 1.0 
max_competition_participation = 1.0 
min_activity_score = 0.0
max_activity_score = 100.0

# Load model and encoders
try:
    model = joblib.load("model/model_regressor.pkl")
    oe_position = joblib.load("model/oe_position.pkl")
    oe_chat = joblib.load("model/oe_chat.pkl")
    
    # Ensure feature_importances is a Pandas Series
    loaded_feature_importances = joblib.load("model/feature_importances.pkl")
    if isinstance(loaded_feature_importances, dict):
        feature_importances = pd.Series(loaded_feature_importances)
    elif isinstance(loaded_feature_importances, pd.Series):
        feature_importances = loaded_feature_importances
    else:
        print("Warning: feature_importances.pkl is not a dict or Series. Initializing as empty.")
        feature_importances = pd.Series()

except Exception as e:
    print(f"Error loading model or encoders: {e}")
    print("Please ensure train_model.py has been run and all necessary files are in the 'model/' directory.")


# Definisi mapping untuk OrdinalEncoder (harus sama dengan train_model.py)
position_order_map = {
    "Staff": 0, "Supervisor": 1,
    "Department Head": 2, "Deputy Department Head": 2, "Division Head": 2, "Vice Division Head": 2,
    "Secretary": 3, "Treasurer": 3, "Vice Chair": 3, "Chair": 3
}
chat_activity_order_map = {
    "Inactive": 0, "Rarely Active": 1, "Moderate": 2, "Active": 3, "Very Active": 4
}
# Tambahkan max_values untuk skala 0-1 di app.py
max_position_encoded_value = max(position_order_map.values()) if position_order_map else 1
max_chat_activity_encoded_value = max(chat_activity_order_map.values()) if chat_activity_order_map else 1


# Fungsi untuk mengkategorikan skor keaktifan
def categorize_activity_score(score):
    if score >= 80:
        return "Very Active"
    elif score >= 60:
        return "Active"
    elif score >= 40:
        return "Moderate"
    elif score >= 30:
        return "Rarely Active"
    else:
        return "Inactive"

# Fungsi untuk menghasilkan rekomendasi aksi
def generate_recommendations(member_data, predicted_status):
    recommendations = []

    if predicted_status in ["Inactive", "Rarely Active"]:
        recommendations.append(f"Consider initiating a personal check-in with {member_data['name']} to understand challenges.")
        
        if member_data["event_attendance"] < 0.5:
            recommendations.append("Encourage participation in upcoming events and provide incentives.")
        if member_data["meeting_attendance"] < 0.5:
            recommendations.append("Remind to attend next important meetings and highlight meeting benefits.")
        if member_data["duty_participation"] < 0.5:
            recommendations.append("Assign simple, clear duties to rebuild engagement and provide easy wins.")
            
        if member_data["submitted_works"] < 3:
             recommendations.append("Offer support or ideas for creative projects/works submission.")
        if member_data["competition_participation"] == 0:
            recommendations.append("Encourage participation in photography/design competitions or provide mentorship.")

        if member_data["chat_activity"] in ["Inactive", "Rarely Active"]:
            recommendations.append("Boost engagement in group chats or communication channels by asking direct questions.")

    elif predicted_status == "Moderate":
        recommendations.append(f"Continue to monitor {member_data['name']}'s engagement closely and provide growth opportunities.")
        if member_data["event_attendance"] < 0.7:
            recommendations.append("Motivate to attend more events by emphasizing networking or learning aspects.")
        if member_data["submitted_works"] < 5:
             recommendations.append("Provide opportunities for more creative contributions and acknowledge efforts.")
        if member_data["chat_activity"] == "Moderate":
            recommendations.append("Encourage more active participation in discussions by creating engaging topics.")

    elif predicted_status in ["Active", "Very Active"]:
        recommendations.append(f"{member_data['name']} is doing great! Keep up the good work and consider leadership roles.")
        if member_data["event_attendance"] > 0.9:
            recommendations.append("Utilize their high event attendance as an example for other members.")
        if member_data["submitted_works"] > 10:
            recommendations.append("Showcase their excellent submitted works as inspiration and mentor new members.")
        if member_data["competition_participation"] > 2:
            recommendations.append("Consider involving them in mentoring others for competitions or leading a team.")

    return list(set(recommendations))


# Global preprocessing function (for both mass and single prediction)
def preprocess_data(df_input):
    df = df_input.copy()

    # DEBUG: Print original input DataFrame
    print(f"\nDEBUG PREPROCESS: Original input df:\n{df}\n")

    def convert_fraction_value(value):
        try:
            if isinstance(value, str) and '/' in str(value):
                num, den = map(int, value.split('/'))
                return round(num / den, 2)
            return float(value)
        except (ValueError, TypeError, ZeroDivisionError):
            return np.nan

    for col in ["event_attendance", "meeting_attendance", "duty_participation"]:
        df[col] = df[col].apply(convert_fraction_value)
        df[col] = df[col].fillna(0)

    # DEBUG: Print df after fraction conversion and fillna
    print(f"DEBUG PREPROCESS: df after fraction conversion and fillna:\n{df[['event_attendance', 'meeting_attendance', 'duty_participation']]}\n")

    # Encode categorical features using the pre-loaded OrdinalEncoders
    df["position_encoded"] = df["position"].apply(
        lambda x: position_order_map.get(x, -1) # -1 for unknown position
    )
    df["chat_activity_encoded"] = df["chat_activity"].apply(
        lambda x: chat_activity_order_map.get(x, -1) # -1 for unknown chat activity
    )

    if (-1 in df["position_encoded"].values) or (-1 in df["chat_activity_encoded"].values):
        flash("Warning: One or more categorical inputs were not recognized. Predictions for affected rows might be inaccurate.", "warning")

    # NEW: Scale encoded categorical features to 0-1 for consistency with model training
    df['chat_activity_scaled_0_1'] = df['chat_activity_encoded'] / (max_chat_activity_encoded_value if max_chat_activity_encoded_value > 0 else 1)
    df['position_scaled_0_1'] = df['position_encoded'] / (max_position_encoded_value if max_position_encoded_value > 0 else 1)


    # DEBUG: Print df after categorical encoding and scaling
    print(f"DEBUG PREPROCESS: df after categorical encoding and scaling:\n{df[['position', 'position_encoded', 'position_scaled_0_1', 'chat_activity', 'chat_activity_encoded', 'chat_activity_scaled_0_1']]}\n")

    # Normalisasi 'submitted_works' dan 'competition_participation'
    df["submitted_works"] = pd.to_numeric(df["submitted_works"], errors='coerce').fillna(0)
    df["competition_participation"] = pd.to_numeric(df["competition_participation"], errors='coerce').fillna(0)

    # Corrected line: Normalize from original 'competition_participation'
    df["submitted_works_norm"] = df["submitted_works"] / (max_submitted_works if max_submitted_works > 0 else 1.0)
    df["competition_participation_norm"] = df["competition_participation"] / (max_competition_participation if max_competition_participation > 0 else 1.0)

    df["submitted_works_norm"] = df["submitted_works_norm"].fillna(0)
    df["competition_participation_norm"] = df["competition_participation_norm"].fillna(0)

    # DEBUG: Print df after numerical normalization
    print(f"DEBUG PREPROCESS: df after numerical normalization:\n{df[['submitted_works', 'submitted_works_norm', 'competition_participation', 'competition_participation_norm']]}\n")


    # UPDATE: features_to_predict sekarang harus mencakup SEMUA fitur yang digunakan saat TRAINING
    features_to_predict = [
        "position_encoded", 
        "event_attendance",
        "meeting_attendance",
        "duty_participation",
        "submitted_works_norm",
        "competition_participation_norm",
        "chat_activity_scaled_0_1",
        "position_scaled_0_1"
    ]

    # Check if all features_to_predict exist in df
    missing_features = [f for f in features_to_predict if f not in df.columns]
    if missing_features:
        flash(f"Error: Missing features for prediction: {', '.join(missing_features)}", "error")
        raise ValueError(f"Missing features for prediction: {', '.join(missing_features)}")


    # DEBUG: Print final features DataFrame
    print(f"DEBUG PREPROCESS: Final features DataFrame for prediction:\n{df[features_to_predict]}\n")

    return df, features_to_predict


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_mass', methods=['POST'])
def predict_mass():
    if model is None:
        flash("Model not loaded. Please train the model first.", "error")
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash("No file part", "error")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash("No selected file", "error")
        return redirect(request.url)
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        df_original = pd.read_csv(filepath)
        df_processed, features_to_predict = preprocess_data(df_original)

        # Predict scores
        predicted_scores_raw = model.predict(df_processed[features_to_predict])
        
        # DEBUG: Print raw predicted scores
        print(f"DEBUG PREDICT MASS: Raw predicted scores: {predicted_scores_raw}")

        # Karena model dilatih untuk memprediksi langsung 0-100, kita langsung clip hasilnya
        predicted_scores_scaled = np.clip(predicted_scores_raw, 0, 100).round(2)
        
        # DEBUG: Print scaled predicted scores
        print(f"DEBUG PREDICT MASS: Scaled predicted scores: {predicted_scores_scaled}")

        df_original['Predicted Activity Score'] = predicted_scores_scaled
        df_original['Predicted Activity Status'] = df_original['Predicted Activity Score'].apply(categorize_activity_score)
        
        # Calculate status counts for composition graph
        status_counts = df_original['Predicted Activity Status'].value_counts().to_dict()
        
        # Ensure all categories are present, even if 0, and in a desired order
        all_categories_ordered = ["Very Active", "Active", "Moderate", "Rarely Active", "Inactive"]
        sorted_status_composition = {cat: status_counts.get(cat, 0) for cat in all_categories_ordered}


        # Prepare data for display
        display_cols = ['name', 'position', 'event_attendance', 'meeting_attendance',
                        'duty_participation', 'submitted_works', 'competition_participation',
                        'chat_activity', 'Predicted Activity Score', 'Predicted Activity Status']
        df_display = df_original[display_cols] 

        result_path = os.path.join(UPLOAD_FOLDER, 'results_mass.csv')
        df_display.to_csv(result_path, index=False)

        feature_importance_dict = feature_importances.to_dict()
        # Calculate max_importance for scaling feature importance bars in the template
        # Handle case where feature_importances might be empty or all zeros
        max_importance = max(feature_importance_dict.values()) if feature_importance_dict and any(feature_importance_dict.values()) else 1.0
        
        print(f"DEBUG PREDICT MASS: feature_importance_dict: {feature_importance_dict}")
        print(f"DEBUG PREDICT MASS: max_importance: {max_importance}")


        return render_template('results.html',
                               tables=[df_display.to_html(classes='data', index=False)],
                               titles=df_display.columns.values,
                               feature_importances=feature_importance_dict,
                               max_feature_importance=max_importance, # Pass max importance
                               is_mass_prediction=True,
                               status_composition=sorted_status_composition) # Pass status composition

    flash("Invalid file type. Please upload a CSV file.", "error")
    return redirect(url_for('index'))


@app.route('/single_prediction')
def single_prediction_form():
    positions = list(position_order_map.keys())
    chat_activities = list(chat_activity_order_map.keys())
    return render_template('predict_single.html', positions=positions, chat_activities=chat_activities)


@app.route('/predict_single', methods=['POST'])
def predict_single():
    if model is None:
        flash("Model not loaded. Please train the model first.", "error")
        return redirect(url_for('single_prediction_form'))

    try:
        name = request.form['name']
        position = request.form['position']
        event_attendance = request.form['event_attendance']
        meeting_attendance = request.form['meeting_attendance']
        duty_participation = request.form['duty_participation']
        submitted_works = int(request.form['submitted_works'])
        competition_participation = int(request.form['competition_participation'])
        chat_activity = request.form['chat_activity']

        member_data_raw = {
            'name': [name],
            'position': [position],
            'event_attendance': [event_attendance],
            'meeting_attendance': [meeting_attendance],
            'duty_participation': [duty_participation],
            'submitted_works': [submitted_works],
            'competition_participation': [competition_participation],
            'chat_activity': [chat_activity]
        }
        df_single = pd.DataFrame(member_data_raw)

        df_processed, features_to_predict = preprocess_data(df_single)
        
        predicted_scores_raw = model.predict(df_processed[features_to_predict])
        
        # DEBUG: Print raw predicted score for single prediction
        print(f"DEBUG SINGLE PREDICT: Raw predicted score: {predicted_scores_raw[0]:.2f}")

        # Karena model dilatih untuk memprediksi langsung 0-100, kita langsung clip hasilnya
        predicted_score_scaled = np.clip(predicted_scores_raw[0], 0, 100).round(2)
        predicted_status = categorize_activity_score(predicted_score_scaled)

        # DEBUG: Print scaled predicted score for single prediction
        print(f"DEBUG SINGLE PREDICT: Scaled predicted score: {predicted_score_scaled:.2f}")


        original_member_for_recs = {
            'name': name,
            'position': position,
            'event_attendance': df_processed['event_attendance'].iloc[0], 
            'meeting_attendance': df_processed['meeting_attendance'].iloc[0],
            'duty_participation': df_processed['duty_participation'].iloc[0],
            'submitted_works': submitted_works,
            'competition_participation': competition_participation,
            'chat_activity': chat_activity 
        }

        # Pass only the necessary arguments to generate_recommendations
        recommendations = generate_recommendations(original_member_for_recs, predicted_status) 
        
        # Calculate max_importance for scaling feature importance bars in the template
        # Ensure feature_importances is converted to dict and handle empty case
        feature_importance_dict = feature_importances.to_dict()
        max_importance = max(feature_importance_dict.values()) if feature_importance_dict and any(feature_importance_dict.values()) else 1.0

        print(f"DEBUG SINGLE PREDICT: feature_importance_dict: {feature_importance_dict}")
        print(f"DEBUG SINGLE PREDICT: max_importance: {max_importance}")


        return render_template('results.html',
                               single_prediction_result={
                                   'name': name,
                                   'position': position,
                                   'score': predicted_score_scaled,
                                   'status': predicted_status,
                                   'recommendations': recommendations
                               },
                               feature_importances=feature_importance_dict,
                               max_feature_importance=max_importance, # Pass max importance
                               is_mass_prediction=False) 

    except Exception as e:
        flash(f"An error occurred during single prediction: {e}", "error")
        print(f"ERROR during single prediction: {e}") # Tambahkan print error agar terlihat di konsol
        return redirect(url_for('single_prediction_form'))

if __name__ == '__main__':
    app.run(debug=True)