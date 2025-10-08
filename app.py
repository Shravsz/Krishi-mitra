from flask import Flask, render_template, request, redirect, session, url_for, jsonify
import json
import os
import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = 'AIzaSyBOja-aPAutC4kSN4fgBcPvtJCCNyYOZlk'  # Replace this with an env variable in production

# ---------- Gemini Configuration ----------
genai.configure(api_key="AIzaSyBOja-aPAutC4kSN4fgBcPvtJCCNyYOZlk")
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------- Utility Functions ----------
def load_users():
    if not os.path.exists('users.json'):
        return []
    with open('users.json', 'r') as f:
        return json.load(f)

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=4)

# ---------- Fertilizer Model Preparation ----------pyth
fertilizer_df = pd.read_csv('C:/Users/hp/OneDrive/Desktop/krishi_mitra final/data/fertilizer.csv')


le_soil = LabelEncoder()
le_crop = LabelEncoder()
fertilizer_df['Soil Type'] = le_soil.fit_transform(fertilizer_df['Soil Type'])
fertilizer_df['Crop Type'] = le_crop.fit_transform(fertilizer_df['Crop Type'])
X_fert = fertilizer_df[['Temparature', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Phosphorous', 'Potassium']]
y_fert = fertilizer_df['Fertilizer Name']
fert_model = RandomForestClassifier(n_estimators=100, random_state=42)
fert_model.fit(X_fert, y_fert)

# ---------- Home ----------
@app.route('/')
def home():
    return redirect('/login')

# ---------- Login ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        users = load_users()
        user = next((u for u in users if u['email'] == email), None)

        if user:
            if user['password'] == password:
                session['name'] = user['name']
                session['email'] = user['email']
                return redirect('/dashboard')
            else:
                return render_template('login.html', error="Wrong password.")
        else:
            return render_template('login.html', error="User not found.")

    return render_template('login.html')

# ---------- Signup ----------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match.")

        users = load_users()
        if any(u['email'] == email for u in users):
            return render_template('signup.html', error="User already exists.")

        users.append({'name': name, 'email': email, 'password': password, 'profile_image': ''})

        save_users(users)

        session['name'] = name
        session['email'] = email
        return redirect('/dashboard')

    return render_template('signup.html')

# ---------- Dashboard ----------
@app.route('/dashboard')
def dashboard():
    if 'name' in session and 'email' in session:
        return render_template('dashboard.html', name=session['name'], email=session['email'])
    else:
        return redirect('/login')

# ---------- Profile ----------
@app.route('/profile')
def profile():
    if 'email' not in session:
        return redirect('/login')

    users = load_users()
    user = next((u for u in users if u['email'] == session['email']), None)

    if user:
        return render_template('profile.html', user=user)
    else:
        return redirect('/login')

# ---------- Edit Profile ----------
@app.route('/edit-profile', methods=['GET', 'POST'])
def edit_profile():
    if 'email' not in session:
        return redirect('/login')

    users = load_users()
    user = next((u for u in users if u['email'] == session['email']), None)

    if not user:
        return redirect('/login')

    if request.method == 'POST':
        new_name = request.form['name']
        new_password = request.form['password']
        profile_photo = request.files.get('profile_photo')

        if new_name:
            user['name'] = new_name
            session['name'] = new_name

        if new_password:
            user['password'] = new_password

        # ✅ Save uploaded profile photo
        if profile_photo and profile_photo.filename != '':
            ext = os.path.splitext(profile_photo.filename)[1]
            filename = f"{user['email'].replace('@', '_').replace('.', '_')}{ext}"
            filepath = os.path.join('static', 'profile_photos', filename)
            profile_photo.save(filepath)
            user['profile_image'] = url_for('static', filename=f'profile_photos/{filename}')

        save_users(users)
        return redirect('/profile')

    return render_template('edit_profile.html', user=user)


# ---------- Delete Account ----------
@app.route('/delete-account', methods=['POST'])
def delete_account():
    if 'email' not in session:
        return redirect('/login')

    users = load_users()
    users = [u for u in users if u['email'] != session['email']]
    save_users(users)

    session.clear()
    return redirect('/signup')

# ---------- Logout ----------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# ---------- Placeholder Routes ----------
@app.route('/crop', methods=['GET', 'POST'])
def crop():
    # Load dataset and model
    df_crop = pd.read_csv('C:/Users/hp/OneDrive/Desktop/krishi_mitra final/data/LAST_FINAL DATASET.csv')
    df_crop = df_crop.dropna(subset=["Location", "Season", "Soil type", "Crops"])
    df_crop["Location"] = df_crop["Location"].astype(str).str.strip()

    le_location = LabelEncoder()
    le_season = LabelEncoder()
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()

    df_crop["Location"] = le_location.fit_transform(df_crop["Location"])
    df_crop["Season"] = le_season.fit_transform(df_crop["Season"])
    df_crop["Soil type"] = le_soil.fit_transform(df_crop["Soil type"])
    df_crop["Crops"] = le_crop.fit_transform(df_crop["Crops"])

    X = df_crop[["Location", "Season", "Soil type"]]
    y = df_crop["Crops"]

    model = RandomForestClassifier()
    model.fit(X, y)

    if request.method == 'POST':
        location = request.form['location']
        season = request.form['season']
        soil_type = request.form['soil_type']

        try:
            input_data = np.array([[
                le_location.transform([location])[0],
                le_season.transform([season])[0],
                le_soil.transform([soil_type])[0]
            ]])

            prediction = model.predict(input_data)
            predicted_crop = le_crop.inverse_transform(prediction)[0]

            return render_template('crop_result.html', crop=predicted_crop)
        except Exception as e:
            return render_template('crop_form.html', error=str(e))

    return render_template(
        'crop_form.html',
        locations=le_location.classes_,
        seasons=le_season.classes_,
        soil_types=le_soil.classes_
    )




@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    recommendation = None

    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        moisture = float(request.form['moisture'])
        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']
        nitrogen = float(request.form['nitrogen'])
        phosphorous = float(request.form['phosphorous'])
        potassium = float(request.form['potassium'])

        input_data = np.array([[
            temperature,
            moisture,
            le_soil.transform([soil_type])[0],
            le_crop.transform([crop_type])[0],
            nitrogen,
            phosphorous,
            potassium
        ]])

        prediction = fert_model.predict(input_data)[0]
        recommendation = prediction

    return render_template(
        'fertilizer.html',
        soil_types=le_soil.classes_,
        crop_types=le_crop.classes_,
        recommendation=recommendation
    )



@app.route('/api/karnataka-schemes')
def api_karnataka_schemes():
    try:
       
        
        with open('schemes_data.json', 'r', encoding='utf-8') as f:


            schemes_data = json.load(f)
        
        category = request.args.get('category', 'All')

        if category == 'All':
            filtered = schemes_data
        else:
            filtered = [s for s in schemes_data if category in s['eligible_for']]

        return jsonify(filtered)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------------------- Render Schemes Page -----------------------------
@app.route('/schemes')
def schemes():
    return render_template('karnataka_schemes.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/weather')
def weather():
    return render_template('weather.html')

# ---------- Chatbot Routes ----------
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json['message']
    try:
        response = model.generate_content(user_message)
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'response': f"❌ ದೋಷ: {str(e)}"})

# ---------- Run App ----------
if __name__ == '__main__':
    app.run(debug=True)
