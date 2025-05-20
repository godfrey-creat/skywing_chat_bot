from flask import Flask, render_template, request, jsonify, session
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
import secrets
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_url_path='/static')
app.secret_key = secrets.token_hex(16)

app.recent_recommendations = []
MAX_RECENT_RECOMMENDATIONS = 10

# Load model
MODEL_PATH = 'hybrid_airline_recommender.pkl'
try:
    with open(MODEL_PATH, 'rb') as file:
        hybrid_model = pickle.load(file)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    hybrid_model = None

def load_data():
    try:
        df = pd.read_csv('new_data.csv')
        print("Using New Data")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return create_sample_data()

def create_sample_data():
    airlines = [
        "Singapore Airlines", "Qatar Airways", "All Nippon Airways",
        "Emirates", "Japan Airlines", "Cathay Pacific Airways",
        "EVA Air", "Lufthansa", "Korean Air", "Delta Air Lines"
    ]
    routes = [
        "Asia to Asia", "Asia to Europe", "Europe to Asia",
        "America to Asia", "Asia to America", "Europe to America",
        "America to Europe", "Oceania to Asia", "Asia to Oceania"
    ]
    classes = ["Economy Class", "Premium Economy", "Business Class", "First Class"]
    data = []

    for airline in airlines:
        for route in routes:
            for travel_class in classes:
                data.append({
                    'Airline': airline,
                    'Continent_Route': route,
                    'Class': travel_class,
                    'Seat Comfort': np.random.uniform(3, 5),
                    'Staff Service': np.random.uniform(3, 5),
                    'Food & Beverages': np.random.uniform(3, 5),
                    'Inflight Entertainment': np.random.uniform(3, 5),
                    'Value For Money': np.random.uniform(3, 5),
                    'Overall Rating': np.random.uniform(7, 10),
                    'Recommended': 'yes' if np.random.random() > 0.2 else 'no',
                    'Season': np.random.choice(['Winter', 'Spring', 'Summer', 'Autumn']),
                    'Type of Traveller': np.random.choice(['Solo Leisure', 'Family Leisure', 'Business', 'Couple Leisure'])
                })

    return pd.DataFrame(data)

df = load_data()

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    return 'Autumn'

def generate_recommendations(user_data):
    if not hybrid_model:
        return None

    try:
        origin = user_data.get('origin')
        destination = user_data.get('destination')
        travel_class = user_data.get('class')
        departure_date = user_data.get('departure')
        traveller_type = user_data.get('travellerType')
        route = f"{origin} to {destination}"

        season = get_season(datetime.strptime(departure_date, '%Y-%m-%d').month) if departure_date else get_season(datetime.now().month)
        candidates = df.copy()

        route_candidates = candidates[candidates['Continent_Route'].str.contains(route, case=False, na=False)]
        if route_candidates.empty:
            route_candidates = candidates[(candidates['Continent_Route'].str.contains(origin, case=False, na=False)) &
                                          (candidates['Continent_Route'].str.contains(destination, case=False, na=False))]
        if not route_candidates.empty:
            candidates = route_candidates

        if travel_class and travel_class != "Any":
            class_map = {
                "economy": "Economy Class",
                "premium": "Premium Economy",
                "business": "Business Class",
                "first": "First Class"
            }
            for key in class_map:
                if key in travel_class.lower():
                    travel_class = class_map[key]
            candidates = candidates[candidates['Class'] == travel_class]

        if traveller_type and traveller_type != "Any":
            candidates = candidates[candidates['Type of Traveller'] == traveller_type]

        if 'Season' in candidates.columns:
            candidates = candidates[candidates['Season'] == season]

        unique_combinations = candidates.drop_duplicates(['Airline', 'Continent_Route', 'Class']).reset_index(drop=True)
        if unique_combinations.empty:
            return None

        prediction_data = []
        for _, row in unique_combinations.iterrows():
            prediction_data.append({
                'Airline': row['Airline'],
                'Continent_Route': row['Continent_Route'],
                'Class': row['Class'],
                'Type of Traveller': traveller_type or row.get('Type of Traveller', 'Solo Leisure'),
                'Season': season,
                'Seat Comfort': row.get('Seat Comfort', df['Seat Comfort'].mean()),
                'Staff Service': row.get('Staff Service', df['Staff Service'].mean()),
                'Food & Beverages': row.get('Food & Beverages', df['Food & Beverages'].mean()),
                'Inflight Entertainment': row.get('Inflight Entertainment', df['Inflight Entertainment'].mean()),
                'Value For Money': row.get('Value For Money', df['Value For Money'].mean())
            })

        pred_df = pd.DataFrame(prediction_data)

        for col in ['Airline', 'Continent_Route', 'Class', 'Type of Traveller', 'Season']:
            le = LabelEncoder()
            pred_df[f'{col}_encoded'] = le.fit_transform(pred_df[col])

        if hasattr(hybrid_model, 'feature_names_in_'):
            for col in hybrid_model.feature_names_in_:
                if col not in pred_df.columns:
                    pred_df[col] = 0
            X_pred = pred_df[hybrid_model.feature_names_in_]
        else:
            X_pred = pred_df.select_dtypes(include=[np.number])

        predictions = hybrid_model.predict(X_pred)
        unique_combinations['predicted_score'] = predictions
        top_recommendations = unique_combinations.sort_values('predicted_score', ascending=False).head(5)

        result = []
        for _, rec in top_recommendations.iterrows():
            base_price = np.random.uniform(300, 800)
            price_multiplier = {
                'Economy Class': 1.0,
                'Premium Economy': 1.5,
                'Business Class': 3.0,
                'First Class': 5.0
            }.get(rec['Class'], 1.0)
            price = base_price * price_multiplier * (0.9 + np.random.random() * 0.2)
            result.append({
                'airline': rec['Airline'],
                'route': rec['Continent_Route'],
                'class': rec['Class'],
                'rating': f"{rec.get('Overall Rating', 7.5):.1f}/10",
                'value_rating': f"{rec.get('Value For Money', 4.0):.1f}/5",
                'price': f"${int(price)}",
                'season': rec.get('Season', season)
            })

        return result

    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat")
def chat_interface():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def chat_api():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_input = data.get("message", "").strip()
    chat_session = data.get("session", {})

    if not chat_session:
        chat_session = {"step": "origin"}

    user_input_clean = user_input.lower()

    if chat_session.get("step") == "origin":
        chat_session["origin"] = user_input
        chat_session["step"] = "destination"
        return jsonify({"response": "Where are you flying to?", "session": chat_session})

    elif chat_session.get("step") == "destination":
        chat_session["destination"] = user_input
        chat_session["step"] = "class"
        return jsonify({"response": "What class would you like? (Economy/Premium/Business/First Class)", "session": chat_session})

    elif chat_session.get("step") == "class":
        valid_classes = ["economy", "premium", "business", "first", "first class"]
        if not any(cls in user_input_clean for cls in valid_classes):
            return jsonify({"response": "Please enter a valid class.", "session": chat_session})

        if "economy" in user_input_clean:
            chat_session["class"] = "Economy Class"
        elif "premium" in user_input_clean:
            chat_session["class"] = "Premium Economy"
        elif "business" in user_input_clean:
            chat_session["class"] = "Business Class"
        elif "first" in user_input_clean:
            chat_session["class"] = "First Class"

        chat_session["step"] = "departure"
        return jsonify({"response": "When is your departure date? (YYYY-MM-DD)", "session": chat_session})

    elif chat_session.get("step") == "departure":
        try:
            datetime.strptime(user_input, '%Y-%m-%d')
            chat_session["departure"] = user_input
            chat_session["step"] = "travellerType"
            return jsonify({"response": "What type of traveler are you? (Solo Leisure, Business, Couple Leisure, Family Leisure)", "session": chat_session})
        except ValueError:
            return jsonify({"response": "Invalid date format. Please use YYYY-MM-DD.", "session": chat_session})

    elif chat_session.get("step") == "travellerType":
        chat_session["travellerType"] = user_input
        chat_session["step"] = "complete"

        recommendations = generate_recommendations(chat_session)
        if not recommendations:
            return jsonify({"response": "Sorry, no recommendations found.", "session": chat_session})

        response_lines = ["Here are some recommendations:\n"]
        for rec in recommendations:
            line = f"- {rec['airline']} | {rec['route']} | {rec['class']} | {rec['rating']} | {rec['price']}"
            response_lines.append(line)
        return jsonify({"response": "\n".join(response_lines), "session": chat_session})

    return jsonify({"response": "Invalid step.", "session": chat_session})

if __name__ == "__main__":
    app.run(debug=True)
