from flask import Flask, render_template, request, jsonify
from datetime import datetime
import pickle
import os

app = Flask(__name__)

# Load the trained recommender model
MODEL_PATH = "/hybrid_airline_reccommender.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        recommender_model = pickle.load(f)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    recommender_model = None

# Recommender function using trained model
def get_recommendations(user_data):
    """
    Example assumes the model has a .predict() method that returns a list of dicts.
    Adapt based on your trained model structure.
    """
    try:
        # Predict using the trained model
        predictions = recommender_model.predict([user_data])
        return predictions[:3]  # return top 3 results
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return [{"id": "ERR001", "airline": "Unknown", "price": "$0"}]

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    session = request.json.get("session")

    # Normalize input for validation
    user_input_clean = user_input.strip().lower()

    if session.get("step") == "origin":
        session["origin"] = user_input
        session["step"] = "destination"
        return jsonify(response="Where are you flying to?", session=session)

    elif session.get("step") == "destination":
        session["destination"] = user_input
        session["step"] = "class"
        return jsonify(
            response="What class would you like? (Economy / Premium / Business / First Class)",
            session=session
        )

    elif session.get("step") == "class":
        valid_classes = ["economy", "premium", "business", "first class"]
        if user_input_clean not in valid_classes:
            return jsonify(
                response="Please enter a valid class: Economy, Premium, Business, or First Class.",
                session=session
            )
        session["class"] = user_input.title()
        session["step"] = "departure"
        return jsonify(response="What's your departure date? (YYYY-MM-DD)", session=session)

    elif session.get("step") == "departure":
        # Basic date format validation
        try:
            datetime.strptime(user_input, "%Y-%m-%d")
        except ValueError:
            return jsonify(
                response="Invalid date format. Please enter the departure date as YYYY-MM-DD.",
                session=session
            )
        session["departure"] = user_input
        session["step"] = "travellerType"
        return jsonify(
            response="What type of traveller are you? (Solo Leisure / Family Leisure / Business / Couple Leisure)",
            session=session
        )

    elif session.get("step") == "travellerType":
        valid_traveller_types = ["solo leisure", "family leisure", "business", "couple leisure"]
        if user_input_clean not in valid_traveller_types:
            return jsonify(
                response="Please specify your traveller type: Solo Leisure, Family Leisure, Business, or Couple Leisure.",
                session=session
            )
        session["travellerType"] = user_input.title()
        session["step"] = "recommend"

        # Gather inputs and send to trained model
        user_data = {
            "origin": session["origin"],
            "destination": session["destination"],
            "class": session["class"],
            "departure": session["departure"],
            "travellerType": session["travellerType"]
        }

        recommendations = get_recommendations(user_data)
        session["recommendations"] = recommendations
        session["step"] = "select"

        msg = "Here are your flight options:\n"
        for i, r in enumerate(recommendations):
            msg += f"{i + 1}. {r['airline']} - {r['price']}\n"
        msg += "Choose a flight number (e.g., 1)"

        return jsonify(response=msg, session=session)

    elif session.get("step") == "select":
        try:
            index = int(user_input.strip()) - 1
            selected = session["recommendations"][index]
            session["step"] = "done"
            return jsonify(
                response=f"🎉 Booking confirmed with {selected['airline']}! Thank you for using our service.",
                session=session
            )
        except:
            return jsonify(
                response="Invalid selection. Please enter a number like 1, 2 or 3.",
                session=session
            )

    else:
        session["step"] = "origin"
        return jsonify(response="Hi! Where are you flying from?", session=session)

if __name__ == "__main__":
    app.run(debug=True)
