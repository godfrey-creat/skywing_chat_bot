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

# Load the saved model
MODEL_PATH = 'hybrid_airline_recommender.pkl'
try:
    with open(MODEL_PATH, 'rb') as file:
        hybrid_model = pickle.load(file)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    hybrid_model = None

# Load airlines and routes data
def load_data():
    try:
        df = pd.read_csv('new_data.csv')
        print("Using New Data")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample data if original data isn't available"""
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
                rating = {
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
                }
                data.append(rating)

    return pd.DataFrame(data)

# Load the data
df = load_data()

continent_map = {
    'AF': 'Africa',
    'AS': 'Asia',
    'EU': 'Europe',
    'NA': 'North America',
    'SA': 'South America',
    'OC': 'Oceania'
}

def get_dropdown_options():
    airlines = sorted(df['Airline'].unique().tolist())

    continents = set()
    for route in df['Continent_Route'].unique():
        parts = route.split(' to ')
        if len(parts) == 2:
            continents.add(parts[0])
            continents.add(parts[1])
    continents = sorted(list(continents))

    classes = sorted(df['Class'].unique().tolist())
    traveller_types = sorted(df['Type of Traveller'].unique().tolist())

    return {
        'airlines': airlines,
        'continents': continents,
        'classes': classes,
        'traveller_types': traveller_types
    }

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:  # 9, 10, 11
        return 'Autumn'

# Routes
@app.route('/')
def index():
    # If user is logged in, pass username to template
    username = session.get('username', None)
    dropdown_options = get_dropdown_options()
    return render_template('index.html',
                          username=username,
                          airlines=dropdown_options['airlines'],
                          continents=dropdown_options['continents'],
                          classes=dropdown_options['classes'],
                          traveller_types=dropdown_options['traveller_types'])

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if username and password == '1234':
        session['username'] = username
        return jsonify({'success': True, 'username': username})
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/logout')
def logout():
    session.pop('username', None)
    return jsonify({'success': True})

@app.route('/update_profile', methods=['POST'])
def update_profile():
    data = request.get_json()
    session['profile'] = {
        'age': data.get('age'),
        'gender': data.get('gender'),
        'travel_frequency': data.get('travelFrequency'),
        'preferred_class': data.get('preferredClass'),
        'preferred_airline': data.get('preferredAirline')
    }
    return jsonify({'success': True})

@app.route('/get_profile')
def get_profile():
    profile = session.get('profile', {})
    return jsonify(profile)

@app.route('/recommend', methods=['POST'])
def recommend():

    request_data = request.get_json()
    request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if not hybrid_model:
        return jsonify({
            'success': False,
            'message': 'Model not available. Please contact support.'
        })

    data = request.get_json()
    origin = data.get('origin')
    destination = data.get('destination')
    travel_class = data.get('class')
    departure_date = data.get('departure')
    traveller_type = data.get('travellerType', None)

    # User profile
    profile = session.get('profile', {})

    try:
        route = f"{origin} to {destination}"

        ## Get Month
        if departure_date:
            departure_dt = datetime.strptime(departure_date, '%Y-%m-%d')
            season = get_season(departure_dt.month)
        else:
            current_month = datetime.now().month
            season = get_season(current_month)

        ## CANDIDATE GENERATION & SELECTION

        candidates = df.copy()

        route_candidates = candidates[candidates['Continent_Route'].str.contains(route, case=False, na=False)]
        if route_candidates.empty:
            route_candidates = candidates[(candidates['Continent_Route'].str.contains(origin, case=False, na=False)) &
                                         (candidates['Continent_Route'].str.contains(destination, case=False, na=False))]

        if not route_candidates.empty:
            candidates = route_candidates

        if travel_class and travel_class != "Any":
            class_candidates = candidates[candidates['Class'] == travel_class]
            if not class_candidates.empty:
                candidates = class_candidates

        if traveller_type and traveller_type != "Any":
            type_candidates = candidates[candidates['Type of Traveller'] == traveller_type]
            if not type_candidates.empty:
                candidates = type_candidates

        if 'Season' in candidates.columns:
            season_candidates = candidates[candidates['Season'] == season]
            if not season_candidates.empty and len(season_candidates) >= 10:
                candidates = season_candidates

        unique_combinations = candidates.drop_duplicates(['Airline', 'Continent_Route', 'Class']).reset_index(drop=True)

        if len(unique_combinations) == 0:
            return jsonify({
                'success': False,
                'message': 'No flight options found for your search criteria. Please try different options.'
            })

        # PREPARE FEAUTURES
        needed_cols = ['Airline_encoded', 'Continent_Route_encoded', 'Class_encoded',
                      'Type of Traveller_encoded', 'Season_encoded']

        # encoders
        for col in ['Airline', 'Continent_Route', 'Class', 'Type of Traveller', 'Season']:
            encoded_col = f'{col}_encoded'
            if encoded_col not in unique_combinations.columns and col in unique_combinations.columns:
                le = LabelEncoder()
                unique_combinations[encoded_col] = le.fit_transform(unique_combinations[col])

        # average ratings for user, airline, and route
        user_avg_ratings = {}
        airline_avg_ratings = {}
        route_avg_ratings = {}

        rating_cols = ['Seat Comfort', 'Staff Service', 'Food & Beverages',
                       'Inflight Entertainment', 'Value For Money', 'Overall Rating']

        # airline averages
        for airline in unique_combinations['Airline'].unique():
            airline_data = df[df['Airline'] == airline]
            if not airline_data.empty:
                airline_avg_ratings[airline] = airline_data[rating_cols].mean().to_dict()

        # route averages
        for route in unique_combinations['Continent_Route'].unique():
            route_data = df[df['Continent_Route'] == route]
            if not route_data.empty:
                route_avg_ratings[route] = route_data[rating_cols].mean().to_dict()

        # feature rows
        prediction_data = []
        for _, row in unique_combinations.iterrows():
            # Base features
            feature_row = {
                'Airline_encoded': row.get('Airline_encoded', 0),
                'Continent_Route_encoded': row.get('Continent_Route_encoded', 0),
                'Class_encoded': row.get('Class_encoded', 0),
            }

            # Add type and season if available
            if 'Type of Traveller_encoded' in row:
                feature_row['Type of Traveller_encoded'] = row['Type of Traveller_encoded']
            if 'Season_encoded' in row:
                feature_row['Season_encoded'] = row['Season_encoded']

            # Add rating components if available
            for col in rating_cols[:-1]:
                if col in row:
                    feature_row[col] = row[col]
                else:
                    feature_row[col] = df[col].mean()

            # Add derived features - airline averages
            airline = row['Airline']
            if airline in airline_avg_ratings:
                for col in rating_cols:
                    feature_row[f'{col}_airline_avg'] = airline_avg_ratings[airline][col]

            # Add derived features - route averages
            route = row['Continent_Route']
            if route in route_avg_ratings:
                for col in rating_cols:
                    feature_row[f'{col}_route_avg'] = route_avg_ratings[route][col]

            prediction_data.append(feature_row)

        # Convert to DataFrame
        pred_df = pd.DataFrame(prediction_data)

        # missing values imputed with means
        for col in pred_df.columns:
            if col in df.columns and pred_df[col].isnull().any():
                pred_df[col] = pred_df[col].fillna(df[col].mean())
            elif pred_df[col].isnull().any():
                pred_df[col] = pred_df[col].fillna(0)

        # Ensure all required feature columns exist
        for col in hybrid_model.feature_names_in_:
            if col not in pred_df.columns:
                pred_df[col] = 0  # Default value for missing features

        # Select only columns that the model was trained on
        model_features = [col for col in hybrid_model.feature_names_in_ if col in pred_df.columns]
        X_pred = pred_df[model_features]

        # Make predictions
        try:
            predictions = hybrid_model.predict(X_pred)
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            unique_combinations['Overall Rating'].fillna(0)
            top_recommendations = unique_combinations.sort_values('score', ascending=False).head(5)
            predictions = [1] * len(top_recommendations)

        # Add predictions to the dataframe
        unique_combinations['predicted_score'] = predictions

        # Sort by predicted score and get top 5
        top_recommendations = unique_combinations.sort_values('predicted_score', ascending=False).head(5)

        # Format results
        result = []
        for _, rec in top_recommendations.iterrows():
            base_price = np.random.uniform(300, 800)

            # Price multipliers by class
            price_multipliers = {
                'Economy Class': 1.0,
                'Premium Economy': 1.5,
                'Business Class': 3.0,
                'First Class': 5.0
            }

            # Apply multiplier based on class
            rec_class = rec['Class']
            price_multiplier = price_multipliers.get(rec_class, 1.0)

            # Generate price with some randomness
            price = base_price * price_multiplier * (0.9 + np.random.random() * 0.2)

            # Get overall rating and value for money
            overall_rating = rec.get('Overall Rating', 7.5)
            value_rating = rec.get('Value For Money', 4.0)

            result.append({
                'airline': rec['Airline'],
                'route': rec['Continent_Route'],
                'class': rec['Class'],
                'rating': f"{overall_rating:.1f}/10",
                'value_rating': f"{value_rating:.1f}/5",
                'price': f"${int(price)}",
                'season': rec.get('Season', season)
            })

        # Apply user preferences if profile exists
        if profile:
            # Prioritize preferred airline
            preferred_airline = profile.get('preferred_airline')
            if preferred_airline:
                for i, rec in enumerate(result):
                    if rec['airline'] == preferred_airline:
                        result.insert(0, result.pop(i))
                        break

        recommendation_record = {
        'timestamp': request_time,
        'request': request_data,
        'response': {
            'success': True,
            'count': len(result)
        },
        'user': session.get('username', 'anonymous'),
        'results': result[:2]
    }

        # Add to the recent recommendations list
        app.recent_recommendations.insert(0, recommendation_record)

        # Latest N recommendations
        if len(app.recent_recommendations) > MAX_RECENT_RECOMMENDATIONS:
            app.recent_recommendations = app.recent_recommendations[:MAX_RECENT_RECOMMENDATIONS]

        return jsonify({
            'success': True,
            'recommendations': result
        })

    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error generating recommendations: {str(e)}"
        })

@app.route('/status')
def status():
    """
    Debugging endpoint that shows model and data status information
    Access this at http://localhost:5000/status
    """
    status_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_loaded': hybrid_model is not None,
        'data_loaded': df is not None
    }

    # Add dataset info if available
    if df is not None:
        status_info['dataset'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB",
            'missing_values': df.isnull().sum().sum(),
            'sample_rows': df.head(5).to_dict(orient='records')
        }

        # Data statistics
        status_info['statistics'] = {
            'airlines': {
                'count': df['Airline'].nunique(),
                'values': df['Airline'].unique().tolist()
            },
            'routes': {
                'count': df['Continent_Route'].nunique(),
                'values': df['Continent_Route'].unique().tolist()
            },
            'classes': {
                'count': df['Class'].nunique(),
                'values': df['Class'].unique().tolist()
            },
            'traveller_types': {
                'count': df['Type of Traveller'].nunique(),
                'values': df['Type of Traveller'].unique().tolist()
            }
        }

        # Rating statistics
        rating_cols = ['Seat Comfort', 'Staff Service', 'Food & Beverages',
                      'Inflight Entertainment', 'Value For Money', 'Overall Rating']

        rating_stats = {}
        for col in rating_cols:
            if col in df.columns:
                rating_stats[col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }

        status_info['rating_statistics'] = rating_stats

    # Add model info if available
    if hybrid_model is not None:
        try:
            status_info['model'] = {
                'type': type(hybrid_model).__name__,
                'features': hybrid_model.feature_names_in_.tolist() if hasattr(hybrid_model, 'feature_names_in_') else [],
                'feature_count': len(hybrid_model.feature_names_in_) if hasattr(hybrid_model, 'feature_names_in_') else 0,
                'parameters': hybrid_model.get_params() if hasattr(hybrid_model, 'get_params') else {}
            }

            # Add feature importance if available
            if hasattr(hybrid_model, 'feature_importances_'):
                feature_importance = hybrid_model.feature_importances_
                if len(feature_importance) > 0:
                    feature_importance_dict = {}
                    for i, importance in enumerate(feature_importance):
                        if i < len(hybrid_model.feature_names_in_):
                            feature_importance_dict[hybrid_model.feature_names_in_[i]] = float(importance)

                    # Sort by importance (descending)
                    sorted_importance = {k: v for k, v in sorted(
                        feature_importance_dict.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )}

                    status_info['model']['feature_importance'] = sorted_importance
        except Exception as e:
            status_info['model_info_error'] = str(e)

    # Add active sessions count
    status_info['active_sessions'] = len(session)

    # Add recent recommendations (store these in a global variable in your app)
    if hasattr(app, 'recent_recommendations'):
        status_info['recent_recommendations'] = app.recent_recommendations

    # Make it pretty for browser viewing
    html_output = "<html><head>"
    html_output += "<title>SkyWings Recommendation System Status</title>"
    html_output += "<style>"
    html_output += "body { font-family: Arial, sans-serif; margin: 20px; }"
    html_output += "h1, h2, h3 { color: #1e40af; }"
    html_output += "pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow: auto; }"
    html_output += "table { border-collapse: collapse; width: 100%; }"
    html_output += "th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }"
    html_output += "th { background-color: #f2f2f2; }"
    html_output += ".container { margin-bottom: 20px; }"
    html_output += ".status-ok { color: green; }"
    html_output += ".status-error { color: red; }"
    html_output += "</style>"
    html_output += "</head><body>"

    # Header
    html_output += "<h1>SkyWings Recommendation System Status</h1>"
    html_output += f"<p>Generated at: {status_info['timestamp']}</p>"

    # System Status
    html_output += "<div class='container'>"
    html_output += "<h2>System Status</h2>"
    html_output += "<table>"
    html_output += "<tr><th>Component</th><th>Status</th></tr>"

    model_status = "OK" if status_info['model_loaded'] else "ERROR"
    model_class = "status-ok" if status_info['model_loaded'] else "status-error"
    html_output += f"<tr><td>Model</td><td class='{model_class}'>{model_status}</td></tr>"

    data_status = "OK" if status_info['data_loaded'] else "ERROR"
    data_class = "status-ok" if status_info['data_loaded'] else "status-error"
    html_output += f"<tr><td>Dataset</td><td class='{data_class}'>{data_status}</td></tr>"

    html_output += f"<tr><td>Active Sessions</td><td>{status_info['active_sessions']}</td></tr>"
    html_output += "</table>"
    html_output += "</div>"

    # Dataset Info
    if 'dataset' in status_info:
        html_output += "<div class='container'>"
        html_output += "<h2>Dataset Information</h2>"
        ds = status_info['dataset']
        html_output += "<table>"
        html_output += f"<tr><td>Rows</td><td>{ds['rows']}</td></tr>"
        html_output += f"<tr><td>Columns</td><td>{ds['columns']}</td></tr>"
        html_output += f"<tr><td>Memory Usage</td><td>{ds['memory_usage']}</td></tr>"
        html_output += f"<tr><td>Missing Values</td><td>{ds['missing_values']}</td></tr>"
        html_output += "</table>"

        # Column names
        html_output += "<h3>Columns</h3>"
        html_output += "<ul>"
        for col in ds['column_names']:
            html_output += f"<li>{col}</li>"
        html_output += "</ul>"

        # Sample data
        html_output += "<h3>Sample Data (5 rows)</h3>"
        html_output += "<pre>" + str(pd.DataFrame(ds['sample_rows']).to_string()) + "</pre>"
        html_output += "</div>"

    # Statistics
    if 'statistics' in status_info:
        html_output += "<div class='container'>"
        html_output += "<h2>Data Statistics</h2>"

        stats = status_info['statistics']
        for category, data in stats.items():
            html_output += f"<h3>{category.title()} ({data['count']})</h3>"
            html_output += "<ul>"
            for value in data['values']:
                html_output += f"<li>{value}</li>"
            html_output += "</ul>"
        html_output += "</div>"

    # Rating statistics
    if 'rating_statistics' in status_info:
        html_output += "<div class='container'>"
        html_output += "<h2>Rating Statistics</h2>"
        html_output += "<table>"
        html_output += "<tr><th>Rating Category</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th></tr>"

        for category, stats in status_info['rating_statistics'].items():
            html_output += "<tr>"
            html_output += f"<td>{category}</td>"
            html_output += f"<td>{stats['mean']:.2f}</td>"
            html_output += f"<td>{stats['median']:.2f}</td>"
            html_output += f"<td>{stats['min']:.2f}</td>"
            html_output += f"<td>{stats['max']:.2f}</td>"
            html_output += "</tr>"

        html_output += "</table>"
        html_output += "</div>"

    # Model Info
    if 'model' in status_info:
        html_output += "<div class='container'>"
        html_output += "<h2>Model Information</h2>"
        model = status_info['model']
        html_output += "<table>"
        html_output += f"<tr><td>Type</td><td>{model['type']}</td></tr>"
        html_output += f"<tr><td>Feature Count</td><td>{model['feature_count']}</td></tr>"
        html_output += "</table>"

        # Model parameters
        html_output += "<h3>Model Parameters</h3>"
        html_output += "<pre>" + str(model['parameters']) + "</pre>"

        # Feature importance
        if 'feature_importance' in model:
            html_output += "<h3>Feature Importance (Top 20)</h3>"
            html_output += "<table>"
            html_output += "<tr><th>Feature</th><th>Importance</th></tr>"

            count = 0
            for feature, importance in model['feature_importance'].items():
                html_output += f"<tr><td>{feature}</td><td>{importance:.6f}</td></tr>"
                count += 1
                if count >= 20:
                    break

            html_output += "</table>"

        # All features
        html_output += "<h3>All Features</h3>"
        html_output += "<ul>"
        for feature in model['features']:
            html_output += f"<li>{feature}</li>"
        html_output += "</ul>"
        html_output += "</div>"

    # Recent recommendations
    if 'recent_recommendations' in status_info:
        html_output += "<div class='container'>"
        html_output += "<h2>Recent Recommendations</h2>"
        html_output += "<pre>" + str(status_info['recent_recommendations']) + "</pre>"
        html_output += "</div>"

    html_output += "</body></html>"

    return html_output

@app.route('/debug/model', methods=['GET', 'POST'])
def debug_model():
    """
    Debug endpoint that allows testing model predictions directly
    Access at http://localhost:5000/debug/model
    """
    prediction_results = None
    form_data = {}

    if request.method == 'POST':
        # Get form data
        form_data = {
            'origin': request.form.get('origin', ''),
            'destination': request.form.get('destination', ''),
            'class': request.form.get('class', 'Any'),
            'traveller_type': request.form.get('traveller_type', 'Any'),
            'season': request.form.get('season', 'Summer')
        }

        try:
            # Create test data for the model
            route = f"{form_data['origin']} to {form_data['destination']}"

            # Filter dataset
            filtered_data = df.copy()
            if form_data['origin'] and form_data['destination']:
                route_data = filtered_data[filtered_data['Continent_Route'].str.contains(route, case=False, na=False)]
                if route_data.empty:
                    route_data = filtered_data[(filtered_data['Continent_Route'].str.contains(form_data['origin'], case=False, na=False)) &
                                             (filtered_data['Continent_Route'].str.contains(form_data['destination'], case=False, na=False))]
                if not route_data.empty:
                    filtered_data = route_data

            if form_data['class'] != 'Any':
                class_data = filtered_data[filtered_data['Class'] == form_data['class']]
                if not class_data.empty:
                    filtered_data = class_data

            if form_data['traveller_type'] != 'Any':
                traveller_data = filtered_data[filtered_data['Type of Traveller'] == form_data['traveller_type']]
                if not traveller_data.empty:
                    filtered_data = traveller_data

            if 'Season' in filtered_data.columns:
                season_data = filtered_data[filtered_data['Season'] == form_data['season']]
                if not season_data.empty:
                    filtered_data = season_data

            # Get unique airline-route combinations
            test_data = filtered_data.drop_duplicates(['Airline', 'Continent_Route', 'Class']).reset_index(drop=True)

            if len(test_data) == 0:
                prediction_results = {
                    'error': 'No matching data found for the criteria. Try different parameters.'
                }
            else:
                # Prepare features for prediction
                features = []
                for _, row in test_data.iterrows():
                    # Create feature row similar to your recommendation endpoint
                    feature_row = {}

                    # Add encoded features
                    for col in ['Airline', 'Continent_Route', 'Class', 'Type of Traveller', 'Season']:
                        encoded_col = f'{col}_encoded'
                        if encoded_col in row:
                            feature_row[encoded_col] = row[encoded_col]

                    # Add rating features
                    rating_cols = ['Seat Comfort', 'Staff Service', 'Food & Beverages',
                                  'Inflight Entertainment', 'Value For Money']
                    for col in rating_cols:
                        if col in row:
                            feature_row[col] = row[col]

                    features.append(feature_row)

                # Convert to DataFrame
                features_df = pd.DataFrame(features)

                # Fill missing values
                for col in features_df.columns:
                    if features_df[col].isnull().any():
                        if col in df.columns:
                            features_df[col] = features_df[col].fillna(df[col].mean())
                        else:
                            features_df[col] = features_df[col].fillna(0)

                # Ensure all model features exist
                if hasattr(hybrid_model, 'feature_names_in_'):
                    for col in hybrid_model.feature_names_in_:
                        if col not in features_df.columns:
                            features_df[col] = 0

                    # Select only features the model was trained on
                    model_features = [col for col in hybrid_model.feature_names_in_ if col in features_df.columns]
                    X_pred = features_df[model_features]

                    if len(model_features) < len(hybrid_model.feature_names_in_):
                        missing_features = set(hybrid_model.feature_names_in_) - set(model_features)
                        print(f"Warning: Missing features: {missing_features}")
                else:
                    X_pred = features_df

                try:
                    # Make predictions
                    predictions = hybrid_model.predict(X_pred)

                    # Add predictions to results
                    test_data['predicted_score'] = predictions

                    # Sort by predicted score
                    results = test_data.sort_values('predicted_score', ascending=False)

                    # Format for display
                    display_results = []
                    for _, row in results.iterrows():
                        display_results.append({
                            'airline': row['Airline'],
                            'route': row['Continent_Route'],
                            'class': row['Class'],
                            'predicted_score': float(row['predicted_score']),
                            'rating': float(row.get('Overall Rating', 0)),
                            'value_for_money': float(row.get('Value For Money', 0))
                        })

                    prediction_results = {
                        'count': len(display_results),
                        'results': display_results
                    }

                except Exception as e:
                    prediction_results = {
                        'error': f"Error making predictions: {str(e)}"
                    }
        except Exception as e:
            prediction_results = {
                'error': f"Error processing request: {str(e)}"
            }

    # Generate HTML
    dropdown_options = get_dropdown_options()

    html_output = "<html><head>"
    html_output += "<title>Model Debug Tool</title>"
    html_output += "<style>"
    html_output += "body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }"
    html_output += "h1, h2 { color: #1e40af; }"
    html_output += "form { background-color: #f0f4f8; padding: 20px; border-radius: 8px; margin-bottom: 20px; }"
    html_output += "label { display: block; margin-bottom: 5px; font-weight: bold; }"
    html_output += "select, input { width: 100%; padding: 8px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 4px; }"
    html_output += "button { background-color: #1e40af; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; }"
    html_output += "button:hover { background-color: #1e3a8a; }"
    html_output += "table { width: 100%; border-collapse: collapse; margin-top: 20px; }"
    html_output += "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }"
    html_output += "th { background-color: #f2f2f2; }"
    html_output += "tr:nth-child(even) { background-color: #f9f9f9; }"
    html_output += ".error { color: red; padding: 10px; background-color: #fff0f0; border-left: 4px solid red; margin-bottom: 20px; }"
    html_output += "</style>"
    html_output += "</head><body>"

    # Header
    html_output += "<h1>Model Debug Tool</h1>"
    html_output += "<p>Use this page to test model predictions directly.</p>"

    # Form
    html_output += "<form method='post'>"
    html_output += "<div>"
    html_output += "<label for='origin'>Origin Continent:</label>"
    html_output += "<select name='origin' id='origin'>"
    html_output += "<option value=''>Select origin</option>"
    for continent in dropdown_options['continents']:
        selected = "selected" if form_data.get('origin') == continent else ""
        html_output += f"<option value='{continent}' {selected}>{continent}</option>"
    html_output += "</select>"
    html_output += "</div>"

    html_output += "<div>"
    html_output += "<label for='destination'>Destination Continent:</label>"
    html_output += "<select name='destination' id='destination'>"
    html_output += "<option value=''>Select destination</option>"
    for continent in dropdown_options['continents']:
        selected = "selected" if form_data.get('destination') == continent else ""
        html_output += f"<option value='{continent}' {selected}>{continent}</option>"
    html_output += "</select>"
    html_output += "</div>"

    html_output += "<div>"
    html_output += "<label for='class'>Travel Class:</label>"
    html_output += "<select name='class' id='class'>"
    html_output += "<option value='Any'>Any Class</option>"
    for travel_class in dropdown_options['classes']:
        selected = "selected" if form_data.get('class') == travel_class else ""
        html_output += f"<option value='{travel_class}' {selected}>{travel_class}</option>"
    html_output += "</select>"
    html_output += "</div>"

    html_output += "<div>"
    html_output += "<label for='traveller_type'>Traveller Type:</label>"
    html_output += "<select name='traveller_type' id='traveller_type'>"
    html_output += "<option value='Any'>Any Type</option>"
    for traveller_type in dropdown_options['traveller_types']:
        selected = "selected" if form_data.get('traveller_type') == traveller_type else ""
        html_output += f"<option value='{traveller_type}' {selected}>{traveller_type}</option>"
    html_output += "</select>"
    html_output += "</div>"

    html_output += "<div>"
    html_output += "<label for='season'>Season:</label>"
    html_output += "<select name='season' id='season'>"
    for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
        selected = "selected" if form_data.get('season') == season else ""
        html_output += f"<option value='{season}' {selected}>{season}</option>"
    html_output += "</select>"
    html_output += "</div>"

    html_output += "<button type='submit'>Test Model Predictions</button>"
    html_output += "</form>"

    # Results
    if prediction_results:
        html_output += "<h2>Prediction Results</h2>"

        if 'error' in prediction_results:
            html_output += f"<div class='error'>{prediction_results['error']}</div>"
        else:
            html_output += f"<p>Found {prediction_results['count']} matching results. "
            html_output += "Below are the results sorted by predicted score:</p>"

            html_output += "<table>"
            html_output += "<tr><th>Rank</th><th>Airline</th><th>Route</th><th>Class</th><th>Predicted Score</th><th>Rating</th><th>Value for Money</th></tr>"

            for i, result in enumerate(prediction_results['results']):
                html_output += "<tr>"
                html_output += f"<td>{i+1}</td>"
                html_output += f"<td>{result['airline']}</td>"
                html_output += f"<td>{result['route']}</td>"
                html_output += f"<td>{result['class']}</td>"
                html_output += f"<td>{result['predicted_score']:.4f}</td>"
                html_output += f"<td>{result['rating']:.1f}</td>"
                html_output += f"<td>{result['value_for_money']:.1f}</td>"
                html_output += "</tr>"

            html_output += "</table>"

    # Links back
    html_output += "<p><a href='/status'>Back to Status Page</a></p>"
    html_output += "</body></html>"

    return html_output

if __name__ == '__main__':
    app.run(debug=True)
