from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import csv
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load player data
player_df = pd.read_csv('players.csv')
history_df = pd.read_csv('player_history.csv') if os.path.exists('player_history.csv') else pd.DataFrame(
    columns=['ID', 'Name', 'Timestamp', 'Training_Load', 'Heart_Rate'])


# Load or initialize user preferences
def load_preferences():
    if os.path.exists('user_preferences.csv'):
        return pd.read_csv('user_preferences.csv')
    return pd.DataFrame(columns=['username', 'show_graph1', 'show_graph2', 'show_graph3', 'show_graph4'])


user_preferences_df = load_preferences()

# Custom Plotly template for dashboard styling
custom_template = {
    'layout': {
        'font': {'family': 'Arial, sans-serif', 'color': '#1F2937'},
        'plot_bgcolor': '#F3F4F6',
        'paper_bgcolor': '#FFFFFF',
        'colorscale': {'sequential': [[0, '#2563EB'], [1, '#60A5FA']]},
        'xaxis': {'gridcolor': '#E5E7EB'},
        'yaxis': {'gridcolor': '#E5E7EB'},
    }
}


# Initialize and train an enhanced injury prediction model
def train_injury_model():
    features = player_df[['training_load', 'heart_rate', 'past_injuries', 'age', 'Weight', 'Height']]
    labels = player_df['injury_type']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features_scaled, labels)
    return model, scaler


# Initialize and train a training adjustment model
def train_adjustment_model():
    if history_df.empty:
        return None, None
    features = []
    labels = []
    for player_id in history_df['ID'].unique():
        player_history = history_df[history_df['ID'] == player_id].sort_values('Timestamp')
        if len(player_history) >= 3:
            for i in range(2, len(player_history)):
                recent = player_history.iloc[i - 2:i]
                current = player_history.iloc[i]
                player_data = player_df[player_df['ID'] == player_id].iloc[0]
                feat = [
                    current['Training_Load'],
                    current['Heart_Rate'],
                    player_data['past_injuries'],
                    player_data['age'],
                    player_data['Weight'],
                    player_data['Height'],
                    recent['Training_Load'].mean(),
                    recent['Heart_Rate'].mean()
                ]
                next_load = player_history.iloc[i + 1]['Training_Load'] if i + 1 < len(player_history) else current[
                    'Training_Load']
                label = next_load - current['Training_Load']
                features.append(feat)
                labels.append(label)
    if not features:
        return None, None
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features_scaled, labels)
    return model, scaler


# Load models and scalers
injury_model, injury_scaler = train_injury_model()
adjustment_model, adjustment_scaler = train_adjustment_model()


# Function to predict injury risk with probabilities
def predict_injury_risk(player_data):
    features = np.array([
        player_data['training_load'],
        player_data['heart_rate'],
        player_data['past_injuries'],
        player_data['age'],
        player_data['Weight'],
        player_data['Height']
    ]).reshape(1, -1)
    features_scaled = injury_scaler.transform(features)
    prediction = injury_model.predict(features_scaled)[0]
    probabilities = injury_model.predict_proba(features_scaled)[0]
    injury_prob = max(probabilities) * 100
    return prediction, injury_prob


# Function to predict training load adjustment
def predict_training_adjustment(player_data, player_history):
    if adjustment_model is None or adjustment_scaler is None:
        return 0.0
    recent = player_history.tail(2)
    if len(recent) < 2:
        return 0.0
    features = np.array([
        player_data['training_load'],
        player_data['heart_rate'],
        player_data['past_injuries'],
        player_data['age'],
        player_data['Weight'],
        player_data['Height'],
        recent['Training_Load'].mean(),
        recent['Heart_Rate'].mean()
    ]).reshape(1, -1)
    features_scaled = adjustment_scaler.transform(features)
    adjustment = adjustment_model.predict(features_scaled)[0]
    return adjustment


# Generate player-specific graphs
def generate_player_graphs(player, username):
    global user_preferences_df
    prefs = user_preferences_df[user_preferences_df['username'] == username]
    show_graph1 = prefs['show_graph1'].iloc[0] if not prefs.empty else True
    show_graph2 = prefs['show_graph2'].iloc[0] if not prefs.empty else True
    show_graph3 = prefs['show_graph3'].iloc[0] if not prefs.empty else True
    show_graph4 = prefs['show_graph4'].iloc[0] if not prefs.empty else True

    # Graph 1: Scatter Plot with Dropdown Filter
    if show_graph1:
        fig1 = go.Figure()
        # All players
        fig1.add_trace(go.Scatter(
            x=player_df['training_load'],
            y=player_df['heart_rate'],
            mode='markers',
            marker=dict(size=8, color='#60A5FA', opacity=0.5),
            text=player_df['Name'],
            hovertemplate='%{text}<br>Training Load: %{x}<br>Heart Rate: %{y}',
            name='All Players'
        ))
        # Highlight searched player
        fig1.add_trace(go.Scatter(
            x=[player['training_load']],
            y=[player['heart_rate']],
            mode='markers+text',
            marker=dict(size=15, color='#2563EB', symbol='star'),
            text=[player['Name']],
            textposition='top center',
            hovertemplate='%{text}<br>Training Load: %{x}<br>Heart Rate: %{y}',
            name=player['Name']
        ))
        # Add trend line
        z = np.polyfit(player_df['training_load'], player_df['heart_rate'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(player_df['training_load'].min(), player_df['training_load'].max(), 100)
        fig1.add_trace(go.Scatter(
            x=x_range,
            y=p(x_range),
            mode='lines',
            line=dict(color='#1F2937', dash='dash'),
            name='Trend Line'
        ))
        # Add dropdown for filtering
        injury_types = player_df['injury_type'].unique().tolist()
        age_groups = ['All', '<20', '20-25', '25-30', '30+']
        buttons = [
            dict(
                label='All',
                method='update',
                args=[{
                    'x': [player_df['training_load']],
                    'y': [player_df['heart_rate']],
                    'text': [player_df['Name']],
                    'visible': [True, True, True]
                }]
            )
        ]
        for injury in injury_types:
            mask = player_df['injury_type'] == injury
            buttons.append(dict(
                label=f'Injury: {injury}',
                method='update',
                args=[{
                    'x': [player_df[mask]['training_load']],
                    'y': [player_df[mask]['heart_rate']],
                    'text': [player_df[mask]['Name']],
                    'visible': [True, True, True]
                }]
            ))
        for age_group in age_groups[1:]:
            if age_group == '<20':
                mask = player_df['age'] < 20
            elif age_group == '20-25':
                mask = (player_df['age'] >= 20) & (player_df['age'] <= 25)
            elif age_group == '25-30':
                mask = (player_df['age'] > 25) & (player_df['age'] <= 30)
            else:
                mask = player_df['age'] > 30
            buttons.append(dict(
                label=f'Age: {age_group}',
                method='update',
                args=[{
                    'x': [player_df[mask]['training_load']],
                    'y': [player_df[mask]['heart_rate']],
                    'text': [player_df[mask]['Name']],
                    'visible': [True, True, True]
                }]
            ))
        fig1.update_layout(
            title='Training Load vs. Heart Rate (Player Highlighted)',
            xaxis_title='Training Load (Normalized)',
            yaxis_title='Heart Rate (bpm)',
            height=500,
            template=custom_template,
            showlegend=True,
            hovermode='closest',
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction='down',
                    showactive=True,
                    x=0.1,
                    xanchor='left',
                    y=1.2,
                    yanchor='top'
                )
            ]
        )
    else:
        fig1 = None

    # Graph 2: Box Plot with Player Marker
    if show_graph2:
        fig2 = go.Figure()
        for injury in player_df['injury_type'].unique():
            subset = player_df[player_df['injury_type'] == injury]
            fig2.add_trace(go.Box(
                y=subset['training_load'],
                name=injury,
                marker_color='#60A5FA',
                boxpoints='outliers',
                hovertemplate='Training Load: %{y}<br>Injury: %{x}'
            ))
        # Add player's training load marker
        player_injury = player['injury_type']
        fig2.add_trace(go.Scatter(
            x=[player_injury],
            y=[player['training_load']],
            mode='markers+text',
            marker=dict(size=12, color='#2563EB', symbol='diamond'),
            text=[player['Name']],
            textposition='top center',
            hovertemplate='%{text}<br>Training Load: %{y}',
            name=player['Name']
        ))
        fig2.update_layout(
            title='Training Load by Injury Type (Player Marker)',
            xaxis_title='Injury Type',
            yaxis_title='Training Load (Normalized, Log Scale)',
            yaxis_type='log',
            height=500,
            template=custom_template,
            showlegend=True
        )
    else:
        fig2 = None

    # Graph 3: Historical Trends with Clickable Points
    player_history = history_df[history_df['ID'] == player['ID']]
    if not player_history.empty and show_graph3:
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        # Training Load
        fig3.add_trace(go.Scatter(
            x=player_history['Timestamp'],
            y=player_history['Training_Load'],
            mode='lines+markers',
            name='Training Load',
            line=dict(color='#2563EB'),
            marker=dict(size=8),
            customdata=player_history[['Training_Load', 'Heart_Rate', 'Timestamp']].values,
            hovertemplate='Date: %{x}<br>Training Load: %{y}'
        ), secondary_y=False)
        # Heart Rate
        fig3.add_trace(go.Scatter(
            x=player_history['Timestamp'],
            y=player_history['Heart_Rate'],
            mode='lines+markers',
            name='Heart Rate',
            line=dict(color='#F59E0B'),
            marker=dict(size=8),
            customdata=player_history[['Training_Load', 'Heart_Rate', 'Timestamp']].values,
            hovertemplate='Date: %{x}<br>Heart Rate: %{y}'
        ), secondary_y=True)
        # Add annotations for high-risk points
        for idx, row in player_history.iterrows():
            if row['Training_Load'] > 1.5 or row['Heart_Rate'] > 150:
                fig3.add_annotation(
                    x=row['Timestamp'],
                    y=row['Training_Load'],
                    text='High Risk',
                    showarrow=True,
                    arrowhead=2,
                    ax=20,
                    ay=-30,
                    font=dict(color='#DC2626'),
                    secondary_y=False
                )
        fig3.update_layout(
            title=f"Historical Trends for {player['Name']}",
            xaxis_title='Date',
            height=500,
            template=custom_template,
            showlegend=True,
            hovermode='x unified',
            clickmode='event+select'
        )
        fig3.update_yaxes(title_text='Training Load (Normalized)', secondary_y=False)
        fig3.update_yaxes(title_text='Heart Rate (bpm)', secondary_y=True)
        # Add JavaScript for click event
        fig3_html = pio.to_html(fig3, full_html=False, include_plotlyjs=False)
        fig3_html += """
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                var plot = document.querySelector('.js-plotly-plot');
                if (plot) {
                    plot.on('plotly_click', function(data) {
                        var point = data.points[0];
                        var trainingLoad = point.customdata[0];
                        var heartRate = point.customdata[1];
                        var date = point.customdata[2];
                        var details = `Date: ${date}<br>Training Load: ${trainingLoad.toFixed(2)}<br>Heart Rate: ${heartRate}`;
                        document.getElementById('line-plot-details').innerHTML = details;
                    });
                }
            });
        </script>
        """
    else:
        fig3 = None
        fig3_html = None

    # Graph 4: Gauge Chart for Injury Probability
    if show_graph4:
        injury_prob = float(player['injury_probability'].strip('%'))
        fig4 = go.Figure(go.Indicator(
            mode='gauge+number',
            value=injury_prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': 'Injury Risk Probability'},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#1F2937'},
                'bar': {'color': '#2563EB'},
                'bgcolor': '#F3F4F6',
                'borderwidth': 2,
                'bordercolor': '#E5E7EB',
                'steps': [
                    {'range': [0, 33], 'color': '#22C55E'},
                    {'range': [33, 66], 'color': '#F59E0B'},
                    {'range': [66, 100], 'color': '#DC2626'}
                ],
                'threshold': {
                    'line': {'color': '#1F2937', 'width': 4},
                    'thickness': 0.75,
                    'value': injury_prob
                }
            }
        ))
        fig4.update_layout(
            height=400,
            template=custom_template,
            margin=dict(l=20, r=20, t=50, b=20)
        )
    else:
        fig4 = None

    fig1_html = pio.to_html(fig1, full_html=False) if fig1 else None
    fig2_html = pio.to_html(fig2, full_html=False) if fig2 else None
    fig4_html = pio.to_html(fig4, full_html=False) if fig4 else None

    return fig1_html, fig2_html, fig3_html, fig4_html


# Generate heatmap for training load vs. heart rate vs. injury type
def generate_heatmap():
    # Bin training load and heart rate
    bins_tl = pd.cut(player_df['training_load'], bins=10, labels=False)
    bins_hr = pd.cut(player_df['heart_rate'], bins=10, labels=False)
    heatmap_data = pd.crosstab([bins_tl, bins_hr], player_df['injury_type'])
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=[f'TL:{i}-HR:{j}' for i, j in zip(range(10), range(10))],
        colorscale=[[0, '#F3F4F6'], [0.5, '#60A5FA'], [1, '#2563EB']],
        hovertemplate='Injury: %{x}<br>Bin: %{y}<br>Count: %{z}<extra></extra>'
    ))
    fig.update_layout(
        title='Heatmap of Training Load and Heart Rate vs. Injury Type',
        xaxis_title='Injury Type',
        yaxis_title='Training Load and Heart Rate Bins',
        height=500,
        template=custom_template
    )
    return pio.to_html(fig, full_html=False)


# Enhanced Real-Time Recommendations Engine
def get_real_time_recommendations(player_data, injury_type, player_history):
    recommendations = []
    injury_based_recommendations = {
        'Muscle Injury': 'Incorporate dynamic stretching and ensure adequate recovery time.',
        'Knee Injury': 'Perform knee strengthening exercises and use supportive gear.',
        'Ankle Sprain': 'Use proper footwear and practice ankle mobility drills.',
        'Hamstring Strain': 'Implement gradual intensity increases and thorough warm-ups.',
        'none': 'Maintain current training with regular monitoring.'
    }
    recommendations.append(injury_based_recommendations.get(injury_type, 'Rest and monitor training load.'))
    heart_rate = player_data['heart_rate']
    if heart_rate > 150:
        recommendations.append('High heart rate detected: Reduce training intensity immediately.')
    elif heart_rate > 130:
        recommendations.append('Elevated heart rate: Consider lighter training or additional rest.')
    training_load = player_data['training_load']
    if training_load > 1.5:
        recommendations.append('High training load: Schedule a recovery session to prevent overtraining.')
    elif training_load > 1.0:
        recommendations.append('Moderate training load: Monitor for fatigue and adjust as needed.')
    past_injuries = player_data['past_injuries']
    if past_injuries > 20:
        recommendations.append('History of frequent injuries: Prioritize low-impact training and physiotherapy.')
    elif past_injuries > 10:
        recommendations.append('Moderate injury history: Include injury prevention exercises in routine.')
    age = player_data['age']
    if age > 30:
        recommendations.append('Older athlete: Increase recovery time between intense sessions.')
    elif age < 20:
        recommendations.append('Young athlete: Focus on technique and avoid excessive load.')
    weight = player_data['Weight']
    if weight > 85:
        recommendations.append('Higher weight: Opt for low-impact exercises to reduce joint stress.')
    elif weight < 65:
        recommendations.append('Lower weight: Ensure adequate nutrition to support training demands.')
    if not player_history.empty and len(player_history) >= 3:
        recent = player_history.tail(3)
        avg_training_load = recent['Training_Load'].mean()
        avg_heart_rate = recent['Heart_Rate'].mean()
        if avg_training_load > 1.2:
            recommendations.append('Recent high training load trend: Consider a rest day to prevent fatigue.')
        if avg_heart_rate > 140:
            recommendations.append('Recent high heart rate trend: Monitor for overtraining and reduce intensity.')
    adjustment = predict_training_adjustment(player_data, player_history)
    if adjustment > 0.2:
        recommendations.append(
            f'ML Suggestion: Increase training load by {adjustment:.2f} units for optimal performance.')
    elif adjustment < -0.2:
        recommendations.append(f'ML Suggestion: Decrease training load by {-adjustment:.2f} units to reduce strain.')
    else:
        recommendations.append('ML Suggestion: Maintain current training load for balanced performance.')
    return recommendations


@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    user_role = session.get('role', 'User')
    if user_role not in ['User', 'Coach', 'Admin']:
        return render_template('index.html', error="Unauthorized access", user_role=user_role)

    # Initialize variables
    player = None
    fig1_html = fig2_html = fig3_html = fig4_html = None
    player_info = None
    heatmap_html = generate_heatmap()

    # Filter and pagination parameters
    injury_filter = request.args.get('injury_filter', 'All')
    age_filter = request.args.get('age_filter', 'All')
    page = int(request.args.get('page', 1))
    players_per_page = 20

    # Apply filters
    filtered_df = player_df.copy()
    if injury_filter != 'All':
        filtered_df = filtered_df[filtered_df['injury_type'] == injury_filter]
    if age_filter != 'All':
        if age_filter == '<20':
            filtered_df = filtered_df[filtered_df['age'] < 20]
        elif age_filter == '20-25':
            filtered_df = filtered_df[(filtered_df['age'] >= 20) & (filtered_df['age'] <= 25)]
        elif age_filter == '25-30':
            filtered_df = filtered_df[(filtered_df['age'] > 25) & (filtered_df['age'] <= 30)]
        else:  # 30+
            filtered_df = filtered_df[filtered_df['age'] > 30]

    # Calculate injury probabilities and border colors
    players_data = []
    for _, row in filtered_df.iterrows():
        _, injury_prob = predict_injury_risk(row)
        if injury_prob < 33:
            border_color = 'border-green-500'
        elif injury_prob < 66:
            border_color = 'border-yellow-500'
        else:
            border_color = 'border-red-500'
        players_data.append({
            'name': row['Name'],
            'border_color': border_color
        })

    # Pagination
    total_players = len(players_data)
    total_pages = (total_players + players_per_page - 1) // players_per_page
    page = max(1, min(page, total_pages))
    start_idx = (page - 1) * players_per_page
    end_idx = start_idx + players_per_page
    paginated_players = players_data[start_idx:end_idx]

    # Filter options
    injury_types = ['All'] + player_df['injury_type'].unique().tolist()
    age_groups = ['All', '<20', '20-25', '25-30', '30+']

    # Handle player selection (search or box click)
    if request.method == 'POST':
        name = request.form['player_name']
        player_data = player_df[player_df['Name'].str.lower() == name.lower()]
        if not player_data.empty:
            player = player_data.iloc[0].to_dict()
            player_history = history_df[history_df['ID'] == player['ID']]
            injury_type, injury_prob = predict_injury_risk(player)
            recommendations = get_real_time_recommendations(player, injury_type, player_history)
            player['predicted_injury'] = injury_type
            player['injury_probability'] = f"{injury_prob:.2f}%"
            player['recommendations'] = recommendations
            fig1_html, fig2_html, fig3_html, fig4_html = generate_player_graphs(player, session['username'])
            player_info = {
                'ID': player['ID'],
                'Name': player['Name'],
                'Training Load': player['training_load'],
                'Heart Rate': player['heart_rate'],
                'Past Injuries': player['past_injuries'],
                'Age': player['age'],
                'Weight': player['Weight'],
                'Height': player['Height'],
                'Predicted Injury': player['predicted_injury'],
                'Injury Probability': player['injury_probability'],
                'Recommendations': player['recommendations']
            }
            return render_template(
                'index.html',
                player_info=player_info,
                fig1_html=fig1_html,
                fig2_html=fig2_html,
                fig3_html=fig3_html,
                fig4_html=fig4_html,
                heatmap_html=heatmap_html,
                user_role=user_role,
                players=paginated_players,
                injury_types=injury_types,
                age_groups=age_groups,
                current_injury_filter=injury_filter,
                current_age_filter=age_filter,
                current_page=page,
                total_pages=total_pages
            )
        else:
            return render_template(
                'index.html',
                error="Player not found",
                user_role=user_role,
                heatmap_html=heatmap_html,
                players=paginated_players,
                injury_types=injury_types,
                age_groups=age_groups,
                current_injury_filter=injury_filter,
                current_age_filter=age_filter,
                current_page=page,
                total_pages=total_pages
            )

    return render_template(
        'index.html',
        fig1_html=fig1_html,
        fig2_html=fig2_html,
        fig3_html=fig3_html,
        fig4_html=fig4_html,
        heatmap_html=heatmap_html,
        user_role=user_role,
        players=paginated_players,
        injury_types=injury_types,
        age_groups=age_groups,
        current_injury_filter=injury_filter,
        current_age_filter=age_filter,
        current_page=page,
        total_pages=total_pages
    )


@app.route('/update_player_data', methods=['POST'])
def update_player_data():
    if 'username' not in session:
        return redirect(url_for('login'))
    user_role = session.get('role', 'User')
    if user_role not in ['Coach', 'Admin']:
        return render_template('index.html', error="Unauthorized: Only Coaches and Admins can update data",
                               user_role=user_role)
    player_id = request.form['player_id']
    training_load = float(request.form['training_load'])
    heart_rate = int(request.form['heart_rate'])
    player_index = player_df[player_df['ID'] == int(player_id)].index
    if not player_index.empty:
        player_df.loc[player_index, 'training_load'] = training_load
        player_df.loc[player_index, 'heart_rate'] = heart_rate
        player_df.to_csv('players.csv', index=False)
        global history_df
        player = player_df.loc[player_index].iloc[0]
        new_entry = {
            'ID': player['ID'],
            'Name': player['Name'],
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Training_Load': training_load,
            'Heart_Rate': heart_rate
        }
        history_df = pd.concat([history_df, pd.DataFrame([new_entry])], ignore_index=True)
        history_df.to_csv('player_history.csv', index=False)
        player_history = history_df[history_df['ID'] == player['ID']]
        injury_type, injury_prob = predict_injury_risk(player)
        recommendations = get_real_time_recommendations(player, injury_type, player_history)
        player['predicted_injury'] = injury_type
        player['injury_probability'] = f"{injury_prob:.2f}%"
        player['recommendations'] = recommendations
        fig1_html, fig2_html, fig3_html, fig4_html = generate_player_graphs(player, session['username'])
        heatmap_html = generate_heatmap()
        player_info = {
            'ID': player['ID'],
            'Name': player['Name'],
            'Training Load': player['training_load'],
            'Heart Rate': player['heart_rate'],
            'Past Injuries': player['past_injuries'],
            'Age': player['age'],
            'Weight': player['Weight'],
            'Height': player['Height'],
            'Predicted Injury': player['predicted_injury'],
            'Injury Probability': player['injury_probability'],
            'Recommendations': player['recommendations']
        }
        # Get filter and pagination parameters
        injury_filter = request.args.get('injury_filter', 'All')
        age_filter = request.args.get('age_filter', 'All')
        page = int(request.args.get('page', 1))
        players_per_page = 20

        # Apply filters
        filtered_df = player_df.copy()
        if injury_filter != 'All':
            filtered_df = filtered_df[filtered_df['injury_type'] == injury_filter]
        if age_filter != 'All':
            if age_filter == '<20':
                filtered_df = filtered_df[filtered_df['age'] < 20]
            elif age_filter == '20-25':
                filtered_df = filtered_df[(filtered_df['age'] >= 20) & (filtered_df['age'] <= 25)]
            elif age_filter == '25-30':
                filtered_df = filtered_df[(filtered_df['age'] > 25) & (filtered_df['age'] <= 30)]
            else:  # 30+
                filtered_df = filtered_df[filtered_df['age'] > 30]

        # Calculate injury probabilities and border colors
        players_data = []
        for _, row in filtered_df.iterrows():
            _, injury_prob = predict_injury_risk(row)
            if injury_prob < 33:
                border_color = 'border-green-500'
            elif injury_prob < 66:
                border_color = 'border-yellow-500'
            else:
                border_color = 'border-red-500'
            players_data.append({
                'name': row['Name'],
                'border_color': border_color
            })

        # Pagination
        total_players = len(players_data)
        total_pages = (total_players + players_per_page - 1) // players_per_page
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * players_per_page
        end_idx = start_idx + players_per_page
        paginated_players = players_data[start_idx:end_idx]

        # Filter options
        injury_types = ['All'] + player_df['injury_type'].unique().tolist()
        age_groups = ['All', '<20', '20-25', '25-30', '30+']

        return render_template(
            'index.html',
            player_info=player_info,
            fig1_html=fig1_html,
            fig2_html=fig2_html,
            fig3_html=fig3_html,
            fig4_html=fig4_html,
            heatmap_html=heatmap_html,
            user_role=user_role,
            players=paginated_players,
            injury_types=injury_types,
            age_groups=age_groups,
            current_injury_filter=injury_filter,
            current_age_filter=age_filter,
            current_page=page,
            total_pages=total_pages
        )
    else:
        # Get filter and pagination parameters
        injury_filter = request.args.get('injury_filter', 'All')
        age_filter = request.args.get('age_filter', 'All')
        page = int(request.args.get('page', 1))
        players_per_page = 20

        # Apply filters
        filtered_df = player_df.copy()
        if injury_filter != 'All':
            filtered_df = filtered_df[filtered_df['injury_type'] == injury_filter]
        if age_filter != 'All':
            if age_filter == '<20':
                filtered_df = filtered_df[filtered_df['age'] < 20]
            elif age_filter == '20-25':
                filtered_df = filtered_df[(filtered_df['age'] >= 20) & (filtered_df['age'] <= 25)]
            elif age_filter == '25-30':
                filtered_df = filtered_df[(filtered_df['age'] > 25) & (filtered_df['age'] <= 30)]
            else:  # 30+
                filtered_df = filtered_df[filtered_df['age'] > 30]

        # Calculate injury probabilities and border colors
        players_data = []
        for _, row in filtered_df.iterrows():
            _, injury_prob = predict_injury_risk(row)
            if injury_prob < 33:
                border_color = 'border-green-500'
            elif injury_prob < 66:
                border_color = 'border-yellow-500'
            else:
                border_color = 'border-red-500'
            players_data.append({
                'name': row['Name'],
                'border_color': border_color
            })

        # Pagination
        total_players = len(players_data)
        total_pages = (total_players + players_per_page - 1) // players_per_page
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * players_per_page
        end_idx = start_idx + players_per_page
        paginated_players = players_data[start_idx:end_idx]

        # Filter options
        injury_types = ['All'] + player_df['injury_type'].unique().tolist()
        age_groups = ['All', '<20', '20-25', '25-30', '30+']

        return render_template(
            'index.html',
            error="Player not found",
            user_role=user_role,
            heatmap_html=generate_heatmap(),
            players=paginated_players,
            injury_types=injury_types,
            age_groups=age_groups,
            current_injury_filter=injury_filter,
            current_age_filter=age_filter,
            current_page=page,
            total_pages=total_pages
        )


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if os.path.exists('users.csv'):
            with open('users.csv', 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row and row[0] == username and row[1] == password:
                        session['username'] = username
                        session['role'] = row[2]
                        return redirect(url_for('index'))
        return render_template('login.html', error='Invalid Credentials')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form.get('role', 'User')
        if not os.path.exists('users.csv'):
            with open('users.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['username', 'password', 'role'])
                writer.writerow([username, password, role])
        else:
            with open('users.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([username, password, role])
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/help')
def help():
    if 'username' not in session:
        return redirect(url_for('login'))
    user_role = session.get('role', 'User')
    return render_template('help.html', user_role=user_role)


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'username' not in session:
        return redirect(url_for('login'))
    user_role = session.get('role', 'User')
    if user_role not in ['Coach', 'Admin']:
        return render_template('index.html', error="Unauthorized: Only Coaches and Admins can access settings",
                               user_role=user_role)

    global user_preferences_df
    users = pd.read_csv('users.csv') if os.path.exists('users.csv') else pd.DataFrame(
        columns=['username', 'password', 'role'])

    if request.method == 'POST':
        if 'add_user' in request.form:
            username = request.form['username']
            password = request.form['password']
            role = request.form['role']
            if user_role == 'Admin' and username not in users['username'].values:
                with open('users.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([username, password, role])
                users = pd.read_csv('users.csv')
        elif 'delete_user' in request.form:
            username = request.form['delete_username']
            if user_role == 'Admin' and username != session['username']:
                users = users[users['username'] != username]
                users.to_csv('users.csv', index=False)
        elif 'update_preferences' in request.form:
            username = session['username']
            show_graph1 = 'show_graph1' in request.form
            show_graph2 = 'show_graph2' in request.form
            show_graph3 = 'show_graph3' in request.form
            show_graph4 = 'show_graph4' in request.form
            user_prefs = user_preferences_df[user_preferences_df['username'] == username]
            if user_prefs.empty:
                new_prefs = pd.DataFrame([{
                    'username': username,
                    'show_graph1': show_graph1,
                    'show_graph2': show_graph2,
                    'show_graph3': show_graph3,
                    'show_graph4': show_graph4
                }])
                user_preferences_df = pd.concat([user_preferences_df, new_prefs], ignore_index=True)
            else:
                user_preferences_df.loc[
                    user_preferences_df['username'] == username, ['show_graph1', 'show_graph2', 'show_graph3',
                                                                  'show_graph4']] = [show_graph1, show_graph2,
                                                                                     show_graph3, show_graph4]
            user_preferences_df.to_csv('user_preferences.csv', index=False)

    prefs = user_preferences_df[user_preferences_df['username'] == session['username']]
    preferences = {
        'show_graph1': prefs['show_graph1'].iloc[0] if not prefs.empty else True,
        'show_graph2': prefs['show_graph2'].iloc[0] if not prefs.empty else True,
        'show_graph3': prefs['show_graph3'].iloc[0] if not prefs.empty else True,
        'show_graph4': prefs['show_graph4'].iloc[0] if not prefs.empty else True
    }
    return render_template('settings.html', user_role=user_role, users=users.to_dict('records'),
                           preferences=preferences)


@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)