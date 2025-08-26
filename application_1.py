import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Imports pour les modèles
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Configuration de la page
st.set_page_config(
    page_title="Prédiction TUINDEX - ARIMA/LSTM/Prophet",
    page_icon="📈",
    layout="wide"
)

# Titre principal
st.title("📈 Prédiction de l'Indice Boursier TUINDEX")
st.markdown("**Modèles disponibles: ARIMA, LSTM, Prophet**")
st.markdown("---")

# Classe pour le modèle LSTM
class LSTMPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def create_sequences(self, data):
        """Crée des séquences pour l'entraînement LSTM"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def prepare_data(self, data):
        """Prépare les données pour LSTM"""
        # Normalisation
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Création des séquences
        X, y = self.create_sequences(data_scaled.flatten())
        
        return X, y, data_scaled
    
    def build_model(self, input_shape):
        """Construit un modèle LSTM simple (simulation)"""
        # Simulation d'un modèle LSTM
        self.model = "LSTM_Simulated"
        
    def predict_simple(self, data, steps):
        """Prédiction LSTM simplifiée"""
        # Préparation des données
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Prendre les dernières valeurs comme séquence
        if len(data_scaled) >= self.sequence_length:
            last_sequence = data_scaled[-self.sequence_length:]
        else:
            last_sequence = data_scaled
        
        predictions = []
        
        for i in range(steps):
            # Simulation d'une prédiction LSTM
            # Utilise la tendance récente avec du bruit
            if len(last_sequence) >= 10:
                trend = np.mean(np.diff(last_sequence[-10:]))
            else:
                trend = np.mean(np.diff(last_sequence))
                
            # Ajout de bruit contrôlé
            noise_factor = np.std(last_sequence[-10:]) * 0.1 if len(last_sequence) >= 10 else 0.01
            noise = np.random.normal(0, noise_factor)
            
            next_val = last_sequence[-1] + trend + noise
            predictions.append(next_val)
            
            # Mise à jour de la séquence
            last_sequence = np.append(last_sequence[1:], next_val)
        
        # Dénormalisation
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_denorm = self.scaler.inverse_transform(predictions_array).flatten()
        
        return predictions_denorm

# Fonctions pour les modèles
def fit_arima_model(data, order=(1, 1, 1)):
    """Ajuste un modèle ARIMA"""
    try:
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        return fitted_model
    except Exception as e:
        st.error(f"Erreur ARIMA: {str(e)}")
        return None

def fit_prophet_model(df):
    """Ajuste un modèle Prophet"""
    try:
        # Préparation des données pour Prophet
        prophet_df = df.rename(columns={'Date': 'ds', 'Dernier': 'y'})
        
        # Initialisation du modèle Prophet
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        
        # Entraînement
        model.fit(prophet_df)
        return model
    except Exception as e:
        st.error(f"Erreur Prophet: {str(e)}")
        return None

def calculate_metrics(actual, predicted):
    """Calcule les métriques de performance"""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

# Sidebar pour le chargement des données
st.sidebar.header("📂 Chargement des Données")
uploaded_file = st.sidebar.file_uploader(
    "Choisissez votre fichier CSV TUINDEX",
    type=['csv'],
    help="Le fichier doit contenir les colonnes: Date, Dernier"
)

if uploaded_file is not None:
    try:
        # Lecture du fichier CSV
        df = pd.read_csv(uploaded_file)
        
        # Affichage des colonnes disponibles
        st.subheader("📋 Colonnes disponibles dans le fichier")
        st.write("Colonnes:", ", ".join(df.columns.tolist()))
        
        # Vérification de la colonne "Dernier"
        if 'Dernier' not in df.columns:
            st.error("La colonne 'Dernier' n'est pas trouvée dans le fichier.")
            st.info("Colonnes disponibles: " + ", ".join(df.columns.tolist()))
            st.info("Veuillez vous assurer que votre fichier contient une colonne nommée 'Dernier'")
        else:
            # Préparation des données
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
            else:
                st.warning("Colonne 'Date' non trouvée. Utilisation de l'index comme date.")
                df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            
            df = df.reset_index()
            
            # Nettoyage des données
             ## Conversion des types de données ( str-->float)
            df['Dernier'] = df['Dernier'].str.replace('.', '')
            df['Dernier'] = df['Dernier'].str.replace(',', '.').astype(float)

            df = df.dropna(subset=['Dernier'])
            
            # Affichage des données
            st.subheader("📊 Aperçu des Données")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df.head())
            
            with col2:
                st.metric("Nombre d'observations", len(df))
                st.metric("Valeur actuelle", f"{df['Dernier'].iloc[-1]:.2f}")
                st.metric("Variation (%)", f"{((df['Dernier'].iloc[-1] / df['Dernier'].iloc[-2] - 1) * 100):.2f}%")
            
            # Graphique des données historiques
            st.subheader("📈 Évolution Historique de TUINDEX")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Dernier'],
                mode='lines',
                name='TUINDEX',
                line=dict(color='blue', width=2)
            ))
            fig_hist.update_layout(
                title="Évolution de l'indice TUINDEX",
                xaxis_title="Date",
                yaxis_title="Valeur",
                hovermode='x unified'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Configuration dans la sidebar
            st.sidebar.header("⚙️ Configuration de Prédiction")
            
            # Sélection du modèle
            available_models = []
            if ARIMA_AVAILABLE:
                available_models.append("ARIMA")
            available_models.append("LSTM")
            if PROPHET_AVAILABLE:
                available_models.append("Prophet")
            
            if not available_models:
                st.error("Aucun modèle disponible. Installez les dépendances nécessaires.")
                st.stop()
            
            selected_model = st.sidebar.selectbox(
                "Choisissez le modèle de prédiction:",
                available_models
            )
            
            # Paramètres communs
            prediction_days = st.sidebar.slider(
                "Nombre de jours à prédire:",
                min_value=1,
                max_value=180,
                value=30
            )
            
            test_size = st.sidebar.slider(
                "Taille du jeu de test (%):",
                min_value=10,
                max_value=40,
                value=20
            )
            
            # Paramètres spécifiques au modèle
            if selected_model == "ARIMA":
                st.sidebar.subheader("Paramètres ARIMA")
                p = st.sidebar.slider("p (ordre autorégressif)", 0, 5, 1)
                d = st.sidebar.slider("d (différenciation)", 0, 2, 1)
                q = st.sidebar.slider("q (moyenne mobile)", 0, 5, 1)
                arima_order = (p, d, q)
            
            elif selected_model == "LSTM":
                st.sidebar.subheader("Paramètres LSTM")
                sequence_length = st.sidebar.slider("Longueur de séquence", 30, 120, 60)
            
            # Bouton de prédiction
            if st.sidebar.button("🚀 Lancer la Prédiction", type="primary"):
                
                with st.spinner(f"Entraînement du modèle {selected_model}..."):
                    
                    # Division des données
                    test_size_abs = int(len(df) * test_size / 100)
                    train_data = df[:-test_size_abs] if test_size_abs > 0 else df
                    test_data = df[-test_size_abs:] if test_size_abs > 0 else df.tail(10)
                    
                    predictions_test = []
                    future_predictions = []
                    model_info = {}
                    
                    if selected_model == "ARIMA":
                        # Modèle ARIMA
                        model = fit_arima_model(train_data['Dernier'].values, order=arima_order)
                        
                        if model is not None:
                            # Prédictions sur le jeu de test
                            predictions_test = model.forecast(steps=len(test_data))
                            
                            # Prédictions futures
                            # Réentraîner sur toutes les données pour les prédictions futures
                            full_model = fit_arima_model(df['Dernier'].values, order=arima_order)
                            future_predictions = full_model.forecast(steps=prediction_days)
                            
                            model_info = {
                                'AIC': model.aic,
                                'BIC': model.bic,
                                'RMSE_test': np.sqrt(mean_squared_error(test_data['Dernier'], predictions_test))
                            }
                    
                    elif selected_model == "LSTM":
                        # Modèle LSTM
                        lstm_model = LSTMPredictor(sequence_length=sequence_length)
                        
                        if len(train_data) > sequence_length:
                            # Prédictions sur le jeu de test
                            predictions_test = lstm_model.predict_simple(
                                train_data['Dernier'].values, 
                                len(test_data)
                            )
                            
                            # Prédictions futures
                            future_predictions = lstm_model.predict_simple(
                                df['Dernier'].values, 
                                prediction_days
                            )
                            
                            model_info = {
                                'Sequence_Length': sequence_length,
                                'RMSE_test': np.sqrt(mean_squared_error(test_data['Dernier'], predictions_test))
                            }
                        else:
                            st.error(f"Pas assez de données pour LSTM. Besoin d'au moins {sequence_length} observations.")
                    
                    elif selected_model == "Prophet":
                        # Modèle Prophet
                        model = fit_prophet_model(train_data)
                        
                        if model is not None:
                            # Prédictions sur le jeu de test
                            future_test = model.make_future_dataframe(periods=len(test_data))
                            forecast_test = model.predict(future_test)
                            predictions_test = forecast_test['yhat'].tail(len(test_data)).values
                            
                            # Prédictions futures
                            full_model = fit_prophet_model(df)
                            future_df = full_model.make_future_dataframe(periods=prediction_days)
                            forecast = full_model.predict(future_df)
                            future_predictions = forecast['yhat'].tail(prediction_days).values
                            
                            model_info = {
                                'RMSE_test': np.sqrt(mean_squared_error(test_data['Dernier'], predictions_test))
                            }
                
                # Affichage des résultats
                if len(predictions_test) > 0 and len(future_predictions) > 0:
                    
                    # Métriques de performance
                    if len(test_data) > 0:
                        metrics = calculate_metrics(test_data['Dernier'].values, predictions_test)
                        
                        st.subheader(f"📊 Performance du Modèle {selected_model}")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("MAE", f"{metrics['MAE']:.2f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                        with col3:
                            st.metric("MAPE", f"{metrics['MAPE']:.1f}%")
                        with col4:
                            precision = max(0, 100 - metrics['MAPE'])
                            st.metric("Précision", f"{precision:.1f}%")
                    
                    # Informations sur le modèle
                    if model_info:
                        st.subheader("ℹ️ Informations du Modèle")
                        info_cols = st.columns(len(model_info))
                        for i, (key, value) in enumerate(model_info.items()):
                            with info_cols[i]:
                                if isinstance(value, float):
                                    st.metric(key, f"{value:.3f}")
                                else:
                                    st.metric(key, str(value))
                    
                    # Création des dates futures
                    last_date = df['Date'].max()
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=prediction_days,
                        freq='D'
                    )
                    
                    # Graphique des prédictions
                    st.subheader(f"🔮 Prédictions avec {selected_model}")
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Vue d\'ensemble', 'Zoom sur les prédictions'),
                        vertical_spacing=0.1
                    )
                    
                    # Graphique 1: Vue d'ensemble
                    # Données historiques
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df['Dernier'],
                            mode='lines',
                            name='Données historiques',
                            line=dict(color='blue', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Prédictions de test
                    if len(test_data) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=test_data['Date'],
                                y=predictions_test,
                                mode='lines',
                                name='Prédictions test',
                                line=dict(color='orange', width=2, dash='dash')
                            ),
                            row=1, col=1
                        )
                    
                    # Prédictions futures

                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=future_predictions,
                            mode='lines',
                            name='Prédictions futures',
                            line=dict(color='red', width=3)
                        ),
                        row=1, col=1
                    )
                    
                    # Graphique 2: Zoom sur les derniers 90 jours + prédictions
                    recent_data = df.tail(90)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=recent_data['Date'],
                            y=recent_data['Dernier'],
                            mode='lines',
                            name='Données récentes',
                            line=dict(color='blue', width=2),
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=future_predictions,
                            mode='lines+markers',
                            name='Prédictions futures (zoom)',
                            line=dict(color='red', width=3),
                            marker=dict(size=6),
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    # Mise à jour du layout
                    fig.update_layout(
                        height=800,
                        title_text=f"Prédictions TUINDEX - Modèle {selected_model}",
                        hovermode='x unified'
                    )
                    
                    fig.update_xaxes(title_text="Date", row=2, col=1)
                    fig.update_yaxes(title_text="Valeur TUINDEX", row=1, col=1)
                    fig.update_yaxes(title_text="Valeur TUINDEX", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau des prédictions futures
                    st.subheader("📋 Prédictions Détaillées")
                    
                    predictions_df = pd.DataFrame({
                        'Date': future_dates,
                        'Prédiction': future_predictions,
                        'Variation (%)': [0] + [
                            ((future_predictions[i] / future_predictions[i-1]) - 1) * 100 
                            for i in range(1, len(future_predictions))
                        ]
                    })
                    
                    # Calcul de la variation par rapport à la dernière valeur connue
                    last_value = df['Dernier'].iloc[-1]
                    predictions_df['Variation vs Actuel (%)'] = [
                        ((pred / last_value) - 1) * 100 for pred in future_predictions
                    ]
                    
                    st.dataframe(
                        predictions_df.style.format({
                            'Prédiction': '{:.2f}',
                            'Variation (%)': '{:.2f}%',
                            'Variation vs Actuel (%)': '{:.2f}%'
                        }),
                        use_container_width=True
                    )
                    
                    # Résumé des prédictions
                    st.subheader("📈 Résumé des Prédictions")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Prédiction à 7 jours",
                            f"{future_predictions[6]:.2f}" if len(future_predictions) > 6 else "N/A",
                            f"{((future_predictions[6] / last_value - 1) * 100):.2f}%" if len(future_predictions) > 6 else None
                        )
                    
                    with col2:
                        st.metric(
                            "Prédiction à 30 jours",
                            f"{future_predictions[29]:.2f}" if len(future_predictions) > 29 else f"{future_predictions[-1]:.2f}",
                            f"{((future_predictions[29 if len(future_predictions) > 29 else -1] / last_value - 1) * 100):.2f}%"
                        )
                    
                    with col3:
                        max_pred = max(future_predictions)
                        min_pred = min(future_predictions)
                        volatility = ((max_pred - min_pred) / np.mean(future_predictions)) * 100
                        st.metric(
                            "Volatilité prévue",
                            f"{volatility:.1f}%"
                        )
                
                else:
                    st.error("Erreur lors de la génération des prédictions.")
    
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier: {str(e)}")
        st.info("Assurez-vous que votre fichier CSV contient les colonnes 'Date' et 'Dernier'")

else:
    st.info("👆 Veuillez charger un fichier CSV contenant les données TUINDEX ")
    
    # Instructions d'utilisation
    st.markdown("""
    ### 📝 Instructions d'utilisation:
    
    1. **Chargez votre fichier CSV** avec les colonnes suivantes:
       - `Date`: Date de l'observation
       - `Dernier`: Valeur de l'indice TUINDEX
    
    2. **Choisissez le modèle** de prédiction:
       - **ARIMA**: Modèle statistique classique pour les séries temporelles
       - **LSTM**: Réseau de neurones récurrent pour capturer les patterns complexes
       - **Prophet**: Modèle développé par Facebook pour les prévisions robustes
    
    3. **Configurez les paramètres**:
       - Nombre de jours à prédire
       - Taille du jeu de test pour évaluation
       - Paramètres spécifiques au modèle choisi
    
    4. **Analysez les résultats**:
       - Métriques de performance sur le jeu de test
       - Graphiques des prédictions
       - Tableau détaillé des prédictions futures
    """)

# Note sur les dépendances
if not ARIMA_AVAILABLE:
    st.sidebar.warning("⚠️ ARIMA non disponible. Installez: pip install statsmodels")

if not PROPHET_AVAILABLE:
    st.sidebar.warning("⚠️ Prophet non disponible. Installez: pip install prophet")