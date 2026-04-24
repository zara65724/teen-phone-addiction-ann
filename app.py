import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Teen Phone Addiction - ANN", page_icon="📱", layout="wide")
st.title("📱 Teen Phone Addiction Predictor")
st.markdown("**Artificial Neural Network (MLP)** | Addiction Level Prediction")
st.markdown("---")

@st.cache_data
def load_data():
    return pd.read_csv("teen_phone_addiction.csv")

@st.cache_resource
def train_model(hidden1, hidden2, hidden3, max_iter, alpha):
    df = load_data()
    raw_df = df.copy()
    df.drop(columns=[c for c in ["ID","Name","Location","Phone_Usage_Purpose"] if c in df.columns], inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.factorize(df[col])[0]
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop("Addiction_Level", axis=1))
    y = df["Addiction_Level"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    layers = tuple(x for x in [hidden1, hidden2, hidden3] if x > 0)
    model = MLPRegressor(hidden_layer_sizes=layers, activation='relu', solver='adam',
                         alpha=alpha, max_iter=max_iter, early_stopping=True,
                         validation_fraction=0.2, random_state=42, verbose=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_test, y_pred, scaler, raw_df, df

st.sidebar.header("⚙️ Model Settings")
hidden1  = st.sidebar.slider("Layer 1 Neurons", 16, 128, 64, step=16)
hidden2  = st.sidebar.slider("Layer 2 Neurons", 0,  64,  32, step=16)
hidden3  = st.sidebar.slider("Layer 3 Neurons", 0,  32,  16, step=8)
max_iter = st.sidebar.slider("Max Iterations",  50, 500, 200, step=50)
alpha    = st.sidebar.select_slider("L2 Regularization", options=[0.0001,0.001,0.01,0.1], value=0.001)
train_btn = st.sidebar.button("🚀 Train Model", use_container_width=True)

if "trained" not in st.session_state or train_btn:
    with st.spinner("Model train ho raha hai... 🧠"):
        model, y_test, y_pred, scaler, raw_df, proc_df = train_model(hidden1, hidden2, hidden3, max_iter, alpha)
    st.session_state.update({"trained":True,"model":model,"y_test":y_test,"y_pred":y_pred,
                              "scaler":scaler,"raw_df":raw_df,"proc_df":proc_df})
    st.success("✅ Model successfully trained!")

if st.session_state.get("trained"):
    model   = st.session_state.model
    y_test  = st.session_state.y_test
    y_pred  = st.session_state.y_pred
    scaler  = st.session_state.scaler
    raw_df  = st.session_state.raw_df
    proc_df = st.session_state.proc_df

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📉 MSE", f"{mse:.4f}")
    c2.metric("📏 MAE", f"{mae:.4f}")
    c3.metric("📈 R² Score", f"{r2:.4f}")
    c4.metric("🗂️ Dataset", f"{len(raw_df)} rows")
    st.markdown("---")

    st.subheader("📊 Model Performance")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='white', linewidth=0.3)
    mn,mx = min(y_test.min(),y_pred.min()), max(y_test.max(),y_pred.max())
    axes[0].plot([mn,mx],[mn,mx],'r--',linewidth=1.5,label='Perfect fit')
    axes[0].set(title="Actual vs Predicted", xlabel="Actual", ylabel="Predicted")
    axes[0].legend()
    axes[1].plot(model.loss_curve_, color='steelblue', label='Train Loss')
    axes[1].set(title="Loss Curve", xlabel="Iteration", ylabel="Loss")
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔍 Dataset Explorer")
    col_left, col_right = st.columns([2,1])
    with col_left:
        st.dataframe(raw_df.head(20), use_container_width=True)
    with col_right:
        st.write("**Key Stats:**")
        st.dataframe(raw_df[["Daily_Usage_Hours","Sleep_Hours","Anxiety_Level","Addiction_Level"]].describe().round(2))

    st.markdown("---")
    st.subheader("🎯 Live Addiction Level Predictor")
    p1,p2,p3,p4 = st.columns(4)
    age       = p1.slider("Age", 10, 20, 15)
    daily_hrs = p2.slider("Daily Usage (hrs)", 0.0, 16.0, 6.0, step=0.5)
    sleep     = p3.slider("Sleep Hours", 3.0, 12.0, 7.0, step=0.5)
    anxiety   = p4.slider("Anxiety Level", 0, 10, 5)
    p5,p6,p7,p8 = st.columns(4)
    depression = p5.slider("Depression Level", 0, 10, 4)
    social     = p6.slider("Social Interactions", 0, 10, 5)
    exercise   = p7.slider("Exercise Hours/day", 0.0, 5.0, 1.0, step=0.5)
    parental   = p8.slider("Parental Control", 0, 10, 5)
    q1,q2,q3,q4 = st.columns(4)
    screen_bed = q1.slider("Screen Time Before Bed (hrs)", 0.0, 6.0, 1.0, step=0.5)
    checks     = q2.slider("Phone Checks/Day", 0, 200, 80)
    apps       = q3.slider("Apps Used Daily", 1, 30, 10)
    social_med = q4.slider("Time on Social Media (hrs)", 0.0, 8.0, 2.0, step=0.5)
    r1,r2_,r3,r4 = st.columns(4)
    gaming    = r1.slider("Time on Gaming (hrs)", 0.0, 8.0, 1.0, step=0.5)
    education = r2_.slider("Time on Education (hrs)", 0.0, 8.0, 1.0, step=0.5)
    family    = r3.slider("Family Communication", 0, 10, 5)
    weekend   = r4.slider("Weekend Usage (hrs)", 0.0, 16.0, 8.0, step=0.5)

    if st.button("🔮 Predict Addiction Level", use_container_width=True):
        feature_cols = proc_df.drop("Addiction_Level", axis=1).columns.tolist()
        med = proc_df.median()
        input_dict = {col: med[col] for col in feature_cols}
        for k,v in {"Age":age,"Daily_Usage_Hours":daily_hrs,"Sleep_Hours":sleep,
                    "Anxiety_Level":anxiety,"Depression_Level":depression,
                    "Social_Interactions":social,"Exercise_Hours":exercise,
                    "Parental_Control":parental,"Screen_Time_Before_Bed":screen_bed,
                    "Phone_Checks_Per_Day":checks,"Apps_Used_Daily":apps,
                    "Time_on_Social_Media":social_med,"Time_on_Gaming":gaming,
                    "Time_on_Education":education,"Family_Communication":family,
                    "Weekend_Usage_Hours":weekend}.items():
            if k in input_dict: input_dict[k] = v
        inp = scaler.transform(np.array([[input_dict[c] for c in feature_cols]]))
        prediction = float(np.clip(model.predict(inp)[0], 0, 10))
        level = "🟢 Low" if prediction < 4 else "🟡 Moderate" if prediction < 7 else "🔴 High"
        st.markdown(f"### Predicted Addiction Level: **{prediction:.2f} / 10** — {level}")
        st.progress(prediction / 10)

st.markdown("---")
st.caption("Made with ❤️ using Scikit-learn MLPRegressor + Streamlit")
