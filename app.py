import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from rbm_model import ContextAwareRBM

st.set_page_config(page_title="Diabetes Meal Planner", page_icon="🍽️", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("generated_dataset_with_varied_meals.csv")
    except FileNotFoundError:
        st.error("Dataset file 'generated_dataset_with_varied_meals.csv' not found.")
        return None, None, None, None, None

    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Activity Level'] = df['Activity Level'].astype('category').cat.codes
    df['Healthcare Access'] = df['Healthcare Access'].astype('category').cat.codes
    df['Comorbidity Count'] = df['Comorbidities'].apply(lambda x: len(x.split(',')))
    
    features = ['Age', 'Gender', 'Glucose', 'HbA1c', 'Diet Score', 'Activity Level', 'Healthcare Access', 'Comorbidity Count']
    context_features = ['Diet Score', 'Activity Level']
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features])
    context = df[context_features].values
    
    return df, X, context, scaler, features

@st.cache_resource
def train_model(X, context):
    if X is None or context is None:
        return None

    X_tensor = torch.tensor(X, dtype=torch.float32)
    context_tensor = torch.tensor(context, dtype=torch.float32)
    model = ContextAwareRBM(n_visible=X.shape[1], n_hidden=64, n_context=context.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        v_prob = model(X_tensor, context_tensor)
        loss = torch.mean((X_tensor - v_prob) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "trained_rbm_model.pth")
    return model

def recommend_meal(user_input, context_input, df, model, scaler, features):
    if model is None:
        return None

    user_vector = scaler.transform([user_input])
    context_indices = [features.index('Diet Score'), features.index('Activity Level')]
    context_min = scaler.data_min_[context_indices]
    context_max = scaler.data_max_[context_indices]

    context_vector = (np.array(context_input) - context_min) / (context_max - context_min + 1e-8)
    context_vector = np.clip(context_vector, 0, 1).reshape(1, -1)

    user_tensor = torch.tensor(user_vector, dtype=torch.float32)
    context_tensor = torch.tensor(context_vector, dtype=torch.float32)

    v_prob = model(user_tensor, context_tensor).detach().numpy()
    closest_idx = np.argmin(np.linalg.norm(scaler.transform(df[features]) - v_prob, axis=1))
    recommended_meals = df.iloc[closest_idx][['Breakfast', 'Lunch', 'Dinner']]
    return recommended_meals

def main():
    st.title("Diabetes Meal Planner")
    df, X, context, scaler, features = load_and_preprocess_data()
    if df is not None:
        model = train_model(X, context)
    else:
        st.stop()

    st.sidebar.header("Your Health Profile")
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    glucose = st.sidebar.number_input("Glucose Level", min_value=50, max_value=300, value=100)
    hba1c = st.sidebar.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
    diet_score = st.sidebar.slider("Diet Score", 1.0, 20.0, 15.0)
    
    activity_level = st.sidebar.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"])
    activity_map = {"Sedentary": 0, "Lightly Active": 1, "Moderately Active": 2, "Very Active": 3, "Extremely Active": 4}
    activity_code = activity_map[activity_level]

    healthcare_access = st.sidebar.selectbox("Healthcare Access", ["Low", "Medium", "High"])
    healthcare_map = {"Low": 0, "Medium": 1, "High": 2}
    healthcare_code = healthcare_map[healthcare_access]

    comorbidities = st.sidebar.text_input("Comorbidities (comma-separated)", "Hypertension")

    if st.sidebar.button(" Get My Meal Plan"):
        gender_code = 0 if gender == "Male" else 1
        comorbidity_count = len(comorbidities.split(',')) if comorbidities.strip() else 0

        user_input = [age, gender_code, glucose, hba1c, diet_score, activity_code, healthcare_code, comorbidity_count]
        context_input = [diet_score, activity_code]

        recommendations = recommend_meal(user_input, context_input, df, model, scaler, features)

        if recommendations is not None:
            meal_plan_df = pd.DataFrame({
                "Meal": ["Breakfast", "Lunch", "Dinner"],
                "Recommendation": recommendations.values
            })

            st.write("###  Your Personalized Meal Plan", meal_plan_df)

            # ✅ Save every time
            meal_plan_df.to_csv("saved_meal_plan.csv", index=False)

if __name__ == "__main__":
    main()
