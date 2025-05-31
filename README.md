# Context-Aware-Restricted-Boltzmann-Machine-for-Personalized-Diabetes-Management

# 🍽️ Diabetes Meal Planner using Context-Aware RBM

This project is a personalized meal recommendation system for diabetes management. It uses a **Context-Aware Restricted Boltzmann Machine (CA-RBM)** to tailor meal plans based on individual health metrics, lifestyle factors, and socio-cultural context. The goal is to enhance dietary adherence and treatment effectiveness for diabetic patients.

## 🔍 Key Features

- **Streamlit Web App**: Simple UI to input health data and get personalized meal plans
- **Context-Aware Machine Learning**: Incorporates cultural, lifestyle, and healthcare accessibility context
- **Auto-Saving**: Every meal plan is saved as a CSV locally
- **Interactive**: Visual output + downloadable meal plans

## 💡 Background

This tool is based on the research paper:  
_**“Context-Aware Restricted Boltzmann Machine for Personalized Diabetes Management”**_  
Authored by **Maha Laxmy S**, **Laavanya R**, **Dr. Nirmala B**  
Sri Ramachandra Faculty of Engineering and Technology, Chennai, India.

## 🧠 Model

We use an enhanced RBM that integrates:
- Demographics (Age, Gender)
- Health metrics (Glucose, HbA1c)
- Lifestyle (Diet score, Activity level)
- Medical History (Comorbidities)
- Contextual features (healthcare access, regional practices)

## 🛠 How to Run

### 🔧 Installation

```bash
pip install -r requirements.txt
