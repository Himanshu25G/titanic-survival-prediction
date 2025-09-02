
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import joblib
import re

RANDOM_STATE = 42

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to get a survival prediction.")

# ---------- Feature Engineering helpers (same as notebook) ----------
def extract_title(name: str) -> str:
    if pd.isna(name):
        return "None"
    m = re.search(r",\s*([^\.]+)\.", name)
    return m.group(1).strip() if m else "None"

def unify_title(t):
    mapping = {
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Royalty', 'Countess': 'Royalty', 'Dona':'Royalty',
        'Sir': 'Royalty', 'Jonkheer': 'Royalty', 'Don':'Royalty',
        'Capt':'Officer','Col':'Officer','Major':'Officer','Dr':'Officer','Rev':'Officer'
    }
    return mapping.get(t, t)

def cabin_deck(cabin):
    if pd.isna(cabin) or not isinstance(cabin, str):
        return "Unknown"
    return cabin[0]

def ticket_prefix(ticket):
    if pd.isna(ticket) or not isinstance(ticket, str):
        return "NONE"
    t = ticket.replace(".", "").replace("/", "").split()
    prefix = "".join([p for p in t if not p.isdigit()])
    return prefix.upper() if prefix else "NONE"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].apply(extract_title).apply(unify_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Deck"] = df["Cabin"].apply(cabin_deck)
    df["TicketPrefix"] = df["Ticket"].apply(ticket_prefix)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"].replace(0, 1)
    return df

numeric_features = ["Age","SibSp","Parch","Fare","FamilySize","FarePerPerson"]
categorical_features = ["Pclass","Sex","Embarked","Title","Deck","TicketPrefix","IsAlone"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

# Try to load pre-trained model; if not found, optionally quick-train from train.csv
best_model = None
if os.path.exists("best_model.pkl"):
    try:
        best_model = joblib.load("best_model.pkl")
    except Exception:
        best_model = None

if best_model is None:
    st.info("No pre-trained model found. I can quickly train one if `train.csv` is present.")
    if os.path.exists("train.csv"):
        train = pd.read_csv("train.csv")
        train_fe = add_features(train)
        # Impute basic fields
        train_fe["Fare"] = train_fe["Fare"].fillna(train_fe["Fare"].median())
        train_fe["Embarked"] = train_fe["Embarked"].fillna(train_fe["Embarked"].mode()[0])
        age_medians = train_fe.groupby(["Title","Pclass","Sex"])["Age"].median()
        def impute_age(row):
            if pd.isna(row["Age"]):
                return age_medians.get((row["Title"], row["Pclass"], row["Sex"]), train_fe["Age"].median())
            return row["Age"]
        train_fe["Age"] = train_fe.apply(impute_age, axis=1)

        X = train_fe.drop(columns=["Survived"])
        y = train_fe["Survived"].astype(int)

        # Keep it simple & fast: Logistic Regression with small CV
        model = Pipeline(steps=[("preprocess", preprocessor), ("clf", LogisticRegression(max_iter=200, random_state=RANDOM_STATE))])
        model.fit(X, y)
        best_model = model
        joblib.dump(best_model, "best_model.pkl")
        st.success("Quick model trained and saved as best_model.pkl")
    else:
        st.warning("Upload `train.csv` (Kaggle) to auto-train a quick model.")

# ---------- UI Inputs ----------
with st.form("passenger_form"):
    name = st.text_input("Name (e.g., 'Doe, Mr. John')", "Doe, Mr. John")
    pclass = st.selectbox("Passenger Class (Pclass)", [1,2,3], index=2)
    sex = st.selectbox("Sex", ["male","female"], index=0)
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, value=10.0, step=0.1)
    embarked = st.selectbox("Embarked", ["S","C","Q"], index=0)
    cabin = st.text_input("Cabin (optional)", "")
    ticket = st.text_input("Ticket", "A/5 21171")
    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a single-row dataframe mimicking the test schema
    row = pd.DataFrame([{
        "PassengerId": 0,
        "Pclass": pclass,
        "Name": name,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": ticket,
        "Fare": fare,
        "Cabin": cabin if cabin else np.nan,
        "Embarked": embarked
    }])
    row_fe = add_features(row)
    # Basic imputations
    row_fe["Fare"] = row_fe["Fare"].fillna(0)
    row_fe["Embarked"] = row_fe["Embarked"].fillna("S")
    if best_model is None:
        st.error("No model available. Please provide `train.csv` to train or place `best_model.pkl`.")
    else:
        proba = None
        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(row_fe)[:,1][0]
        pred = best_model.predict(row_fe)[0]
        st.subheader("Prediction")
        st.write(f"**Survived:** {'Yes ✅' if int(pred)==1 else 'No ❌'}")
        if proba is not None:
            st.write(f"**Probability of Survival:** {proba:.2%}")
