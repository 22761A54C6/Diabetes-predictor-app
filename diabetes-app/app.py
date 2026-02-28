from flask import Flask, render_template, request
import pickle
import pandas as pd
import shap

app = Flask(__name__)

model = pickle.load(open("GB_DATIBETES_MODEL.pkl", "rb"))

feature_names = [
    "gender","age","hypertension","heart_disease",
    "smoking_history","bmi","HbA1c_level","blood_glucose_level"
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "gender": request.form["gender"],
        "age": float(request.form["age"]),
        "hypertension": int(request.form["hypertension"]),
        "heart_disease": int(request.form["heart_disease"]),
        "smoking_history": request.form["smoking_history"],
        "bmi": float(request.form["bmi"]),
        "HbA1c_level": float(request.form["hba1c"]),
        "blood_glucose_level": float(request.form["glucose"])
    }

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    explanation = None

    if prediction == 1:
        result = "High Risk — Likely Diabetes"
        color = "danger"

        preprocess = model.named_steps["preprocess"]
        gb_model = model.named_steps["gb"]
        df_proc = preprocess.transform(df)

        explainer = shap.Explainer(
            gb_model,
            feature_names=preprocess.get_feature_names_out(feature_names)
        )
        shap_values = explainer(df_proc)

        def clean(name):
            return name.split("__")[-1]

        shap_importance = sorted(
            list(zip(explainer.feature_names, shap_values.values[0])),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        explanation = [
            f"{clean(name)}: influence {round(val, 3)}"
            for name, val in shap_importance
        ]

    else:
        result = "Low Risk — No Diabetes Detected"
        color = "success"

    return render_template(
        "index.html",
        result=result,
        color=color,
        explanation=explanation
    )

if __name__ == "__main__":
    app.run(debug=True)
