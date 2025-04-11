import traceback
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load models
bank_model = joblib.load("models/bank_model.pkl")
ott_model = joblib.load("models/ott_model.pkl")
retail_model = joblib.load("models/retail_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    company = data.get("company")
    features = data.get("features")

    try:
        if company == "Banking":
            input_vector = [
                int(features["CreditScore"]),
                int(features["Age"]),
                int(features["Tenure"]),
                float(features["Balance"]),
                int(features["NumOfProducts"]),
                1 if features["HasCrCard"] == "Yes" else 0,
                1 if features["IsActiveMember"] == "Yes" else 0,
                float(features["EstimatedSalary"]),
                1 if features["Geography"] == "Germany" else 0,
                1 if features["Geography"] == "Spain" else 0,
                1 if features["Gender"] == "Male" else 0
            ]
            prediction = bank_model.predict(np.array([input_vector]))[0]

        elif company == "OTT Platform":
            input_vector = [
                float(features["Age"]),
                float(features["Subscription_Length_Months"]),
                float(features["Monthly_Bill"]),
                float(features["Total_Usage_GB"]),
                float(features["Support_Calls"]),
                {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}[features["Contract_Type"]],
                1 if features["Has_Additional_Services"] == "Yes" else 0
            ]
            prediction = ott_model.predict([input_vector])[0]

        elif company == "Online Retail Store":
            input_vector = [
                float(features["Age"]),
                float(features["Annual_Income"]),
                float(features["Total_Spend"]),
                float(features["Years_as_Customer"]),
                float(features["Num_of_Purchases"]),
                float(features["Average_Transaction_Amount"]),
                float(features["Num_of_Returns"]),
                float(features["Num_of_Support_Contacts"]),
                float(features["Satisfaction_Score"]),
                float(features["Last_Purchase_Days_Ago"]),
                1 if features["Gender"] == "Male" else 0,
                1 if features["Gender"] == "Other" else 0,
                1 if features["Promotion_Response"] == "Responded" else 0,
                1 if features["Promotion_Response"] == "Unsubscribed" else 0
            ]
            prediction = retail_model.predict([input_vector])[0]

        else:
            return jsonify({"error": "Invalid company type"}), 400

        result = "Churn" if prediction == 1 else "No Churn"
        return jsonify({"prediction": result})


    except Exception as e:

        print("🔥 Error during prediction:", str(e))

        print("🧪 Input data was:", features)

        traceback.print_exc()

        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
