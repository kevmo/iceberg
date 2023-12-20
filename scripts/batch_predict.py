import sys
import pandas as pd
import requests

def predict_survival(input_filepath, output_filepath):
    # Load the preprocessed data
    test_data = pd.read_csv(input_filepath)

    # Initialize an empty list to store predictions
    predictions = []

    # Iterate through each row in the test data
    # Iterate through each row in the test data
    for _, row in test_data.iterrows():
        # Prepare the data for the /predict endpoint
        data = {
            "PassengerId": row["PassengerId"],
            "Pclass": row["Pclass"],
            "Age": row["Age"],
            "SibSp": row["SibSp"],
            "Parch": row["Parch"],
            "Fare": row["Fare"],
            "Sex_female": row.get("Sex_female", False),  # Handle missing column
            "Sex_male": row.get("Sex_male", False),  # Handle missing column
            "Embarked_C": row.get("Embarked_C", False),  # Handle missing column
            "Embarked_Q": row.get("Embarked_Q", False),  # Handle missing column
            "Embarked_S": row.get("Embarked_S", False),  # Handle missing column
            "Title_Capt": row.get("Title_Capt", False),  # Handle missing column
            "Title_Col": row.get("Title_Col", False),  # Handle missing column
            "Title_Countess": row.get("Title_Countess", False),  # Handle missing column
            "Title_Don": row.get("Title_Don", False),  # Handle missing column
            "Title_Dr": row.get("Title_Dr", False),  # Handle missing column
            "Title_Jonkheer": row.get("Title_Jonkheer", False),  # Handle missing column
            "Title_Lady": row.get("Title_Lady", False),  # Handle missing column
            "Title_Major": row.get("Title_Major", False),  # Handle missing column
            "Title_Master": row.get("Title_Master", False),  # Handle missing column
            "Title_Miss": row.get("Title_Miss", False),  # Handle missing column
            "Title_Mlle": row.get("Title_Mlle", False),  # Handle missing column
            "Title_Mme": row.get("Title_Mme", False),  # Handle missing column
            "Title_Mr": row.get("Title_Mr", False),  # Handle missing column
            "Title_Mrs": row.get("Title_Mrs", False),  # Handle missing column
            "Title_Ms": row.get("Title_Ms", False),  # Handle missing column
            "Title_Rev": row.get("Title_Rev", False),  # Handle missing column
            "Title_Sir": row.get("Title_Sir", False),  # Handle missing column
        }

        # Make a POST request to the /predict endpoint
        response = requests.post("http://localhost:5000/predict", json=data)

        # Get the prediction from the response
        prediction_response = response.json()
        print("Full prediction response:", prediction_response)

        # Access the correct key in the response
        predictions_list = prediction_response.get("predictions", None)

        if predictions_list is not None and predictions_list:
            # Assuming only one prediction in the list
            prediction = predictions_list[0]
            predictions.append([row["PassengerId"], int(prediction)])
        else:
            # Handle the case where the predictions key is missing or empty
            print("Warning: 'predictions' key is missing or empty in the response.")
            predictions.append([row["PassengerId"], None])

        # # Get the prediction from the response
        # prediction = response.json()
        # print("Full prediction response:", prediction)

        # # Access the correct key in the response
        # prediction = prediction.get("predictions", None)
        # if prediction is not None:
        #     predictions.append([row["PassengerId"], int(prediction)])
        # else:
        #     # Handle the case where the prediction key is missing
        #     print("Warning: 'prediction' key is missing in the response.")
        #     predictions.append([row["PassengerId"], None])

        # # Extract the predicted value from the response
        # prediction = response.json()["prediction"]

        # # Append PassengerId and the predicted value to the list
        # predictions.append({"PassengerId": int(row["PassengerId"]), "Survived": int(prediction)})

    # Convert the list of predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Save the predictions to the specified output file
    predictions_df.to_csv(output_filepath, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_predict.py <input_csv_filepath>")
        sys.exit(1)

    input_csv_filepath = sys.argv[1]
    output_csv_filepath = input_csv_filepath.replace('processed', 'predictions')

    predict_survival(input_csv_filepath, output_csv_filepath)
