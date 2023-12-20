import sys
import pandas as pd

def preprocess_data(csv_filepath):
    # Load Data
    titanic_data = pd.read_csv(csv_filepath)

    # Impute Missing Values
    titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
    titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
    titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)

    # Dropping a column with many missing values
    titanic_data.drop('Cabin', axis=1, inplace=True)

    # One-hot encoding of remaining categorical variables
    titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'])

    # Extracting information from non-numerical column "Name"
    titanic_data['Title'] = titanic_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    titanic_data.drop('Name', axis=1, inplace=True)
    titanic_data = pd.get_dummies(titanic_data, columns=['Title'])

    # Dropping a final non-numeric column
    titanic_data.drop('Ticket', axis=1, inplace=True)

    # Save the Preprocessed Features
    output_filepath = csv_filepath.replace('raw', 'processed')
    titanic_data.to_csv(output_filepath, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocessing.py <input_csv_filepath>")
        sys.exit(1)

    input_csv_filepath = sys.argv[1]
    preprocess_data(input_csv_filepath)
