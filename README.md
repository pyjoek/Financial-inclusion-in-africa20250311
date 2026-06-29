# Financial inclusion in africa20250311
## Zindi Challenge Demo
I tried using Logistic Regression for the chalenge
## Steps
### Import Library
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Machine Learning
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import sklearn.model_selection

    #import classifier algorithm here
    from sklearn.linear_model import LogisticRegression

    #import preprocessing module
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler

    # import evaluation metrics
    from sklearn.metrics import confusion_matrix, accuracy_score

    # Suppress warnings for better readability
    import warnings
    warnings.filterwarnings('ignore')

### Declared the Train and Test Set
    train = pd.read_csv('Train.csv')  # Training dataset
    test = pd.read_csv('Test.csv')  # Test dataset (no labels)
    ss = pd.read_csv('SampleSubmission.csv')  # Sample submission format
    variables = pd.read_csv('VariableDefinitions.csv')  # Data dictionary

### Check Shape
    # print(f"✅ Train dataset: {train.shape[0]} rows, {train.shape[1]} columns")
    # print(f"✅ Test dataset: {test.shape[0]} rows, {test.shape[1]} columns")

### Declare Target
    le = LabelEncoder()
    train['bank_account'] = le.fit_transform(train['bank_account'])

### Separate Training Feature from Target
    X_train = train.drop(['bank_account'], axis=1)
    y_train = train['bank_account']

### Feature Engineering
    def preprocessing_data(data):
        # Convert the following numerical labels from interger to float
        float_array = data[["household_size", "age_of_respondent", "year"]].values.astype(float)

        # categorical features to be onverted to One Hot Encoding
        categ = ["relationship_with_head", "marital_status", "education_level", "job_type", "country"]

        # One Hot Encoding conversion
        data = pd.get_dummies(data, prefix_sep="_", columns=categ)

        # Label Encoder conversion
        data["location_type"] = le.fit_transform(data["location_type"])
        data["cellphone_access"] = le.fit_transform(data["cellphone_access"])
        data["gender_of_respondent"] = le.fit_transform(data["gender_of_respondent"])

        # drop uniquid column
        data = data.drop(["uniqueid"], axis=1)

        # scale our data into range of 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)

        return data

### Process The Training Data
    processed_train = preprocessing_data(X_train)
    processed_test = preprocessing_data(test)

    X_Train, X_Val, y_Train, y_val = train_test_split(processed_train, y_train, stratify = y_train, test_size = 0.1, random_state=42)

### Init Model
    clf = LogisticRegression(max_iter=5000, class_weight="balanced")

### Fit Model
    clf.fit(X_Train, y_Train)

### Sample Predict
    clf_model = clf.predict(X_Val)

### Check the proportion of incorrect prediction made by the model
    print("Error rate of LogisticRegression classifier: ", 1 - accuracy_score(y_val, clf_model))

### Predict
    test.bank_account = clf.predict(processed_test)

### Format to required output format
    submission = pd.DataFrame({"uniqueid": test["uniqueid"] + " x " + test["country"], "bank_account": test.bank_account})

### Save
    submission.to_csv('first_submission.csv', index = False)