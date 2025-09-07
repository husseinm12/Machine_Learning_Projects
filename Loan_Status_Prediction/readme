# 🏦 Loan Status Prediction using SVM  

This project is a **machine learning classification system** that predicts whether a loan will be approved or not based on customer details. It uses a **Support Vector Machine (SVM)** model trained on loan applicant data.

## 📌 How It Works  

1. **Import Libraries**  
   - `numpy`, `pandas` → data handling and preprocessing.  
   - `matplotlib`, `seaborn` → data visualization.  
   - `scikit-learn` → machine learning model, dataset splitting, and accuracy evaluation.  

2. **Load Dataset**  
   - Reads `loan_data.csv` into a DataFrame.  
   - Drops missing values.  
   - Replaces categorical values (e.g., "Y"/"N", "Male"/"Female", "Urban"/"Rural") with numeric codes.  

3. **Feature Engineering**  
   - `Dependents` values like `3+` are converted into `4`.  
   - Categorical columns (`Married`, `Gender`, `Self_Employed`, `Property_Area`, `Education`) are encoded into numbers.  
   - `Loan_Status` is mapped to `1` (Approved) and `0` (Not Approved).  

4. **Splitting Data**  
   - Features `X` = all columns except `Loan_ID` and `Loan_Status`.  
   - Target `Y` = `Loan_Status`.  
   - Data is split into training (90%) and testing (10%) sets using `train_test_split`.  

5. **Model Training**  
   - A Support Vector Classifier (`svm.SVC`) with **linear kernel** is trained on the data.  

6. **Evaluation**  
   - Accuracy is calculated for both training and testing datasets using `accuracy_score`.  
   - Prints performance results in the terminal.  

## 🛠️ Requirements

Install dependencies with: pip install -r requirements.txt

▶️ Usage

Place loan_data.csv in the same folder as the script.
Run the script: python loan_prediction.py
The program will:  1- Preprocess the dataset.
                   2- Train an SVM model.
                   3- Output training and testing accuracy.
                 

📊 Example Output
(614, 13)
Accuracy on training data: 0.80
Accuracy on testing data: 0.78
