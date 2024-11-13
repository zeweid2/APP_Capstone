import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier # Can handle missing value
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
from sklearn.tree import DecisionTreeClassifier
import numpy as np
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr
import seaborn as sns
import itertools
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title('Alliant Credit Union Modeling APP: Version 1.5')

st.header("How This Project Work?")
st.text("This is a project that use certain information of the \ncustomers to determine whether a loan application would be a bad loan or not, \ncan be used for decision marketing to possibly bad loans.")

st.header("The Model We Use")
st.text("We are using the XGboost, logistic model, and EBM")

st.text('To begin with, drop the csv file with the customer data with corresponding variables.')


indirect_file = st.file_uploader('Please Select a CSV File for indirect data', type = ['csv'])
direct_file = st.file_uploader('Please Select a CSV File for direct data', type = ['csv'])
colname_dict = st.file_uploader('Upload Data Dictionary Here', type = ['xlsx'])
target_select = st.selectbox('What target variable would you want to select?', ('Does the loan go DQ at 12 months on book?',
    'Does the loan go 30 days DQ at 12 months on book?',
    'Does the loan go 90 days DQ at 12 months on book?',
    'Does the loan go DQ at 18 months on book?',
    'Does the loan go 30 days DQ at 18 months on book?',
    'Does the loan go 90 days DQ at 18 months on book?',
    'Does the loan go DQ at 24 months on book?',
    'Does the loan go 30 days DQ at 24 months on book?',
    'Does the loan go 90 days DQ at 24 months on book?',
    'Does the loan CHARGE OFF at 12 months on book?',
    'Does the loan CHARGE OFF at 18 months on book?',
    'Does the loan CHARGE OFF at 24 months on book?'), 8)
if (indirect_file is not None and direct_file is not None and colname_dict is not None):
    tab0, tab1, tab2 = st.tabs(["Data Overview", "Model Training & Evaluation", "User Interface(ONLY AVAILABLE AFTER MODEL TRAINING)"])
    df = pd.read_csv(indirect_file)
    df1 = pd.read_csv(direct_file)
    combined_df = pd.concat([df, df1], ignore_index=True)
    combined_df['combinedCreditScore'] = np.where(combined_df['creditScore'].notna(), combined_df['creditScore'], combined_df["ficoCreditScore"])

    combined_df = combined_df.dropna(subset=['ltvRatio'])
    combined_df = combined_df.dropna(subset=['combinedCreditScore'])

    combined_df['application_date'] = pd.to_datetime(combined_df['applicationCreationDate'], format='mixed')
    combined_df['loan_approval_date'] = pd.to_datetime(combined_df['LoanProdOrigLoanDate'], format='mixed', errors='coerce')
    columns_to_fill = ['dq12', 'dq30_12', 'dq90_12', 'dq18', 'dq30_18', 'dq90_18', 'dq24', 'dq30_24', 'dq90_24']
    combined_df[columns_to_fill] = combined_df[columns_to_fill].fillna(0)
    column_mapping = pd.read_excel(colname_dict)
    column_name = dict(zip(column_mapping['Column Name'], column_mapping['Description']))
    combined_df.rename(columns= column_name , inplace= True)
    combined_df.drop(columns = ['Zip code of applicant' , 'Is_Approved', 'Unnamed: 0', 'application_date', 'loan_approval_date', 'application creation date', 'Loan approval date (NA if not approved)' ], inplace = True)

    columns_to_convert = [
    'Does the loan go DQ at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
    'Does the loan go 30 days DQ at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
    'Does the loan go 90 days DQ at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
    'Does the loan go DQ at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
    'Does the loan go 30 days DQ at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
    'Does the loan go 90 days DQ at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
    'Does the loan go DQ at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
    'Does the loan go 30 days DQ at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
    'Does the loan go 90 days DQ at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
    'Does the loan CHARGE OFF at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
    'Does the loan CHARGE OFF at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
    'Does the loan CHARGE OFF at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)']

    # Convert each column to categorical
    for col in columns_to_convert:
        combined_df[col] = combined_df[col].astype('int').astype('category')

    # Step 1: Compute the correlation matrix
    corr_matrix = combined_df.corr().abs()  # Take the absolute value of the correlation matrix

    # Step 2: Create a mask for the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Step 3: Identify features with correlation higher than the threshold (e.g., 0.9)
    threshold = 0.9
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    # Step 4: Drop the highly correlated features from the DataFrame
    df3 = combined_df.drop(columns=to_drop)
    missing_columns = df3.columns[df3.isnull().any()]

    # Display the number of missing values in each column
    missing_data = df3[missing_columns].isnull().sum()
    # Dropping columns with 90% or more missing values
    df3.drop(columns = ['Revolving Utilization', 'Total Unsecured Debt', 'Credit Score', 'Loan purpose', 'Housing type of applicant', 'Employment Status of applicant' ], inplace = True)

    # List of months to check
    months = [12, 18, 24]

    # Initialize a list to store results
    results = []

    # Loop through each month and calculate counts
    for month in months:
        # Column name based on the current month
        dq_column = f'Does the loan go 90 days DQ at {month} months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'

        # Count the number of loans that go 90 days DQ
        dq_90_days_count = df3[df3[dq_column] == 1].shape[0]

        # Count the number of loans with missing ficoCreditScore
        missing_fico_in_dq_90_days = df3[(df3[dq_column] == 1) & (df3['ficoCreditScore'].isnull())].shape[0]

        # Append results to the list
        results.append({
            'Month': month,
            '90 Days DQ Count': dq_90_days_count,
            'Missing FICO Count': missing_fico_in_dq_90_days
        })

    # Create a DataFrame from the results
    dq_summary_df = pd.DataFrame(results)

    # Assuming 'Unique Identifier for row' uniquely identifies each loan in df3

    # Create sets for loans delinquent at each time period
    dq_loans = {month: set(df3[df3[f'Does the loan go 90 days DQ at {month} months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1]['Unique Identifier for row']) for month in months}

    # Find common loans across all periods using set intersection
    same_loans = set.intersection(*dq_loans.values())

    df4 = df3.copy()
    df4 = df4.dropna(subset=['ficoCreditScore'])
    df4.drop(columns = ['Unique Identifier for row'], inplace= True)

    y = df4[target_select + ' (1 if yes, 0 if no, BLANK if loan not sanctioned)']
    X = df4.drop(columns=['Does the loan go 90 days DQ at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
                    'Does the loan go DQ at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
                    'Does the loan go 30 days DQ at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
                    'Does the loan go 90 days DQ at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
                    'Does the loan go DQ at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
                    'Does the loan go DQ at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
                    'Does the loan go 30 days DQ at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
                    'Does the loan go 90 days DQ at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
                    'Does the loan CHARGE OFF at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
                    'Does the loan CHARGE OFF at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
                    'Does the loan go 30 days DQ at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)',
                    'Does the loan CHARGE OFF at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'])
    
    
    tab0.header('Descriptive Statistics For Processed Data')
    a_var = tab0.selectbox('Please Select One Variable:', X.columns, key = 'select_a', placeholder = 'Choose a name')
    tab0.table(X[a_var].describe())
    
    tab0.header('Ratio of Delinquent in variable: ' + target_select)
    tab0.text('Delinquent: ' + str(len(y[y == 1])/len(y)))
    tab0.text('Not Delinquent: ' + str(len(y[y == 0])/len(y)))
    
    tab0.header('Roll Rate Analysis')
    # Calculate roll rates by comparing transitions between stages

    # 12 months roll rates
    roll_rate_12_to_30_12 = ((df3['Does the loan go DQ at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 0) & (df3['Does the loan go 30 days DQ at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1)).mean()
    roll_rate_30_to_90_12 = ((df3['Does the loan go 30 days DQ at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1) & (df3['Does the loan go 90 days DQ at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1)).mean()
    roll_rate_90_to_chargeoff_12 = ((df3['Does the loan go 90 days DQ at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1) & (df3['Does the loan CHARGE OFF at 12 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1)).mean()

    # 18 months roll rates
    roll_rate_12_to_30_18 = ((df3['Does the loan go DQ at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 0) & (df3['Does the loan go 30 days DQ at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1)).mean()
    roll_rate_30_to_90_18 = ((df3['Does the loan go 30 days DQ at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1) & (df3['Does the loan go 90 days DQ at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1)).mean()
    roll_rate_90_to_chargeoff_18 = ((df3['Does the loan go 90 days DQ at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1) & (df3['Does the loan CHARGE OFF at 18 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1)).mean()

    # 24 months roll rates
    roll_rate_12_to_30_24 = ((df3['Does the loan go DQ at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 0) & (df3['Does the loan go 30 days DQ at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1)).mean()
    roll_rate_30_to_90_24 = ((df3['Does the loan go 30 days DQ at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1) & (df3['Does the loan go 90 days DQ at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1)).mean()
    roll_rate_90_to_chargeoff_24 = ((df3['Does the loan go 90 days DQ at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1) & (df3['Does the loan CHARGE OFF at 24 months on book? (1 if yes, 0 if no, BLANK if loan not sanctioned)'] == 1)).mean()

    # Create a DataFrame to store the roll rates and stages
    roll_rates_data = pd.DataFrame({
        'Stage': ['0 to 30 days DQ at 12 months', '30 to 90 days DQ at 12 months', '90 days to Charge-Off at 12 months',
                '0 to 30 days DQ at 18 months', '30 to 90 days DQ at 18 months', '90 days to Charge-Off at 18 months',
                '0 to 30 days DQ at 24 months', '30 to 90 days DQ at 24 months', '90 days to Charge-Off at 24 months'],
        'Roll Rate': [roll_rate_12_to_30_12, roll_rate_30_to_90_12, roll_rate_90_to_chargeoff_12,
                    roll_rate_12_to_30_18, roll_rate_30_to_90_18, roll_rate_90_to_chargeoff_18,
                    roll_rate_12_to_30_24, roll_rate_30_to_90_24, roll_rate_90_to_chargeoff_24]
    })
    fig0, ax0 = plt.subplots(figsize=(10, 6))
    # Plot the roll rates
    ax0.bar(roll_rates_data['Stage'], roll_rates_data['Roll Rate'], color='blue')
    ax0.plot(roll_rates_data['Stage'], roll_rates_data['Roll Rate'], color='black', marker='o')

    # ax0.set_xticklabels(rotation=45, ha="right")
    ax0.set_xlabel('Stages')
    ax0.set_ylabel('Roll Rate')
    ax0.set_title('Loan Roll Rate Analysis')
    # ax0.tight_layout()
    for label in ax0.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    # Show the plot
    tab0.pyplot(fig0)
    
    
    tab1.header("Model Training Setting")
    
    random_seed = tab1.number_input('Setting random seed', 0, 1000000000, 42, 1)
    test_ratio = tab1.slider('Ratio of testing dataset', 0.01, 0.99, 0.20, 0.01)
    pro_var = tab1.slider("Ratio of bad loan over undersampled none bad loan", 0.1, 1.0, 1.0)
    bound_var = tab1.slider("Select the decision boundary", 0.1, 0.9, 0.5)
    if tab1.toggle("Start Model Training", key="start_model_training"):
        # Apply RandomUnderSampler for undersampling the majority class
        rus = RandomUnderSampler(random_state=random_seed, sampling_strategy = pro_var)
        X_resampled, y_resampled = rus.fit_resample(X, y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test_ratio, random_state=random_seed, stratify=y_resampled)

        #------------------------------------------------Logistic Regression----------------------------

        # Standardize features after resampling
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

        # Initialize Logistic Regression with class weights set to 'balanced'
        model_lr = LogisticRegression(random_state=random_seed, class_weight='balanced')

        # Train the model on the undersampled and scaled training data
        model_lr.fit(X_train_scaled, y_train)

        # Predict on the test set
        predictions_lr_prob = model_lr.predict_proba(X_test_scaled)
        predictions_lr = [1 if k > bound_var else 0 for k in predictions_lr_prob[:,1]]
        #--------------------------------------------------XGBoost-----------------------------------------

        # Calculate the ratio of negative to positive samples for scaling
        num_pos = sum(y_train == 1)
        num_neg = sum(y_train == 0)
        scale_pos_weight = num_neg / num_pos


        model_xgb = xgb.XGBClassifier()
        # Train the model
        model_xgb.fit(X_train_scaled, y_train)

        # Predict on the test set
        predictions_xgb_prob = model_xgb.predict_proba(X_test_scaled)
        predictions_xgb = [1 if k > bound_var else 0 for k in predictions_xgb_prob[:,1]]
        #--------------------------------------------------EBM---------------------------------------------

        # Initialize the EBM model (Explainable Boosting Machine)
        ebm = ExplainableBoostingClassifier(random_state=random_seed)

        # Train the model
        ebm.fit(X_train_scaled, y_train)

        # Predict on the test set
        y_pred_ebm_proba = ebm.predict_proba(X_test_scaled)
        y_pred_ebm = [1 if k > bound_var else 0 for k in y_pred_ebm_proba[:,1]]
        


        # You can also get the predicted probabilities
        # y_prob_ebm = ebm.predict_proba(X_test_scaled)[:, 1]  # Probability for the positive class

        #--------------------------------------------------Logistic Regression(Benchmark)-----------------

        X_bench_train = X_train[['ficoCreditScore']]  # Continuous features
        X_bench_test = X_test[['ficoCreditScore']]

        # Standardize the features after undersampling
        scaler = StandardScaler()
        X_train_sbench = scaler.fit_transform(X_bench_train)
        X_test_sbench = scaler.transform(X_bench_test)

        # Train a Logistic Regression model with balanced class weights
        bench_model = LogisticRegression(class_weight='balanced', random_state=random_seed)
        bench_model.fit(X_train_sbench, y_train)


        # Make predictions on the test set and evaluate the model
        y_pred_bench_proba = bench_model.predict_proba(X_test_sbench)
        y_pred_bench = [1 if k > bound_var else 0 for k in y_pred_bench_proba[:,1]]
        
        tab1.header("Logistic Regression Classification Report:")
        tab1.table(pd.DataFrame.from_dict(classification_report(y_test, predictions_lr, output_dict=True)).transpose())

        tab1.header("XGBoost Classification Report:")
        tab1.table(pd.DataFrame.from_dict(classification_report(y_test, predictions_xgb, output_dict=True)).transpose())

        tab1.header("Explainable Boosting Machine Classification Report:")
        tab1.table(pd.DataFrame.from_dict(classification_report(y_test, y_pred_ebm, output_dict=True)).transpose())

        tab1.header("Logistic Regression Classification(Benchmark) Report:")
        tab1.table(pd.DataFrame.from_dict(classification_report(y_test, y_pred_bench, output_dict=True)).transpose())

        y_prob_bench = bench_model.predict_proba(X_test_sbench)[:, 1]  # Second column corresponds to class 1
        y_prob_lr = model_lr.predict_proba(X_test_scaled)[:, 1]  # Second column corresponds to class 1
        y_prob_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]  # Second column corresponds to class 1
        y_prob_ebm = ebm.predict_proba(X_test_scaled)[:, 1]  # Probability for the positive class

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC Curve
        plt.figure(figsize=(10, 8))

        # Benchmark Model ROC
        fpr_benchmark, tpr_benchmark, _ = roc_curve(y_test, y_prob_bench)
        auc_benchmark = roc_auc_score(y_test, y_prob_bench)
        ax.plot(fpr_benchmark, tpr_benchmark, label=f'Benchmark Model (AUC = {auc_benchmark:.2f})')

        # Logistic Regression ROC
        fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_prob_lr)
        auc_log_reg = roc_auc_score(y_test, y_prob_lr)
        ax.plot(fpr_log_reg, tpr_log_reg, label=f'Logistic Regression (AUC = {auc_log_reg:.2f})')

        # XGBoost ROC
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)
        ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.2f})')

        # EBM ROC
        fpr_ebm, tpr_ebm, _ = roc_curve(y_test, y_prob_ebm)
        auc_ebm = roc_auc_score(y_test, y_prob_ebm)
        ax.plot(fpr_ebm, tpr_ebm, label=f'EBM (AUC = {auc_ebm:.2f})')

        # Plot the diagonal line for random guessing
        #plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

        # Add labels and title
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Comparison of ROC Curves Across Different Models')
        ax.legend(loc='lower right')
        ax.grid(True)

        # Show plot
        
        tab1.pyplot(fig)
        tab1.header("Choose the interface model")
        interface_model = tab1.radio("Choose the model for user interface", ['Explainable Boosting Machine', 'Logistic Regression'])
        
        if (interface_model == 'Explainable Boosting Machine'):
            X_test['predicted'] = y_pred_ebm
            X_test_rejected = X_test[X_test['predicted'] == 1]
            X_test_rejected = X_test_rejected.drop(columns = ['predicted'])
            tab2.header("You are currently using: Explainable Boosting Machine(EBM)")
            user_input = {}
            if (tab2.toggle("Input Customization")):
                for feature in X.columns:
                    # Adjust widget type and parameters based on the feature name or type as needed
                    user_input[feature] = tab2.number_input(f"Enter value for {feature}", min_value=-1.0, max_value=1.0, step=0.01, value=X_train[feature].mean())
            else:
                for feature in X.columns:
                    # Adjust widget type and parameters based on the feature name or type as needed
                    user_input[feature] = X_train[feature].mean()
                
                # Convert inputs to a DataFrame or array based on the model's expected input format
            
            user_input_df = pd.DataFrame([user_input])
            
            if (tab2.toggle('Check Random Rejected Case')):
                reject_sample = X_test_rejected.sample(10)
                tab2.table(reject_sample)
                if (tab2.button("Use a sample case as input")):
                    which_one = tab2.number_input("Choose which one as input", 1, 10, 1, 1)
                    user_input = reject_sample.iloc[which_one - 1]
                    user_input_df = pd.DataFrame([user_input])
            
            tab2.header("Current input overview")
            tab2.table(user_input_df)
            prediction_prob = ebm.predict_proba(user_input_df)
        
            # Convert prediction to a readable output
            result = "Rejected" if prediction_prob[:,1] > bound_var  else "Approved"
            
            # Display the result
            tab2.write(f"Your Application Results is: **{result}**")
            
            if (tab2.button(f"Why my application is {result}")):
                ebm_global = ebm.explain_global()
                tab2.header("Here is how we consider your application")
                # Render the Plotly figure in Streamlit
                tab2.plotly_chart(ebm_global.visualize())
                
                tab2.header(f"Reason for being {result}")
                explanation = ebm.explain_local(user_input_df)

                # Create a readable explanation
                explanation_text = f"Decision: {result}\n Reasons:\n"

                # Iterate over the feature explanations
                for i, feature_name in enumerate(explanation.data(0)['names']):
                    feature_value = explanation.data(0)['values'][i]
                    contribution = explanation.data(0)['scores'][i]

                    # Check if the feature is in X_train
                    if feature_name in X_train.columns:
                        feature_median = X_train[feature_name].median()
                        if feature_value > feature_median:
                            value_desc = "High"
                        else:
                            value_desc = "Low"
                    else:
                        # Fallback in case the feature name is not found in X_train
                        value_desc = "average"

                    # Check if the contribution impacts the decision positively or negatively
                    if result == "Reject" and contribution > 0:
                        explanation_text += f" - {value_desc} {feature_name} ({feature_value}) contributed to rejection\n"
                    elif result == "Approved" and contribution > 0:
                        explanation_text += f" - {value_desc} {feature_name} ({feature_value}) contributed to approval\n"
                    else:
                        explanation_text += f" - {value_desc} {feature_name} ({feature_value}) had minimal impact on decision\n"

                tab2.write(explanation_text)
        else:
            X_test['predicted'] = predictions_lr
            X_test_rejected = X_test[X_test['predicted'] == 1]
            X_test_rejected = X_test_rejected.drop(columns = ['predicted'])
            tab2.header("You are currently using: Logistic Regression")
            user_input = {}
            if (tab2.toggle("Input Customization")):
                for feature in X.columns:
                    # Adjust widget type and parameters based on the feature name or type as needed
                    user_input[feature] = tab2.number_input(f"Enter value for {feature}", min_value=-1.0, max_value=1.0, step=0.01, value=X_train[feature].mean())
            else:
                for feature in X.columns:
                    # Adjust widget type and parameters based on the feature name or type as needed
                    user_input[feature] = X_train[feature].mean()
                
                # Convert inputs to a DataFrame or array based on the model's expected input format
            
            user_input_df = pd.DataFrame([user_input])
            
            if (tab2.toggle('Check Random Rejected Case')):
                reject_sample = X_test_rejected.sample(10)
                tab2.table(reject_sample)
                if (tab2.toggle("Use a sample case as input")):
                    which_one = tab2.number_input("Choose which one as input", 1, 10, 1, 1)
                    user_input = reject_sample.iloc[which_one - 1]
                    user_input_df = pd.DataFrame([user_input])
            
            tab2.header("Current input overview")
            tab2.table(user_input_df)
            prediction_prob = ebm.predict_proba(user_input_df)
        
            # Convert prediction to a readable output
            result = "Rejected" if prediction_prob[:,1] > bound_var  else "Approved"
            
            # Display the result
            tab2.write(f"Your Application Results is: **{result}**")

            if (tab2.button(f"Why my application is {result}")):
                feature_names = list(user_input_df.columns.values)

                # Get coefficients
                coefficients = model_lr.coef_[0]
                feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
                })

                # Sort by absolute importance
                feature_importance['Absolute Importance'] = feature_importance['Coefficient'].abs()
                feature_importance = feature_importance.sort_values(by='Absolute Importance', ascending=False)
                tab2.header("This is how we consider your application")
                fig3, ax3 = plt.subplots(figsize=(10, 8))    
                ax3.barh(feature_importance['Feature'][:10], feature_importance['Absolute Importance'][:10])
                ax3.set_xlabel('Absolute Importance')
                ax3.set_title('Top-10 Feature Importance in Logistic Regression')
                ax3.invert_yaxis()
                
                tab2.pyplot(fig3)
                
                multiplications = user_input_df.iloc[0] * coefficients
                multiplication_results = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients,
                    'Feature Value': user_input_df.iloc[0],
                    'Multiplication Result': multiplications
                })
                
                top_5_features = multiplication_results.reindex(
                    multiplication_results['Multiplication Result'].abs().sort_values(ascending=False).index
                ).head(5)

                tab2.header("Top 5 most important features based on multiplication results:")
                tab2.table(top_5_features)
                
                tab2.header("Multiplication results for each feature:")
                tab2.table(multiplication_results)
                