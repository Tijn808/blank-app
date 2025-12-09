import pandas as pd
import streamlit as st

st.markdown("##  Research Questions")

st.markdown("### Initial Research Question")
st.info('Can we predict the onset of diabetes in the Framingham Heart Study population using baseline demographic, lifestyle, and clinical variables?')

st.markdown("### Refined Research Question")
st.info('Can we identify individuals currently positive for or at high risk of diabetes within the Framingham Heart Study population, using readily available baseline demographic, lifestyle, and clinical variables such as age, sex, BMI, blood pressure, cholesterol, glucose, and smoking status?')

data = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')
data.head()
data_raw = data.copy(deep=True) #so the data keeps it original state
selected_columns = [
    'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI',
    'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'GLUCOSE',
    'HYPERTEN', 'DIABETES'
]
st.markdown("## Selected Data Set")

df = data[selected_columns]
st.dataframe(df, use_container_width=True, height=300)
st.write('Selected columns are: age, sex, totchol, sysbp, diabp, cursmoke, cigpday, BMI, bpmeds, prevchd, prevap, prevmi, prevstrk, prevhyp, glucose, hyperten & diabetes')

st.markdown('### Missing Values')
st.dataframe(df.isnull().sum(), use_container_width=True, height=300)
from sklearn.model_selection import train_test_split

st.markdown('## Imputation & Outliers')


with st.expander ('### Imputation'):
    st.info('We used iterative imputation to fill for the missing values.')

# Separate features (X) and target (y)
X = df.drop('DIABETES', axis=1)
y = df['DIABETES']

# Split the data into training and testing sets, stratified by 'DIABETES'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Display the shapes of the resulting sets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
X_test.head()
X_train.head()
y_test.head()
y_train.head()
X_train.describe()
import matplotlib.pyplot as plt
import seaborn as sns

# Get numerical columns from X_train
numerical_cols = X_train.select_dtypes(include=['number']).columns

# Identify binary columns (assuming binary columns only contain 0 and 1, or NaN)
binary_cols = [col for col in numerical_cols if X_train[col].dropna().isin([0, 1]).all()]

# Filter out binary columns for box plotting
meaningful_numerical_cols = [col for col in numerical_cols if col not in binary_cols]

# Check if there are any meaningful columns to plot
if len(meaningful_numerical_cols) == 0:
    st.write("No meaningful numerical columns found to plot after excluding binary ones.")
else:
    with st.expander("### Outliers"):

        st.markdown("#### Box Plots")

        # Add a selectbox to choose which column to plot
        selected_column = st.selectbox(
            "Select a column to see the uncapped boxplots:",
            meaningful_numerical_cols
        )

        # Create boxplot for the selected column
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=X_train[selected_column], ax=ax)
        ax.set_title(f'Box Plot of {selected_column}', fontsize=14, fontweight='bold')
        ax.set_xlabel(selected_column, fontsize=12)

        st.pyplot(fig)
    
        st.pyplot(fig)
        columns_to_cap = ['TOTCHOL', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'GLUCOSE']
        capping_values = {}

        for col in columns_to_cap:
        # Calculate the 99th percentile for the current column in X_train
        percentile_99 = X_train[col].quantile(0.99)
    
    # Store the 99th percentile value
        capping_values[col] = percentile_99
    
    # Cap the values in X_train at the 99th percentile
        X_train[col] = X_train[col].clip(upper=percentile_99)

print("99th percentile capping values:")
print(capping_values)
print("\nX_train after capping outliers (first 5 rows of capped columns):")
print(X_train[columns_to_cap].head())
for col in columns_to_cap:
    # Apply the capping values (calculated from X_train) to X_test
    X_test[col] = X_test[col].clip(upper=capping_values[col])

print("\nX_test after capping outliers (first 5 rows of capped columns):")
print(X_test[columns_to_cap].head())

st.markdown("#### Box Plots with Capped Outliers")

# Add a selectbox to choose which capped column to plot
selected_capped_column = st.selectbox("Select a column to see the capped boxplots:", columns_to_cap)

# Create side-by-side boxplots for the selected column
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.boxplot(x=X_train[selected_capped_column], ax=axes[0])
axes[0].set_title(f'X_train: Box Plot of {selected_capped_column} (Capped)', fontsize=12, fontweight='bold')
axes[0].set_xlabel(selected_capped_column)

sns.boxplot(x=X_test[selected_capped_column], ax=axes[1])
axes[1].set_title(f'X_test: Box Plot of {selected_capped_column} (Capped)', fontsize=12, fontweight='bold')
axes[1].set_xlabel(selected_capped_column)

plt.tight_layout()
st.pyplot(fig)

missing_cols_numeric = ['TOTCHOL', 'CIGPDAY', 'BMI', 'BPMEDS', 'GLUCOSE']
median_values = {}

# Impute missing values in X_train
for col in missing_cols_numeric:
    median_val = X_train[col].median()
    median_values[col] = median_val
    X_train[col].fillna(median_val, inplace=True)

print("Median values used for imputation:")
print(median_values)
print("\nMissing values after imputation in X_train:")
print(X_train[missing_cols_numeric].isnull().sum())
missing_cols_numeric = ['TOTCHOL', 'CIGPDAY', 'BMI', 'BPMEDS', 'GLUCOSE']
median_values = {}

# Impute missing values in X_train using medians from X_train
for col in missing_cols_numeric:
    median_val = X_train[col].median()
    median_values[col] = median_val
    X_train[col] = X_train[col].fillna(median_val)

# Impute missing values in X_test using medians calculated from X_train
for col in missing_cols_numeric:
    X_test[col] = X_test[col].fillna(median_values[col])

print("Median values used for imputation:")
print(median_values)

print("\nMissing values after imputation in X_train:")
print(X_train[missing_cols_numeric].isnull().sum())

print("\nMissing values after imputation in X_test:")
print(X_test[missing_cols_numeric].isnull().sum())
import matplotlib.pyplot as plt
import seaborn as sns

# Identify numerical columns from X_train
numerical_cols = X_train.select_dtypes(include=['number']).columns

# Identify binary columns (assuming binary columns only contain 0 and 1)
binary_cols = [col for col in numerical_cols if X_train[col].dropna().isin([0, 1]).all()]

# Filter out binary columns for histogram plotting
meaningful_numerical_cols = [col for col in numerical_cols if col not in binary_cols]

# Check if there are any meaningful columns to plot
if len(meaningful_numerical_cols) == 0:
    st.write("No meaningful numerical columns found to plot distributions for after excluding binary ones.")
else:
    st.markdown("### Distributions")
    
    # Add a selectbox to choose which column to plot
    selected_dist_column = st.selectbox("Select a column to see the distribution:", meaningful_numerical_cols)
    
    # Create histogram for the selected column
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(x=X_train[selected_dist_column], kde=True, ax=ax)
    ax.set_title(f'Distribution of {selected_dist_column} in X_train', fontsize=14, fontweight='bold')
    ax.set_xlabel(selected_dist_column, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    st.pyplot(fig)
    import matplotlib.pyplot as plt
import seaborn as sns

# Combine X_train and y_train into a single DataFrame
# Reset index of y_train to align with X_train if not already aligned
y_train_reset = y_train.reset_index(drop=True)
X_train_reset = X_train.reset_index(drop=True)

df_combined = pd.concat([X_train_reset, y_train_reset], axis=1)

st.markdown('## Correlation Heatmap')
correlation_matrix = df_combined.corr()

fig_corr, ax_corr = plt.subplots(figsize=(18, 15), constrained_layout=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
ax_corr.set_title('Correlation Heatmap of Features in X_train and y_train')
st.pyplot(fig_corr)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Create a count plot for the y_train Series (render with Streamlit)
st.markdown('### Distribution of Diabetes in Training Data')
fig_count, ax_count = plt.subplots(figsize=(6, 4), constrained_layout=True)
sns.countplot(x=y_train, ax=ax_count)
ax_count.set_title('Distribution of DIABETES in y_train')
ax_count.set_xlabel('DIABETES')
ax_count.set_ylabel('Count')
st.pyplot(fig_count)

key_numerical_features = ['GLUCOSE', 'BMI', 'SYSBP', 'AGE', 'TOTCHOL']

# Reset indices to align X_train and y_train
X_train_reset = X_train.reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)

# Combine features and target into a single DataFrame
df_combined_viz = pd.concat([X_train_reset[key_numerical_features], y_train_reset], axis=1)

# Streamlit selectbox to choose a feature
selected_feature = st.selectbox(
    "Select a feature to see:",
    key_numerical_features
)

# Plot only the selected feature
fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(x='DIABETES', y=selected_feature, data=df_combined_viz, ax=ax)
ax.set_title(f'Distribution of {selected_feature} by DIABETES Outcome')
ax.set_xlabel('DIABETES')
ax.set_ylabel(selected_feature)

# Display the plot in Streamlit
st.pyplot(fig)

from sklearn.preprocessing import StandardScaler

# Reuse meaningful_numerical_cols from previous step
# These are the columns that are numerical and not considered binary (0, 1)
meaningful_numerical_cols = ['AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'GLUCOSE']

# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler only on X_train for the identified columns
X_train[meaningful_numerical_cols] = scaler.fit_transform(X_train[meaningful_numerical_cols])

# Transform both X_train and X_test using the fitted scaler
X_test[meaningful_numerical_cols] = scaler.transform(X_test[meaningful_numerical_cols])

print("Descriptive statistics for scaled numerical columns in X_train:")
print(X_train[meaningful_numerical_cols].describe())

print("\nX_train after scaling (first 5 rows of scaled columns):")
print(X_train[meaningful_numerical_cols].head())

print("\nDescriptive statistics for scaled numerical columns in X_test:")
print(X_test[meaningful_numerical_cols].describe())

print("\nX_test after scaling (first 5 rows of scaled columns):")
print(X_test[meaningful_numerical_cols].head())
from sklearn.linear_model import LogisticRegression

# Instantiate Logistic Regression model with class_weight='balanced'
log_reg_model = LogisticRegression(class_weight='balanced', random_state=42, solver='liblinear')

# Train the model using the scaled X_train and y_train
log_reg_model.fit(X_train, y_train)

print("Logistic Regression model trained successfully with class_weight='balanced'.")
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

# 2. Use the trained log_reg_model to make predictions on X_test
y_pred = log_reg_model.predict(X_test)
y_proba = log_reg_model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

# 3. Print the classification report
st.markdown('## Logistic Regression')
st.markdown('### Classification Report')

# Convert classification report to a table
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df, use_container_width=True, height=210)

# 4. Create and display a confusion matrix (render in browser)
st.markdown('### Confusion Matrix')
fig_cm, ax_cm = plt.subplots(figsize=(8, 6), constrained_layout=True)
ConfusionMatrixDisplay.from_estimator(log_reg_model, X_test, y_test, cmap=plt.cm.Blues, ax=ax_cm)
ax_cm.set_title('Confusion Matrix for Logistic Regression')
st.pyplot(fig_cm)

# 5. Create and display an ROC curve (render in browser)
st.markdown('### ROC Curve')
fig_roc, ax_roc = plt.subplots(figsize=(8, 6), constrained_layout=True)
RocCurveDisplay.from_estimator(log_reg_model, X_test, y_test, name='Logistic Regression', ax=ax_roc)
ax_roc.set_title('ROC Curve for Logistic Regression')
ax_roc.plot([0, 1], [0, 1], 'r--') # Plot random guess line
st.pyplot(fig_roc)

from sklearn.metrics import precision_recall_fscore_support, classification_report, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

st.markdown('## Logistic Regression with Threshold')
# Get probabilities for the positive class (class 1)
y_proba = log_reg_model.predict_proba(X_test)[:, 1]

# Define a specific threshold to use
optimal_f1_threshold = 0.91 # Using the previously identified optimal F1-score threshold

# Classify predictions based on this specific threshold
y_pred_optimal = (y_proba >= optimal_f1_threshold).astype(int)

st.markdown(f'### Classification Report with Threshold={optimal_f1_threshold:.2f}')

# Convert classification report to a table
report_thresh = classification_report(y_test, y_pred_optimal, zero_division=0, output_dict=True)
report_thresh_df = pd.DataFrame(report_thresh).transpose()
st.dataframe(report_thresh_df, use_container_width=True, height=210)

# Create and display a confusion matrix for this threshold (render in browser)
st.markdown('### Confusion Matrix with Threshold')
fig_thresh, ax_thresh = plt.subplots(figsize=(8, 6), constrained_layout=True)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_optimal, cmap=plt.cm.Blues, ax=ax_thresh)
ax_thresh.set_title(f'Confusion Matrix for Logistic Regression (Threshold={optimal_f1_threshold:.2f})')
st.pyplot(fig_thresh)

