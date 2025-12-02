import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')
data.head()
data_raw = data.copy(deep=True) #so the data keeps it original state
selected_columns = [
    'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI',
    'BPMEDS', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'GLUCOSE',
    'HYPERTEN', 'DIABETES'
]

df = data[selected_columns]

print("\nInformation about the selected_features DataFrame:")
df.info()
df.head()
df.isnull().sum()
from sklearn.model_selection import train_test_split

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

# Set up the plotting area dynamically based on the number of meaningful columns
num_plots = len(meaningful_numerical_cols)

# Check if there are any meaningful columns to plot
if num_plots == 0:
    print("No meaningful numerical columns found to plot after excluding binary ones.")
else:
    fig_height = num_plots * 3  # Adjust height per plot for better visibility

    plt.figure(figsize=(5, fig_height))

    for i, column in enumerate(meaningful_numerical_cols):
        plt.subplot(num_plots, 1, i + 1) # Create a subplot for each feature
        sns.boxplot(x=X_train[column])
        plt.title(f'Box Plot of {column}')
        plt.xlabel(column)
        plt.tight_layout() # Adjust layout to prevent overlapping titles/labels

    plt.show()
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
import matplotlib.pyplot as plt
import seaborn as sns

# Columns for which outliers were capped
columns_to_cap = ['TOTCHOL', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'GLUCOSE']

# Set up the plotting area dynamically based on the number of columns
num_plots = len(columns_to_cap)
fig_height = num_plots * 4 # Adjust height per plot for better visibility

print("Box plots for X_train after capping:")
plt.figure(figsize=(15, fig_height))
for i, column in enumerate(columns_to_cap):
    plt.subplot(num_plots, 2, 2*i + 1) # Left column for X_train
    sns.boxplot(x=X_train[column])
    plt.title(f'X_train: Box Plot of {column} (Capped)')
    plt.xlabel(column)
    
    plt.subplot(num_plots, 2, 2*i + 2) # Right column for X_test
    sns.boxplot(x=X_test[column])
    plt.title(f'X_test: Box Plot of {column} (Capped)')
    plt.xlabel(column)

plt.tight_layout()
plt.show()
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

# Set up the plotting area dynamically based on the number of meaningful columns
num_plots = len(meaningful_numerical_cols)

# Check if there are any meaningful columns to plot
if num_plots == 0:
    print("No meaningful numerical columns found to plot histograms for after excluding binary ones.")
else:
    fig_height = num_plots * 4  # Adjust height per plot for better visibility

    plt.figure(figsize=(10, fig_height))

    for i, column in enumerate(meaningful_numerical_cols):
        plt.subplot(num_plots, 1, i + 1)  # Create a subplot for each feature
        sns.histplot(x=X_train[column], kde=True) # Use kde=True for density estimation
        plt.title(f'Distribution of {column} in X_train')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    plt.tight_layout()  # Adjust layout to prevent overlapping titles/labels
    plt.show()
    import matplotlib.pyplot as plt
import seaborn as sns

# Combine X_train and y_train into a single DataFrame
# Reset index of y_train to align with X_train if not already aligned
y_train_reset = y_train.reset_index(drop=True)
X_train_reset = X_train.reset_index(drop=True)

df_combined = pd.concat([X_train_reset, y_train_reset], axis=1)

# Calculate the correlation matrix
correlation_matrix = df_combined.corr()

# Plot the correlation heatmap
plt.figure(figsize=(18, 15)) # Adjust figure size for better readability
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features in X_train and y_train')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Create a count plot for the y_train Series
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train)
plt.title('Distribution of DIABETES in y_train')
plt.xlabel('DIABETES')
plt.ylabel('Count')
plt.show()

# 2. Define a list named key_numerical_features
key_numerical_features = ['GLUCOSE', 'BMI', 'SYSBP', 'AGE', 'TOTCHOL']

# 3. Create a temporary DataFrame by concatenating X_train and y_train
# Reset index for both X_train and y_train to ensure proper alignment
X_train_reset = X_train.reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)

df_combined_viz = pd.concat([X_train_reset[key_numerical_features], y_train_reset], axis=1)

# 4. Iterate through each feature in key_numerical_features and create box plots
num_features = len(key_numerical_features)
plt.figure(figsize=(10, num_features * 4)) # Adjust figure size dynamically

for i, feature in enumerate(key_numerical_features):
    plt.subplot(num_features, 1, i + 1)
    sns.boxplot(x='DIABETES', y=feature, data=df_combined_viz)
    plt.title(f'Distribution of {feature} by DIABETES Outcome')
    plt.xlabel('DIABETES')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()
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
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 4. Create and display a confusion matrix
print("\nConfusion Matrix:")
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(log_reg_model, X_test, y_test, cmap=plt.cm.Blues, ax=ax)
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

# 5. Create and display an ROC curve and calculate ROC-AUC score
print("\nROC Curve:")
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(log_reg_model, X_test, y_test, name='Logistic Regression', ax=ax)
plt.title('ROC Curve for Logistic Regression')
plt.plot([0,
1], [0, 1], 'r--') # Plot random guess line
plt.show()
from sklearn.metrics import precision_recall_fscore_support, classification_report, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Get probabilities for the positive class (class 1)
y_proba = log_reg_model.predict_proba(X_test)[:, 1]

# Define a specific threshold to use
optimal_f1_threshold = 0.91 # Using the previously identified optimal F1-score threshold

# Classify predictions based on this specific threshold
y_pred_optimal = (y_proba >= optimal_f1_threshold).astype(int)

print(f"Using Threshold: {optimal_f1_threshold:.2f}")
print("\nClassification Report with this Threshold:")
print(classification_report(y_test, y_pred_optimal, zero_division=0))

# Create and display a confusion matrix for this threshold
print("\nConfusion Matrix with this Threshold:")
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_optimal, cmap=plt.cm.Blues, ax=ax)
plt.title(f'Confusion Matrix for Logistic Regression (Threshold={optimal_f1_threshold:.2f})')
plt.show()