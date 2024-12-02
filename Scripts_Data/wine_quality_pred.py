import sys
import pandas as pd
import numpy as np
from numpy import percentile, arange
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import cross_val_score, learning_curve, RepeatedKFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# soft code input file path
if len(sys.argv) != 3:
    exit("Usage: python3 wine_quality_pred.py <path/input_file.csv> <path/figs_dir>")
else:
    input_file_path = sys.argv[1]
    output_figures_path = sys.argv[2]
    print(f"Processing file: {input_file_path}")
    print(f"Figures saved to: {output_figures_path}")

# read the input file as a dataframe
df = pd.read_csv(input_file_path, sep=';')

### Exploratory data analysis
## Check if there is NA values
print("Number of missing values per column")
print(pd.isnull(df).sum())
print("\n")

## Histograms showing the data distribution of each feature
df.hist(figsize=(10, 8), bins=20)
# Adjusts subplot params so that subplots fit into the figure area
plt.tight_layout()
plt.rcParams.update({'font.size': 12})
plt.rc('font', size=12)
plt.rc('axes', titlesize=16)
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
plt.savefig(f"{output_figures_path}/histogram.png")
plt.close()

## For loop to draw histogram and qqplot of each variable
for column in df.columns:
    plt.figure(figsize=(10, 4))
    plt.rcParams.update({'font.size': 12})  # global
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=16)
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels

    # Histogram
    plt.subplot(1, 2, 1)
    df[column].hist(bins='auto')
    plt.title(f'Histogram of {column}')

    # Q-Q plot
    plt.subplot(1, 2, 2)
    stats.probplot(df[column], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {column}')

    plt.tight_layout()
    plt.savefig(f"{output_figures_path}/{column}_hist_qq.png")
    plt.close()

# Check the percentage of wine that are rated under 5 and above 7 for quality
values_under_5 = (df['quality'] < 5).mean() * 100
print(f"The percentage of wine rated under 5: {values_under_5:.2f}%")

values_above_7 = (df['quality'] > 7).mean() * 100
print(f"The percentage of wine rated under 7: {values_above_7:.2f}%")

## Draw the heatmap
# Correlation matrix
correlation_matrix = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 12})  # global
plt.rc('font', size=12)
plt.rc('axes', titlesize=16)
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels


# Draw and adjust the plot
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.xticks(rotation=30, ha='right')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig(f"{output_figures_path}/heatmap.png")
plt.close()

## Draw boxplots
# Adjust the fliers
flierprops = dict(marker='o', markersize=3, linestyle='none')

# Adjust the plot size and layout
df.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False, flierprops=flierprops, figsize=(15, 9))
plt.rcParams.update({'font.size': 12})  # global
plt.rc('font', size=12)
plt.rc('axes', titlesize=16)
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
plt.tight_layout()
plt.savefig(f"{output_figures_path}/IQR_boxplots.png")
plt.close()

## Data cleaning
# 1) Check for single value columns
single_value_columns = []
for column in df.columns[0:11]:
    unique_values = df[column].nunique()
    if unique_values == 1:
        single_value_columns.append(column)
print("single value columns:", single_value_columns)
df = df.drop(columns=single_value_columns)
print("------------\n")

# 2) Check for duplicated rows
duplicate_rows = df[df.duplicated()]
print("duplicate rows:", duplicate_rows)
print("number of duplicated rows:", len(duplicate_rows))
print("------------\n")
df = df.drop_duplicates()

# 3) Check for duplicated columns
duplicate_columns = []

for x in range(df.shape[1]):
    # Take column at xth index
    col = df.iloc[:, x]

    # Iterate through all the columns after the current column
    for y in range(x + 1, df.shape[1]):
        # Take column at yth index
        other_col = df.iloc[:, y]

        # Check if two columns at x & y are equal
        if col.equals(other_col):
            # Store the pair of column names as a tuple
            duplicate_columns.append(
                (df.columns.values[x], df.columns.values[y]))
print("duplicate columns:", duplicate_columns)
df = df.drop(columns=duplicate_columns)
print("------------\n")

# 4) Check for outliers
# Initialize an empty list to store row numbers of outliers
nrml_dist_cols = ["pH", "quality"]
outliers_rows = set()
outliers_count = {}

# Loop through each column
for column in df.columns:
    outliers_count[column] = 0

    if column in nrml_dist_cols:
        # Extract column data
        column_data = df[column]
        # Calculate mean and standard deviation for the current column
        mean = column_data.mean()
        std = column_data.std()

        # Calculate cutoff for outliers
        cut_off = std * 3
        lower = mean - cut_off
        upper = mean + cut_off

        # Identify outliers for the current column and store their row numbers
        column_outliers = df[column][(df[column] < lower) | (df[column] > upper)]

    else:
        # Extract column data
        column_data = df[column]

        # calculate the inter-quartile range
        q25, q75 = percentile(column_data, 25), percentile(column_data, 75)
        iqr = q75 - q25

        # calculate the outlier cutoff: k=1.5
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off

        # identify outliers
        column_outliers = df[column][(df[column] < lower) | (df[column] > upper)]

    outliers_count[column] += len(column_outliers)
    outliers_rows.update(column_outliers.index)
    # Extend the list of outliers row numbers
outliers_rows = list(outliers_rows)

for column, count in outliers_count.items():
    print(f"{column} has {count} outliers")

# Remove rows with outlier row numbers from the original DataFrame
df_cleaned = df.drop(outliers_rows)
print(len(outliers_rows))
print(df_cleaned)
print("------------\n")

# Histograms showing the data distribution of each feature after data filtering
df_cleaned.hist(figsize=(10, 8), bins=20)
plt.rcParams.update({'font.size': 12})  # global
plt.rc('font', size=12)
plt.rc('axes', titlesize=16)
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
# Adjusts subplot params so that subplots fit into the figure area
plt.tight_layout()
plt.savefig(f"{output_figures_path}/histogram_filtered.png")
plt.close()

## Dataset splitting
# Split the dataset into features (X) and target variable (y)
X = df_cleaned.drop('quality', axis=1)  # Features
y = df_cleaned['quality']  # Target variable

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=100))

# Print the shape of the training and testing subsets
print(X_train)
print(X_test)
print(y_train)
print(y_test)
print("------------\n")


### Feature Selection, Cross-validation and Preliminary Model Comparison
## Function for drawing Learning Curve
def plot_learning_curve(estimator, X, y, cv, modelname, BeforeOrAfter,
                        train_sizes=np.linspace(0.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='r2', n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 12})  # global
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=16)
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    plt.title(f'Learning Curves {BeforeOrAfter} Optimization ({modelname})')
    plt.ylim(0, 1)
    plt.xlabel('Training examples')
    plt.ylabel('R^2 Score')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, color='r', alpha=0.1)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, color='g', alpha=0.1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')
    plt.legend(loc='best')
    plt.savefig(f"{output_figures_path}/{modelname}_learning_curve_{BeforeOrAfter}.png")
    plt.close()

# Define the model
models = {
    'LR': LinearRegression(),
    'SVM': SVR(gamma='auto'),
    'RF': RandomForestRegressor(random_state=1),
    'GBR': GradientBoostingRegressor(random_state=1)
}

# Define the scoring methods
scoring = {
    'MAE':'neg_mean_absolute_error',
    'MSE':'neg_mean_squared_error',
    'R^2':'r2'
}

# Initialize a dict to store the results
results = {score_name: [] for score_name in scoring.keys()}

# Define kfold
kfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

print('\nModel evaluation - training')

# Dictionary to store accumulated importance
accumulated_importance = {name: np.zeros(len(X_train.columns)) for name in models.keys()}

for name, model in models.items():
    print(f'-------------------------\nModel: {name}')

    for train_index, test_index in kfold.split(X_train, y_train):
        # Split data
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit model and compute permutation importance
        model.fit(X_train_fold, y_train_fold)
        perm_importance = permutation_importance(model, X_test_fold, y_test_fold, n_repeats=10, random_state=0,
                                                 scoring='r2')

        # Accumulate the importance scores
        accumulated_importance[name] += perm_importance.importances_mean / kfold.get_n_splits()

        # Feature selection based on importance
    sorted_indices = np.argsort(accumulated_importance[name])[::-1]
    selected_features = sorted_indices[:5]  # Select top 5 features
    X_train_important = X_train.iloc[:, selected_features]

    # Learning curves for selected features
    plot_learning_curve(model, X_train_important, y_train, kfold, name, "Before")

    # Collect and store cross-validation results
    for score_key, score_method in scoring.items():
        score_values = cross_val_score(model, X_train_important, y_train, scoring=score_method,
                                       cv=kfold)
        results[score_key].append(score_values)
        print(
            f"{name} - {score_key}: Mean={score_values.mean():.3f}, Std={score_values.std():.3f}")

# Plotting feature importance for each model
for name in models.keys():
    sorted_indices = np.argsort(accumulated_importance[name])[::-1]
    top_indices = sorted_indices[:10]  # Select the indices of the top 10 features

    # Plot feature importance in descending order
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 12})  # global
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=16)
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    plt.barh(np.arange(len(top_indices)), accumulated_importance[name][top_indices])
    plt.yticks(ticks=np.arange(len(top_indices)), labels=X_train.columns[top_indices])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance for {name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_figures_path}/{name}_feature_importance.png")
    plt.close()

# Plot boxplots for each scoring metric across all models
for score_name in scoring.keys():
    plt.rcParams.update({'font.size': 12})  # global
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=16)
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    plt.figure(figsize=(10, 6))
    plt.boxplot([results[score_name][i] for i in range(len(models))], labels=models.keys())
    plt.title(f'Algorithm Comparison - {score_name} Before Optimization')
    plt.ylabel(f'{score_name} Score')
    plt.savefig(f"{output_figures_path}/{score_name}_model_comparison.png")
    plt.close()

# Improve accuracy with hyper-parameter tuning
print('\nModel evaluation - hyper-parameter tuning')
print('-----------------------------------------')

model_params = dict()
model_params['LR'] = dict()
model_params['LR']['fit_intercept'] = [True, False]

model_params['SVM'] = dict()
model_params['SVM']['C'] = list(arange(0.5, 1.5, 0.01))

model_params['RF'] = dict()
model_params['RF']['n_estimators'] = [50, 100, 200]
model_params['RF']['max_depth'] = [None, 10, 20]
model_params['RF']['min_samples_split'] = [2, 5, 10]
model_params['RF']['min_samples_leaf'] = [1, 2, 4]

model_params['GBR'] = dict()
model_params['GBR']['n_estimators'] = [50, 100, 200]
model_params['GBR']['learning_rate'] = [0.01, 0.1, 0.2]
model_params['GBR']['max_depth'] = [3, 5, 10]
model_params['GBR']['min_samples_split'] = [2, 5, 10]
model_params['GBR']['min_samples_leaf'] = [1, 2, 4]

best_params = dict()
for name, model in models.items():
    rand_search = RandomizedSearchCV(estimator=model,
                                     param_distributions=model_params[name],
                                     n_iter=5, n_jobs=-1, cv=kfold,
                                     scoring='neg_mean_absolute_error')
    rand_result = rand_search.fit(X_train_important, y_train)
    print("Model %s -- Best: %f using %s" % (name, rand_result.best_score_,
                                             rand_result.best_params_))
    best_params[name] = rand_result.best_params_


# Re-initialize models using best parameter settings
optimized_models = []
(optimized_models.append
 (('LR', LinearRegression(fit_intercept=best_params['LR']['fit_intercept']))))

(optimized_models.append
 (('SVM', SVR(gamma='auto',
              C=best_params['SVM']['C']))))

(optimized_models.append
(('RF', RandomForestRegressor(n_estimators=best_params['RF']['n_estimators'],
                               max_depth=best_params['RF']['max_depth'],
                               min_samples_split=best_params['RF']['min_samples_split'],
                               min_samples_leaf=best_params['RF']['min_samples_leaf']))))

(optimized_models.append
 (('GBR', GradientBoostingRegressor(n_estimators=best_params['GBR']['n_estimators'],
                                   learning_rate=best_params['GBR']['learning_rate'],
                                   max_depth=best_params['GBR']['max_depth'],
                                   min_samples_split=best_params['GBR']['min_samples_split'],
                                   min_samples_leaf=best_params['GBR']['min_samples_leaf']))))


print('\nModel evaluation - optimized')
print('--------------------------')

# Initialize lists to store results and model names
results_mae = []
results_mse = []
results_r2 = []
names = []

for name, model in optimized_models:
    # Perform cross-validation for MAE, MSE, and R2
    cv_results_mae = cross_val_score(model, X_train_important, y_train, cv=kfold,
                                      scoring='neg_mean_absolute_error',
                                      n_jobs=-1, error_score='raise')
    cv_results_mse = cross_val_score(model, X_train_important, y_train, cv=kfold,
                                      scoring='neg_mean_squared_error',
                                      n_jobs=-1, error_score='raise')
    cv_results_r2 = cross_val_score(model, X_train_important, y_train, cv=kfold,
                                     scoring='r2',
                                     n_jobs=-1, error_score='raise')

    # Append results and model name to respective lists
    results_mae.append(cv_results_mae)
    results_mse.append(cv_results_mse)
    results_r2.append(cv_results_r2)
    names.append(name)

    # Print the mean and standard deviation of each metric
    print(f"{name} MAE: Mean={np.absolute(cv_results_mae.mean()):.3f}, Std={np.absolute(cv_results_mae.std()):.3f}")
    print(f"{name} MSE: Mean={np.absolute(cv_results_mse.mean()):.3f}, Std={np.absolute(cv_results_mse.std()):.3f}")
    print(f"{name} R^2: Mean={np.absolute(cv_results_r2.mean()):.3f}, Std={np.absolute(cv_results_r2.std()):.3f}")

# Create boxplots for MAE, MSE, and R2

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 12})  # global
plt.rc('font', size=12)
plt.rc('axes', titlesize=16)
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
plt.boxplot(results_mae, labels=names)
plt.title('Algorithm Comparison - MAE After Optimization')
plt.ylabel('MAE Score')
plt.savefig(f"{output_figures_path}/MAE_after.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 12})  # global
plt.rc('font', size=12)
plt.rc('axes', titlesize=16)
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
plt.boxplot(results_mse, labels=names)
plt.title('Algorithm Comparison - MSE After Optimization')
plt.ylabel('MSE Score')
plt.savefig(f"{output_figures_path}/MSE_after.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 12})  # global
plt.rc('font', size=12)
plt.rc('axes', titlesize=16)
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
plt.boxplot(results_r2, labels=names)
plt.title('Algorithm Comparison - R^2 After Optimization')
plt.ylabel('R^2 Score')
plt.savefig(f"{output_figures_path}/R2_after.png")
plt.close()



# Fit and save optimized models
for name, model in optimized_models:
    model.fit(X_train, y_train)
    filename = name + '_optimized_model.sav'
    joblib.dump(model, filename)

    # Plot learning curve
    plot_learning_curve(model, X_train, y_train, kfold, name, "After")

# Testing model performance on test set
print('\nModel testing')
print('-------------')
for name, model in optimized_models:
    model.fit(X_train, y_train)
    predicted_results = model.predict(X_test)
    mae_result = mean_absolute_error(predicted_results, y_test)
    print('%s Mean Absolute Error: %f' % (name, mae_result))
    mse_result = mean_squared_error(y_test, predicted_results)
    print('%s Mean Squared Error: %f' % (name, mse_result))
    r2_result = r2_score(y_test, predicted_results)
    print('%s R^2: %f' % (name, r2_result))
    plt.scatter(y_test, predicted_results)
    plt.rcParams.update({'font.size': 12})  # global
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=16)
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    plt.title('Test results for ' + name)
    plt.xlabel('Ground truth')
    plt.ylabel('Predicted results')
    m, b = np.polyfit(y_test, predicted_results,1)  # Fit a first-degree polynomial (linear regression)
    plt.plot(y_test, m * y_test + b, color='red')  # Plot the regression line
    plt.savefig(f"{output_figures_path}/{name}_test_results.png")
    plt.close()