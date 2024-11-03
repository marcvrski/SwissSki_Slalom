import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/Users/marcgurber/SwissSki/SwissSki_Slalom/Merged_Course_and_Athlete_Times_by_Gate.csv'
data = pd.read_csv(file_path, delimiter=';')

# Filter data for relevant columns and remove rows with missing values for the selected features
features_data_selected = data[['time_difference', 'Turning Angle [°]', 'Offset [m]', 'Steepness [°]']].dropna()

# Exclude rows where time_difference is less than -1.0
features_data_filtered = features_data_selected[features_data_selected['time_difference'] >= -1.0]

# Perform linear regression and collect results for Turning Angle, Offset, and Steepness with filtered data
selected_features = ['Turning Angle [°]', 'Offset [m]', 'Steepness [°]']
filtered_results = {
    "Feature": [],
    "Slope": [],
    "Intercept": [],
    "R-squared": [],
    "p-value": [],
    "Significant (p < 0.05)": []
}

# Regression analysis for each feature against time_difference on the filtered dataset
for feature in selected_features:
    slope, intercept, r_value, p_value, std_err = linregress(features_data_filtered['time_difference'], features_data_filtered[feature])
    
    # Append results to dictionary
    filtered_results["Feature"].append(feature)
    filtered_results["Slope"].append(slope)
    filtered_results["Intercept"].append(intercept)
    filtered_results["R-squared"].append(r_value**2)
    filtered_results["p-value"].append(p_value)
    filtered_results["Significant (p < 0.05)"].append(p_value < 0.05)

# Convert results to a DataFrame
filtered_results_df = pd.DataFrame(filtered_results)
print("Filtered Significance of Selected Course Feature Relationships")
print(filtered_results_df)

# Plotting the relationships with the filtered data in a single figure with regression lines
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles_selected = [
    "Impact of Turning Angle on Time Difference",
    "Impact of Offset on Time Difference",
    "Impact of Steepness on Time Difference"
]

# Generate scatter plots with regression lines
for ax, feature, title in zip(axes.flatten(), selected_features, titles_selected):
    sns.regplot(x=features_data_filtered['time_difference'], y=features_data_filtered[feature], ax=ax, scatter_kws={'s': 10}, line_kws={'color': 'orange'})
    ax.set_xlabel("Time Difference (seconds)")
    ax.set_ylabel(feature)
    ax.set_title(title)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
