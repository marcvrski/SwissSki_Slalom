    # Plotting the relationships with the filtered data in independent figures with regression lines
    titles_selected = [
        "Impact of Turning Angle on Time Difference",
        "Impact of Offset on Time Difference",
        "Impact of Steepness on Time Difference"
    ]

    # Generate scatter plots with regression lines
    for feature, title in zip(selected_features, titles_selected):
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.regplot(x=features_data_filtered['time_difference'], y=features_data_filtered[feature], ax=ax, scatter_kws={'s': 10}, line_kws={'color': 'orange'})
        ax.set_xlabel("Time Difference (seconds)")
        ax.set_ylabel(feature)
        ax.set_title(title)
        st.pyplot(fig)
