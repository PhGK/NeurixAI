"""Script for functions used for evaluation."""

from scipy.stats import spearmanr

import pandas as pd

import matplotlib.pyplot as plt


def plot_ordered_r_scores(dataframe: pd.DataFrame, group_by_feature: str) -> pd.DataFrame:
    """
    Plot the R scores for different groups based on a specified feature.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        group_by_feature (str): The feature to group by.

    Returns
    -------
        dataframe (pd.DataFrame): The DataFrame contains the r scores for each group.
    """
    # Calculate R score for each group
    r_scores = []
    for group_name, group_data in dataframe.groupby(group_by_feature):
        r_value, _ = spearmanr(group_data["Prediction"], group_data["Target"])
        r_scores.append({"Group": group_name, "R Score": r_value})

    # Convert the list of dictionaries to DataFrame
    r_scores_df = pd.DataFrame(r_scores)

    # Sort the DataFrame by 'R Score' in descending order
    r_scores_df = r_scores_df.sort_values(by="R Score", ascending=False)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(r_scores_df["Group"], r_scores_df["R Score"], color="skyblue")
    plt.xlabel(group_by_feature)
    plt.ylabel("R Score")
    plt.title("R Scores for Different " + group_by_feature)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    return r_scores_df
