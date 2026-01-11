"""
2.2 Exploratory Data Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def plot_histogram(df: pd.DataFrame, column: str, bins: int = 20, interactive: bool = False):
    """
    Plot a histogram for a specified column. Supports static Matplotlib/Seaborn or interactive Plotly plots.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    column : str
        Name of the numeric column to visualize.
    bins : int, optional
        Number of bins for the histogram (default is 20).
    interactive : bool, optional
        If True, use Plotly for interactive plot; otherwise, use Matplotlib (default is False).

    Returns:
    --------
    None
        Displays the histogram plot.

    Raises:
    -------
    ValueError
        If the specified column does not exist in df.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if interactive:
        fig = px.histogram(df, x=column, nbins=bins, title=f"Distribution of {column}")
        fig.show()
        fig.write_html(f"../reports/figures/{column}_histogram.html")
    else:
        plt.figure(figsize=(8,5))
        sns.histplot(df[column], bins=bins, kde=True)
        plt.title(f"Distribution of {column}")
        plt.savefig(f"../reports/figures/{column}_histogram.png")
        plt.show()


def plot_box(df: pd.DataFrame, column: str, interactive: bool = False):
    """
    Plot a boxplot for a specified column. Supports static or interactive plots.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    column : str
        Name of the column to visualize.
    interactive : bool, optional
        If True, use Plotly for interactive plot; otherwise, use Matplotlib/Seaborn (default is False).

    Returns:
    --------
    None
        Displays the boxplot.

    Raises:
    -------
    ValueError
        If the specified column does not exist in df.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if interactive:
        fig = px.box(df, y=column, title=f"Box Plot of {column}")
        fig.show()
        fig.write_html(f"../reports/figures/{column}_boxplot.html")
    else:
        plt.figure(figsize=(6,4))
        sns.boxplot(y=df[column])
        plt.title(f"Box Plot of {column}")
        plt.savefig(f"../reports/figures/{column}_boxplot.png")
        plt.show()


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue: str = None, interactive: bool = False, legend_pos=None):
    """
    Plot a scatter plot of two numeric columns, optionally colored by a categorical column.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    x_col : str
        Column name for the x-axis.
    y_col : str
        Column name for the y-axis.
    hue : str, optional
        Column name for color grouping (default is None).
    interactive : bool, optional
        If True, use Plotly for interactive plot; otherwise, use Matplotlib/Seaborn (default is False).
    legend_pos : dict, optional
        Dictionary of legend positioning parameters (e.g., {'bbox_to_anchor': (0.95, 1), 'loc': 'upper right'}).
        Default is None.

    Returns:
    --------
    None
        Displays the scatter plot.

    Raises:
    -------
    ValueError
        If x_col or y_col does not exist in df.
    """
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if interactive:
        fig = px.scatter(df, x=x_col, y=y_col, color=hue, title=f"{y_col} vs {x_col}")
        fig.show()
        fig.write_html(f"../reports/figures/{y_col}_vs_{x_col}_scatterplot.html")
    else:
        plt.figure(figsize=(8,5))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue)
        plt.title(f"{y_col} vs {x_col}")
        if legend_pos:
            plt.legend(**legend_pos)
        plt.savefig(f"../reports/figures/{y_col}_vs_{x_col}_scatterplot.png")
        plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, interactive: bool = False):
    """
    Plot a correlation heatmap for numeric features in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing numeric columns to correlate.
    interactive : bool, optional
        If True, use Plotly for interactive heatmap; otherwise, use Seaborn/Matplotlib (default is False).

    Returns:
    --------
    None
        Displays the correlation heatmap.

    Raises:
    -------
    ValueError
        If df has no numeric columns.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in DataFrame.")

    corr = df[numeric_cols].corr()

    if interactive:
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        fig.show()
        fig.write_html(f"../reports/figures/correlationheatmap.html")
    else:
        plt.figure(figsize=(12,8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig("../reports/figures/correlationheatmap.png")
        plt.show()


def plot_count(df: pd.DataFrame, column: str, interactive: bool = False):
    """
    Plot a count plot or bar chart for a categorical column.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the categorical column.
    column : str
        Name of the categorical column to visualize.
    interactive : bool, optional
        If True, use Plotly for interactive plot; otherwise, use Seaborn/Matplotlib (default is False).

    Returns:
    --------
    None
        Displays the count/bar plot.

    Raises:
    -------
    ValueError
        If the specified column does not exist in df.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if interactive:
        fig = px.bar(df[column].value_counts().reset_index(),
                     x='index', y=column,
                     title=f"Count Plot of {column}")
        fig.show()
        fig.write_html(f"../reports/figures/{column}_count.html")
    else:
        plt.figure(figsize=(8,5))
        sns.countplot(data=df, x=column)
        plt.title(f"Count Plot of {column}")
        plt.savefig(f"../reports/figures/{column}_count.png")
        plt.show()


def plot_pairplot(df: pd.DataFrame, columns: list = None):
    """
    Plot a pairplot for multiple numeric columns to visualize pairwise relationships.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    columns : list, optional
        List of numeric columns to include in the pairplot. If None, all numeric columns are used.

    Returns:
    --------
    None
        Displays the pairplot.

    Raises:
    -------
    ValueError
        If none of the specified columns exist in df or if no numeric columns are available.
    """
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    else:
        # ensure columns exist
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

    if len(columns) == 0:
        raise ValueError("No numeric columns available for pairplot.")

    sns.pairplot(df[columns])
    plt.suptitle("Pairplot of Selected Features", y=1.02)
    plt.savefig("../reports/figures/pairplot.png")
    plt.show()

    def plot_pairplot_interactive(df: pd.DataFrame, columns: list = None):
        """
        Create an interactive pair plot (scatter matrix) for multiple numeric columns using Plotly.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data to plot.
        columns : list, optional
            List of numeric columns to include. If None, all numeric columns are used.

        Returns:
        --------
        None
            Displays the interactive scatter matrix.

        Raises:
        -------
        ValueError
            If none of the specified columns exist in df or if no numeric columns are available.
        """
        if columns is None:
            columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        else:
            missing = [col for col in columns if col not in df.columns]
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")

        if len(columns) == 0:
            raise ValueError("No numeric columns available for pairplot.")

        fig = px.scatter_matrix(df[columns],
                                dimensions=columns,
                                title="Interactive Pair Plot",
                                height=800)
        fig.update_traces(diagonal_visible=False)  # optional: hide diagonal histograms
        fig.show()
        fig.write_html(f"../reports/figures/pairplot_interactive.html")
