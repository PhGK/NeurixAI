"""File with functions to evaluate the results of the model and test sanity of the data."""

from typing import Any, Dict, List, Union


def check_shared_cells(fold_results: Dict[str, Dict[str, Any]]) -> Union[List[str], str]:
    """
    Check if Cell Lines are shared across folds.

    Args:
        fold_results (Dict[str, Dict[str, Any]]): A dictionary containing fold results,
            where keys are dataframe names and values are dictionaries containing 'cells' key
            with a set of cell lines.

    Returns
    -------
        Union[List[str], str]: A list of error messages indicating shared cells and in which dataframes they occur,
            or a success message if no shared cells are found.

    Raises
    ------
        AssertionError: If shared cells are found between any pair of dataframes.

    Example:
        >>> fold_results = {
        ...     'df1': {'cells': {'A', 'B', 'C'}},
        ...     'df2': {'cells': {'B', 'C', 'D'}},
        ...     'df3': {'cells': {'D', 'E', 'F'}}
        ... }
        >>> check_shared_cells(fold_results)
        ['Cells {'B', 'C'} are present in both dataframe df1 and dataframe df2']
    """
    error_messages = []

    for i, df in enumerate(fold_results):
        cells = fold_results[df]["cells"]
        # Check if each cell is exclusively in one dataframe
        for j, other_df in enumerate(fold_results):
            if i != j:
                other_cells = fold_results[other_df]["cells"]
                common_cells = set(cells) & set(other_cells)
                if common_cells:
                    error_message = f"Cells {common_cells} are present in both dataframe {df} and dataframe {other_df}"
                    error_messages.append(error_message)

    if error_messages:
        raise AssertionError("\n".join(error_messages))

    return "No shared cells found across folds."
