import pandas as pd


def csv_analyzer(path_to_csv: str, single_column: str = "", separator: str = ",") -> None:
    """Analyze a CSV file and print statistical information.

    Prints the mean of single_column when provided, otherwise the full describe() summary.
    """
    try:
        data = pd.read_csv(path_to_csv, sep=separator)
        if single_column:
            print(f"Mean of {single_column} is: {data[single_column].mean()}")
        else:
            print(data.describe())
    except (FileNotFoundError, pd.errors.ParserError) as ex:
        print(f"csv_analyzer error: {ex}")
