import pandas as pd

# Load data 
def load_data():

    file_results = "data/National-and-EP-Election-Results.xlsx"
    file_polls = "data/Election-Pollster-Results.xlsx"

    # Read from main election file
    df_oevk = pd.read_excel(
        file_results,
        sheet_name="Szavazókör_Results_(2014-18-22)"
    )

    df_national = pd.read_excel(
        file_results,
        sheet_name="National_Results_(2014-18-22)"
    )

    df_ep = pd.read_excel(
        file_results,
        sheet_name="EP_Results_(2019-24)"
    )

    # Read pollster file
    df_polls = pd.read_excel(
        file_polls,
        sheet_name="Adatok"
    )

    return df_oevk, df_national, df_ep, df_polls


# Process: Aggregate to OEVK from Szavazókör level for the Hungarian parliamentary elections
def aggregate_to_oevk(df):

    grouped = (
        df.groupby(
            [
                "Year",
                "Megye_No",
                "Megye",
                "OEVK",
                "Party",
                "Candidate"
            ],
            as_index=False
        )
        .agg(
            {
                "Voters": "sum",
                "Valid_Votes": "sum",
                "OEVK_Votes": "sum"
            }
        )
        .rename(
            columns={
                "Voters": "Sum_of_Voters",
                "Valid_Votes": "Sum_of_Valid_Votes"
            }
        )
    )

    grouped = grouped.sort_values(
        by=[
            "Year",
            "Megye_No",
            "OEVK",
            "OEVK_Votes"
        ],
        ascending=[True, True, True, False]
    ).reset_index(drop=True)

    return grouped