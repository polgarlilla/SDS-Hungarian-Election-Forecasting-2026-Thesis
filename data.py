### Import Packages

import pandas as pd



#### 1. Load data 

def load_data():
    file_results = "data/National-and-EP-Election-Results.xlsx"
    file_polls = "data/Polls_Vox_Populi.xlsx"
    file_2026_candidates = "data/2026-OEVK.csv"
    file_salary_vote = "data/salary_vote_share.xlsx"

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

    df_seats = pd.read_excel(
        file_results,
        sheet_name="Hist_Seat_Allocation"
    )

    # Read pollster file
    df_polls = pd.read_excel(
        file_polls,
        sheet_name="Clean"
    )

    # Read 2026 candidates file (CSV format)
    df_2026_candidates = pd.read_csv(
        file_2026_candidates
    )

    # Read salary vote share file
    df_salary_vote = pd.read_excel(
        file_salary_vote
    )

    return df_oevk, df_national, df_ep, df_seats, df_polls, df_2026_candidates, df_salary_vote



### 2. Aggregate to OEVK from Szavazókör level for the Hungarian parliamentary elections
def aggregate_to_oevk(df):
    # This function is used both for the parliamentary election data (which
    # contains a ``Candidate`` column and ``OEVK_Votes``) and the European
    # Parliament results after they have been transformed into a long format
    # (which contains a ``Votes`` column but no ``Candidate`` or
    # ``OEVK_Votes``).  In order to keep the behaviour of the original
    # implementation when ``df`` is the raw ``df_oevk`` and to allow the same
    # call to work on ``df_ep_trans``, we inspect the columns and build the
    # grouping/aggregation dynamically.

    # Base grouping columns that always exist
    grouping_cols = ["Year", "Megye_No", "Megye", "OEVK", "Party"]
    if "Candidate" in df.columns:
        grouping_cols.append("Candidate")

    # Build aggregation dict automatically for all numeric columns that are
    # *not* part of the grouping keys.  This keeps the behaviour consistent
    # with the prior hard-coded sums while also accommodating the different
    # column names present in the EP data.
    agg_dict = {}
    for col in df.columns:
        if col in grouping_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_dict[col] = "sum"

    result = df.groupby(grouping_cols, as_index=False).agg(agg_dict)

    # rename some of the columns the old function used to rename so that the
    # output for ``df_oevk`` remains identical.  We do not touch ``Votes`` or
    # other columns that the EP data introduces.
    rename_map = {}
    if "Voters" in result.columns:
        rename_map["Voters"] = "Sum_of_Voters"
    if "Valid_Votes" in result.columns:
        rename_map["Valid_Votes"] = "Sum_of_Valid_Votes"
    if rename_map:
        result = result.rename(columns=rename_map)

    # choose the appropriate column to sort on; the existing implementation
    # sorted by ``OEVK_Votes`` and descending order, whereas the EP data has
    # ``Votes`` instead.
    sort_on = "OEVK_Votes" if "OEVK_Votes" in result.columns else "Votes"
    result = result.sort_values(
        by=["Year", "Megye_No", "OEVK", sort_on],
        ascending=[True, True, True, False]
    ).reset_index(drop=True)

    return result



### 3. Aggregate EP votes by year and party
def aggregate_ep_votes_by_year(df_ep_trans_agg):
    """
    Aggregate EP-transformed data to a national list-level table by `Year` and
    `Party6`. The input `df_ep_trans_agg` may contain a `Party` (original
    party name) column; that is dropped so that `Party6` (the 6-category
    classification) is the unit of aggregation. The function returns a
    DataFrame with columns: `Year, Voters, Valid_Votes, Party6, Party_List_Votes`.
    """
    # Work on a copy to avoid mutating the caller's frame
    df = df_ep_trans_agg.copy()

    # If the original long-format `Party` column exists, drop it so that
    # aggregation happens purely on the classified `Party6` categories.
    if 'Party' in df.columns:
        df = df.drop(columns=['Party'])

    # Ensure the expected numeric columns exist; if not, create them with 0s
    for col in ['Votes', 'Sum_of_Voters', 'Sum_of_Valid_Votes']:
        if col not in df.columns:
            df[col] = 0

    # Group by Year + Party6 (national list-level aggregation)
    group_cols = ['Year', 'Party6']
    agg_dict = {
        'Votes': 'sum',
        'Sum_of_Voters': 'sum',
        'Sum_of_Valid_Votes': 'sum'
    }
    df_ep_list = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Rename to the desired output column names
    df_ep_list = df_ep_list.rename(columns={
        'Votes': 'Party_List_Votes',
        'Sum_of_Voters': 'Voters',
        'Sum_of_Valid_Votes': 'Valid_Votes'
    })

    # Reorder columns to the requested structure
    df_ep_list = df_ep_list[['Year', 'Voters', 'Valid_Votes', 'Party6', 'Party_List_Votes']]

    return df_ep_list



### 4. Delete rows where megye_no/Votes is null

def delete_null_megye_or_votes(df):
    """Delete rows where Megye_No or Votes is null"""
    return df.dropna(subset=['Megye_No', 'Votes'])



### 5. Classify parties into: Fidesz, Tisza, MiHazánk, DK, MKKP or Other in the election results data

def categorize_party_result(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `Party6` column to the frame based on the existing `Party` values.

    The new column has one of six categories: ``Fidesz``, ``Tisza``, ``MiHazánk``,
    ``DK``, ``MKKP`` or ``Other``.  ``Party`` entries are examined case-
    insensitively; if the string contains one of the keywords the corresponding
    category is assigned.  Anything that doesn't match falls into ``Other``.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a ``Party`` column.

    Returns
    -------
    pandas.DataFrame
        A copy of *df* with an added ``Party6`` column.
    """

    def _map(p: str) -> str:
        if not isinstance(p, str):
            return "Other"
        key = p.lower()
        if "fidesz" in key:
            return "Fidesz"
        if "tisza" in key:
            return "Tisza"
        if "kutya" in key or "mkkp" in key:
            return "MKKP"
        if "hazánk" in key:
            return "MiHazánk"
        if "demokratikus koalíció" in key or "dk" in key:
            return "DK"
        return "Other"

    result = df.copy()
    result["Party6"] = result["Party"].apply(_map)

    # Determine aggregation columns and vote column
    if "OEVK" in result.columns:
        group_cols = ["Year", "Megye", "OEVK", "Party6"]
        vote_col = "OEVK_Votes" if "OEVK_Votes" in result.columns else "Votes"
    else:
        group_cols = ["Year", "Party6"]
        vote_col = "Party_List_Votes" if "Party_List_Votes" in result.columns else "Votes"

    # Separate Other and non-Other rows
    other_rows = result[result["Party6"] == "Other"]
    non_other_rows = result[result["Party6"] != "Other"]

    # Aggregate Other rows
    if not other_rows.empty:
        agg_dict = {vote_col: "sum"}
        # Keep all other columns as is, except Party and Candidate (set to NA for Other)
        for col in result.columns:
            if col not in group_cols and col not in [vote_col, "Party", "Candidate"]:
                agg_dict[col] = "first"
        # Aggregate
        other_agg = other_rows.groupby(group_cols, as_index=False).agg(agg_dict)
        other_agg["Party"] = pd.NA
        if "Candidate" in other_agg.columns:
            other_agg["Candidate"] = pd.NA
        # Only keep one row per group for Other
        other_agg = other_agg.drop_duplicates(subset=group_cols)
    else:
        other_agg = pd.DataFrame(columns=result.columns)

    # Combine aggregated Other with non-Other rows
    combined = pd.concat([non_other_rows, other_agg], ignore_index=True)

    # If OEVK present, ensure only one row per Party6 per group
    if "OEVK" in result.columns:
        combined = combined.groupby(group_cols, as_index=False).agg({
            vote_col: "sum",
            **{col: "first" for col in combined.columns if col not in group_cols and col != vote_col}
        })

    else:
        combined = combined.groupby(group_cols, as_index=False).agg({
            vote_col: "sum",
            **{col: "first" for col in combined.columns if col not in group_cols and col != vote_col}
        })

    # For aggregated Other, Party and Candidate should remain NA
    if "Party" in combined.columns:
        combined.loc[combined["Party6"] == "Other", "Party"] = pd.NA
    if "Candidate" in combined.columns:
        combined.loc[combined["Party6"] == "Other", "Candidate"] = pd.NA

    return combined



### 6. Classify parties into: Fidesz, Tisza, MiHazánk, DK, MKKP or Other in the pollster data

def categorize_party_polls(df_polls: pd.DataFrame) -> pd.DataFrame:
    """
    Re-classifies party columns in df_polls according to:
    - Fidesz column stays as Fidesz
    - MKKP column stays as MKKP
    - Any column with 'DK' or 'EM' in its name becomes DK
    - MH column becomes MiHazánk
    - TISZA column becomes Tisza
    Returns a new DataFrame with renamed columns.
    """
    preserved_cols = list(df_polls.columns[:8])
    preserved_df = df_polls[preserved_cols].copy()

    party_cols = list(df_polls.columns[8:])
    party_df = df_polls[party_cols].copy().fillna(0)

    # Rename FIDESZ to Fidesz if present
    if 'FIDESZ' in party_df.columns:
        party_df = party_df.rename(columns={'FIDESZ': 'Fidesz'})

    # DK aggregation: DK, EM, DK-MSZP-P
    dk_cols = [col for col in party_df.columns if 'DK' in col or 'EM' in col or 'DK-MSZP-P' in col]
    dk = party_df[dk_cols].sum(axis=1) if dk_cols else pd.Series([0]*len(party_df), index=party_df.index)

    # Fidesz (after renaming)
    fidesz = party_df['Fidesz'] if 'Fidesz' in party_df.columns else pd.Series([0]*len(party_df), index=party_df.index)

    # MKKP
    mkkp = party_df['MKKP'] if 'MKKP' in party_df.columns else pd.Series([0]*len(party_df), index=party_df.index)

    # MiHazánk: MH column renamed
    mihazank = party_df['MH'] if 'MH' in party_df.columns else pd.Series([0]*len(party_df), index=party_df.index)

    # Tisza: TISZA column renamed
    tisza = party_df['TISZA'] if 'TISZA' in party_df.columns else pd.Series([0]*len(party_df), index=party_df.index)

    # Other: aggregate MSZP, Jobbik, LMP, Együtt, P, MM, MSZP-P, MMN, NP, Egyéb párt, Egyéb válasz
    other_cols = [col for col in party_df.columns if col in ['MSZP', 'Jobbik', 'LMP', 'Együtt', 'P', 'MM', 'MSZP-P', 'MMN', 'NP', 'Egyéb párt', 'Egyéb válasz']]
    other = party_df[other_cols].sum(axis=1) if other_cols else pd.Series([0]*len(party_df), index=party_df.index)

    party_agg_df = pd.DataFrame({
        'Fidesz': fidesz,
        'MKKP': mkkp,
        'DK': dk,
        'MiHazánk': mihazank,
        'Tisza': tisza,
        'Other': other
    }, index=party_df.index)

    df_final = pd.concat([preserved_df, party_agg_df], axis=1)
    return df_final



### 7. Transform the wide format (parties as columns) to long format (Party and Votes columns)

def transform_wide_to_long(df: pd.DataFrame, id_vars: list = None) -> pd.DataFrame:
    """Transform wide format (parties as columns) to long format (Party and Votes columns).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe in wide format where each party is a column.
    id_vars : list, optional
        Column names to keep as identifiers (e.g., ['Year', 'Megye', 'Megye_No', 'OEVK', 'Voters', 'Valid_Votes']).
        If None, will auto-detect non-numeric columns as identifiers.

    Returns
    -------
    pandas.DataFrame
        Long format dataframe with 'Party' and 'Votes' columns, plus all id_vars,
        sorted by Year, Megye_No, OEVK ascending and Votes descending, with index
        reset to 0.
    """
    if id_vars is None:
        # Default for EP data
        id_vars = ['Year', 'Megye', 'Megye_No', 'OEVK', 'Voters', 'Valid_Votes']
    
    # Get all other columns as party columns
    party_cols = [col for col in df.columns if col not in id_vars]
    
    # Use melt to transform from wide to long
    result = df.melt(
        id_vars=id_vars,
        value_vars=party_cols,
        var_name='Party',
        value_name='Votes'
    )

    
    # Sort by Year, Megye_No, OEVK ascending and Votes descending
    result = result.sort_values(
        by=['Year', 'Megye_No', 'OEVK', 'Votes'],
        ascending=[True, True, True, False]
    )
    
    # Reset index to start from 0
    result = result.reset_index(drop=True)
    
    return result



#### 8. Clean and prepare 2026 candidates data

def clean_candidates(df_2026_candidates):
    """
    Clean and prepare 2026 candidates dataframe.
    
    Performs the following operations:
    1. Filters candidates by citizenship (Hungarian), registration status, and OEVK assignment
    2. Creates Megye column (first word before space, uppercase)
    3. Creates OEVK_No column (numeric district number)
    4. Replaces specific county name variants (e.g., CSONGRÁD-CSANÁD → CSONGRÁD)
    
    Parameters:
    -----------
    df_2026_candidates : pd.DataFrame
        Raw 2026 candidates dataframe from CSV file
    
    Returns:
    --------
    pd.DataFrame
        Cleaned candidates dataframe with Megye and OEVK_No columns
    """
    
    # Filter df_2026_candidates
    df_2026_candidates_filtered = df_2026_candidates[
        (df_2026_candidates['Nemzetiség'].isna() | (df_2026_candidates['Nemzetiség'] == '')) &
        (df_2026_candidates['Státusz'] == 'Nyilvántartásba véve') &
        (df_2026_candidates['OEVK'].notna() & (df_2026_candidates['OEVK'] != ''))
    ].copy()

    # Create Megye and OEVK_No columns from OEVK
    # Megye: first word before space, converted to uppercase, as string type
    df_2026_candidates_filtered['Megye'] = df_2026_candidates_filtered['OEVK'].str.split().str[0].str.upper().astype(str)

    # OEVK_No: extract number, remove leading zeros
    df_2026_candidates_filtered['OEVK_No'] = df_2026_candidates_filtered['OEVK'].str.extract(r'(\d+)').astype(int)

    # Replace specific Megye values
    df_2026_candidates_filtered['Megye'] = df_2026_candidates_filtered['Megye'].replace('CSONGRÁD-CSANÁD', 'CSONGRÁD')
    
    return df_2026_candidates_filtered


#### 9. Create incumbent dummy for 2026 elections

def create_incumbent_dummy(df_oevk, df_2026_candidates_filtered):
    """
    Create incumbent dummy variable for 2026 elections.
    
    Identifies 2022 SMD winners (highest vote getter in each OEVK) and checks if they are 
    running again in 2026. Sets incumbent_dummy = 1 if running, 0 if not.
    
    Parameters:
    -----------
    df_oevk : pd.DataFrame
        District-level election results with Candidate column (from 2014-2022 elections)
    df_2026_candidates_filtered : pd.DataFrame
        Cleaned 2026 candidates dataframe with Megye and OEVK_No columns
    
    Returns:
    --------
    pd.DataFrame
        District-level incumbent dummy (Megye, OEVK_No, incumbent_dummy)
    """
    
    # Get 2022 SMD winners: highest vote getter in each OEVK
    df_2022 = df_oevk[df_oevk['Year'] == 2022].copy()
    
    # Get the candidate with highest votes in each OEVK district
    df_2022_winners = df_2022.loc[df_2022.groupby(['Megye', 'OEVK'])['OEVK_Votes'].idxmax()]
    df_2022_winners = df_2022_winners[['Megye', 'OEVK', 'Candidate']].drop_duplicates()
    df_2022_winners.rename(columns={'Candidate': 'Winner_2022'}, inplace=True)
    
    # Normalize candidate names (strip whitespace)
    df_2022_winners['Winner_2022'] = df_2022_winners['Winner_2022'].str.strip()
    
    # Get 2026 candidates and extract Megye and OEVK_No
    df_2026_cand = df_2026_candidates_filtered[['Megye', 'OEVK_No', 'Jelölt neve']].copy()
    df_2026_cand['Jelölt neve'] = df_2026_cand['Jelölt neve'].str.strip()
    
    # Create incumbent dummy by district
    incumbent_list = []
    
    for _, row_2022 in df_2022_winners.iterrows():
        megye = row_2022['Megye']
        oevk = str(row_2022['OEVK'])  # Convert to string in case it's an integer
        winner_name = row_2022['Winner_2022']
        
        # Extract OEVK number from OEVK string (e.g., "BUDAPEST 1" -> 1)
        oevk_no = int(oevk.split()[-1])
        
        # Check if this winner is running in 2026
        is_running = (df_2026_cand['Megye'] == megye) & (df_2026_cand['OEVK_No'] == oevk_no) & (df_2026_cand['Jelölt neve'] == winner_name)
        incumbent_dummy = 1 if is_running.any() else 0
        
        incumbent_list.append({
            'Megye': megye,
            'OEVK_No': oevk_no,
            'Winner_2022': winner_name,
            'incumbent_dummy': incumbent_dummy
        })
    
    incumbent_df = pd.DataFrame(incumbent_list)
    
    # Exclude BUDAPEST 17-18 (do not exist in 2026)
    incumbent_df = incumbent_df[~(
        (incumbent_df['Megye'] == 'BUDAPEST') & (incumbent_df['OEVK_No'].isin([17, 18]))
    )]
    
    # Add PEST 13-14 with incumbent_dummy=0 if they don't exist
    pest_new_oevks = [13, 14]
    for oevk_no in pest_new_oevks:
        if not ((incumbent_df['Megye'] == 'PEST') & (incumbent_df['OEVK_No'] == oevk_no)).any():
            incumbent_df = pd.concat([
                incumbent_df,
                pd.DataFrame({'Megye': ['PEST'], 'OEVK_No': [oevk_no], 'Winner_2022': [None], 'incumbent_dummy': [0]})
            ], ignore_index=True)
    
    incumbent_df = incumbent_df.sort_values(['Megye', 'OEVK_No']).reset_index(drop=True)
    
    return incumbent_df[['Megye', 'OEVK_No', 'incumbent_dummy']]