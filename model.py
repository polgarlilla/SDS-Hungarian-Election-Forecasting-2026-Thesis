
# Import Packages

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from scipy.stats import dirichlet




### 1. Pollster bias calculation

def pollster_bias(df_polls, df_list, pollster_col='Adatgazda', party_cols=None, year_col='Választás', result_year_col='Year', result_party_col='Party6', result_vote_col='Vote_Perc'):
    """
    Computes bias for each pollster and party using only pre-2026 polls.
    - df_polls: DataFrame with pollster estimates (wide format, party columns)
    - df_list: DataFrame with actual results (long format, Party, Year, Vote_Perc columns)
    - pollster_col: column in df_polls for pollster name
    - party_cols: list of party columns in df_polls (if None, auto-detect)
    - year_col: column in df_polls for year
    - result_year_col: column in df_list for year
    - result_party_col: column in df_list for party name
    - result_vote_col: column in df_list for vote share
    Returns: DataFrame with columns [Pollster, Party, Bias]
    """
    # Do not filter here; caller should provide the pre-election subset if needed
    df_polls = df_polls.copy()
    df_list = df_list.copy()
    
    if party_cols is None:
        party_cols = [col for col in df_polls.columns if col in ['Other', 'Fidesz', 'DK', 'MKKP', 'MiHazánk', 'Tisza']]

    # Use Party6 column for matching party names in df_list
    pollsters = df_polls[pollster_col].unique()
    bias_matrix = pd.DataFrame(index=pollsters, columns=party_cols)
    for pollster in pollsters:
        df_pollster = df_polls[df_polls[pollster_col] == pollster]
        for party in party_cols:
            biases = []
            for _, poll_row in df_pollster.iterrows():
                poll_year = poll_row[year_col]
                poll_value = poll_row[party]
                result_rows = df_list[(df_list[result_year_col] == poll_year) & (df_list[result_party_col] == party)]
                if not result_rows.empty:
                    result_value = result_rows[result_vote_col].values[0]
                    bias = poll_value - result_value
                    biases.append(bias)
            if biases:
                avg_bias = sum(biases) / len(biases)
            else:
                avg_bias = pd.NA
            bias_matrix.loc[pollster, party] = avg_bias
    return bias_matrix



### 2. Pollster quality (RMSE) calculation

def pollster_quality(df_polls, df_list, pollster_col='Adatgazda', party_col='Party6', year_col='Választás', result_year_col='Year', result_party_col='Party6', result_vote_col='Vote_Perc'):
    """
    Calculate RMSE (Quality) for each pollster across all elections and parties using only pre-2026 data.
    
    Formula: RMSE(j) = sqrt( (1 / (N_j * K)) * sum over elections e and parties k of (poll(j,k,e) - result(k,e))^2 )
    
    Where:
        - N_j = number of ELECTIONS (not polls) where pollster j conducted polls
        - K = number of parties
        - For multiple polls in same election, averages them first
    
    Args:
        df_polls: DataFrame with poll results (wide format, party columns)
        df_list: DataFrame with actual election results (long format, Party6 column)
        pollster_col: Column name for pollster in df_polls
        party_col: Column name for party in df_polls (should match party columns)
        year_col: Column name for year in df_polls
        result_year_col: Column name for year in df_list
        result_party_col: Column name for party in df_list (Party6)
        result_vote_col: Column name for actual vote percentage in df_list
    Returns:
         DataFrame with pollster names as index and a single column 'Quality' (RMSE)
    """
    import numpy as np
    
    # Do not hardcode a year filter here; the caller should pass the appropriate
    # pre-election subset (e.g. polls before the target election year).
    df_polls = df_polls.copy()
    df_list = df_list.copy()
    
    pollsters = df_polls[pollster_col].unique()
    parties = df_list[result_party_col].unique()
    K = len(parties)
    quality_dict = {}
    
    for pollster in pollsters:
        # Get all polls by this pollster
        df_polls_j = df_polls[df_polls[pollster_col] == pollster]
        
        # Average polls by election (in case multiple polls per election)
        # For each election and party, take the mean poll value
        df_polls_avg = df_polls_j.groupby(year_col)[parties.tolist()].mean()
        
        # Number of elections
        Nj = df_polls_avg.shape[0]
        
        squared_errors = []
        # For each election
        for election_year in df_polls_avg.index:
            # For each party
            for party in parties:
                poll_value = df_polls_avg.loc[election_year, party]
                # Get actual result
                result_rows = df_list[(df_list[result_year_col] == election_year) & (df_list[result_party_col] == party)]
                if not result_rows.empty:
                    result_value = result_rows[result_vote_col].values[0]
                    if pd.notna(poll_value) and pd.notna(result_value):
                        squared_errors.append((poll_value - result_value) ** 2)
        
        if Nj > 0 and K > 0 and len(squared_errors) > 0:
            rmse = np.sqrt(np.sum(squared_errors) / (Nj * K))
        else:
            rmse = np.nan
        quality_dict[pollster] = rmse
    
    df_quality = pd.DataFrame.from_dict(quality_dict, orient='index', columns=['Quality'])
    return df_quality



### 3. Industry-wide polling error (sigma_poll)

def pollster_sigma(
    df_polls,
    df_list,
    pollster_col='Adatgazda',
    party_col='Party6',
    year_col='Választás',
    result_year_col='Year',
    result_party_col='Party6',
    result_vote_col='Vote_Perc'
):
    """
    Calculate industry-wide polling error (sigma_poll) pooling all pollsters, elections, and parties.
    Args:
        df_polls: DataFrame with poll results (wide format, party columns)
        df_list: DataFrame with actual election results (long format, Party6 column)
        pollster_col: Column name for pollster in df_polls
        party_col: Column name for party in df_polls (should match party columns)
        year_col: Column name for year in df_polls
        result_year_col: Column name for year in df_list
        result_party_col: Column name for party in df_list (Party6)
        result_vote_col: Column name for actual vote percentage in df_list
    Prints:
        sigma_poll: industry-wide average polling error
    """
    import numpy as np
    import pandas as pd

    # Pollsters and parties (from results frame)
    pollsters = df_polls[pollster_col].unique()
    parties = list(df_list[result_party_col].unique())
    K = len(parties)

    # We'll compute per-pollster-per-election average poll values (like pollster_quality)
    # and accumulate squared errors across elections. Denominator = sum_j Nj * K
    total_squared_errors = 0.0
    total_Nj = 0

    for pollster in pollsters:
        df_polls_j = df_polls[df_polls[pollster_col] == pollster]
        # Unique elections this pollster covered
        years = df_polls_j[year_col].dropna().unique()
        Nj = len(years)
        total_Nj += Nj

        for election_year in years:
            df_year = df_polls_j[df_polls_j[year_col] == election_year]
            if df_year.empty:
                continue

            # Average polls by party for this pollster and election (reindex to ensure same party order)
            avg_poll = df_year.reindex(columns=parties).mean()

            # Compare to actual results for that election across parties
            for party in parties:
                poll_value = avg_poll.get(party, np.nan)
                result_rows = df_list[(df_list[result_year_col] == election_year) & (df_list[result_party_col] == party)]
                if not result_rows.empty:
                    result_value = result_rows[result_vote_col].values[0]
                    if pd.notna(poll_value) and pd.notna(result_value):
                        total_squared_errors += (poll_value - result_value) ** 2

    denominator = total_Nj * K
    if denominator > 0:
        sigma_poll = np.sqrt(total_squared_errors / denominator)
    else:
        sigma_poll = np.nan

    sigma = pd.DataFrame({'sigma_poll': [sigma_poll]})
    return sigma



### 4. Compute a bias-corrected, quality-weighted polling average

def polling_avg(df_polls, df_bias, df_quality, election_date="2026-04-12", lambda_decay=0.03, party_list=None):
    """
    Compute bias-corrected, quality-weighted polling average for each party.
    Args:
        df_polls: DataFrame of polls (must include 'Adatgazda', 'Vég', party columns)
        df_bias: DataFrame of pollster bias (index: pollster, columns: party)
        df_quality: DataFrame of pollster RMSE (index: pollster, column: 'Quality')
        election_date: str, date of election (YYYY-MM-DD)
        lambda_decay: float, time decay parameter
        party_list: list of party names to aggregate (default: ['Fidesz', 'Tisza', 'MiHazánk', 'DK', 'MKKP', 'Other'])
    Returns:
        DataFrame with weighted polling average for each party
    """
    import pandas as pd
    from datetime import datetime
    if party_list is None:
        party_list = ['Fidesz', 'Tisza', 'MiHazánk', 'DK', 'MKKP', 'Other']
    # Robustly handle election_date as str, pd.Timestamp, or datetime
    if isinstance(election_date, str):
        try:
            election_dt = datetime.strptime(election_date, "%Y-%m-%d")
        except ValueError:
            election_dt = datetime.strptime(election_date, "%Y-%m-%d %H:%M:%S")
    elif hasattr(election_date, 'to_pydatetime'):
        election_dt = election_date.to_pydatetime()
    elif isinstance(election_date, datetime):
        election_dt = election_date
    else:
        raise ValueError("election_date must be a string, pd.Timestamp, or datetime object")

    # Use polls for the specified election year (derived from election_date).
    # If no polls explicitly labeled for that election year exist, fall back
    # to the most recent polls before the election year.
    election_year = election_dt.year
    df_polls_election = df_polls[df_polls['Választás'] == election_year]
    if df_polls_election.empty:
        df_polls_election = df_polls[df_polls['Választás'] < election_year]
    if df_polls_election.empty:
        # Last resort: use all polls
        df_polls_election = df_polls.copy()

    results = {}
    for party in party_list:
        weighted_sum = 0
        weight_total = 0
        for _, row in df_polls_election.iterrows():
            pollster = row['Adatgazda']
            poll_date = pd.to_datetime(row['Vég'])
            days_until_election = (election_dt - poll_date).days
            poll_value = row.get(party, np.nan)
            bias = df_bias.loc[pollster, party] if pollster in df_bias.index and party in df_bias.columns else 0
            rmse = df_quality.loc[pollster, 'Quality'] if pollster in df_quality.index else np.nan
            # Only use bias if it is not NA, otherwise skip bias correction
            if pd.notna(poll_value) and pd.notna(rmse) and rmse > 0:
                if pd.notna(bias):
                    corrected = poll_value - bias
                else:
                    corrected = poll_value
                weight = np.exp(-lambda_decay * days_until_election) * (1 / rmse**2)
                weighted_sum += weight * corrected
                weight_total += weight
        avg = weighted_sum / weight_total if weight_total > 0 else np.nan
        results[party] = avg
    df_avg = pd.DataFrame([results])
    return df_avg



### 5. Construct the forecast distribution

def forecast_distr(df_avg, sigma_poll, df_polls, election_date="2026-04-12", alpha=0.1, party_list=None):
    """
    Calculate forecast distribution parameters for each party.
    Args:
        df_avg: DataFrame with polling averages (columns: party names)
        sigma_poll: float, industry-wide polling error (RMSE)
        df_polls: DataFrame of polls (must include 'Vég', party columns)
        election_date: str, date of election (YYYY-MM-DD)
        alpha: float, staleness adjustment parameter
        party_list: list of party names to aggregate (default: ['Fidesz', 'Tisza', 'MiHazánk', 'DK', 'MKKP', 'Other'])
    Returns:
        DataFrame with columns ['Party', 'Mu', 'Sigma']
    """
    import pandas as pd
    from datetime import datetime
    if party_list is None:
        party_list = ['Fidesz', 'Tisza', 'MiHazánk', 'DK', 'MKKP', 'Other']
    # Robustly handle election_date
    if isinstance(election_date, str):
        try:
            election_dt = datetime.strptime(election_date, "%Y-%m-%d")
        except ValueError:
            election_dt = datetime.strptime(election_date, "%Y-%m-%d %H:%M:%S")
    elif hasattr(election_date, 'to_pydatetime'):
        election_dt = election_date.to_pydatetime()
    elif isinstance(election_date, datetime):
        election_dt = election_date
    else:
        raise ValueError("election_date must be a string, pd.Timestamp, or datetime object")
    import numpy as np
    distr = []
    # Use polls for the specified election year (derived from election_date)
    election_year = election_dt.year
    df_polls_election = df_polls[df_polls['Választás'] == election_year]
    if not df_polls_election.empty:
        poll_dates = pd.to_datetime(df_polls_election['Vég'])
        last_poll_date = poll_dates.max()
        months_since_last_poll = (election_dt - last_poll_date).days / 30
    else:
        months_since_last_poll = np.nan
    # Allow sigma_poll to be either a scalar or a single-value DataFrame
    try:
        sigma_base = float(sigma_poll)
    except Exception:
        try:
            sigma_base = float(sigma_poll.values[0])
        except Exception:
            sigma_base = np.nan
    sigma = sigma_base * (1 + alpha * months_since_last_poll) if pd.notna(months_since_last_poll) else sigma_base
    for party in party_list:
        mu = df_avg[party].values[0] if party in df_avg.columns else np.nan
        distr.append({'Party': party, 'Mu': mu, 'Sigma': sigma})
    df_distr = pd.DataFrame(distr)
    return df_distr



### 6. Handle correlations between parties

def correl_parties(df_distr, n_draws=1):
    """
    Draw full vector of party vote shares from a Dirichlet distribution,
    parameterized to match the means μ(k) and variances σ(k)² in df_distr.
    
    Two-step approach:
    1. Fix α_k = μ_k · α₀ to ensure means are EXACTLY correct
    2. Optimize α₀ (single parameter) to match the overall uncertainty level
    
    Dirichlet variance: Var[X_k] = μ_k(1 - μ_k) / (α₀ + 1)
    
    Returns an array of shape (n_draws, n_parties) with shares summing to 100%.
    """
    # Convert means to proportions and handle NaNs
    mu_raw = np.array(df_distr['Mu'].values, dtype=float) / 100.0
    mu = np.nan_to_num(mu_raw, nan=0.0)
    # If all zeros (no polling information), fall back to uniform small positive proportions
    if mu.sum() <= 0:
        mu = np.ones_like(mu) / len(mu)
    else:
        mu = mu / mu.sum()

    # Convert variances to proportions and handle NaNs
    sigma2_raw = np.array(df_distr['Sigma'].values, dtype=float) ** 2 / (100.0 ** 2)
    sigma2 = np.nan_to_num(sigma2_raw, nan=np.nan)
    n_parties = len(mu)

    # Step 1: Fix α_k = μ_k · α₀ (ensures correct means)
    # This means we only need to optimize α₀
    
    # Step 2: Optimize α₀ to match the variance level
    # Target: average variance across all parties
    # Use nanmean to ignore missing variance entries; if NaN, set a small default
    try:
        target_var = np.nanmean(sigma2)
        if np.isnan(target_var):
            target_var = 0.0001
    except Exception:
        target_var = 0.0001
    
    def loss(alpha0):
        """Loss function for fitting α₀."""
        alpha0 = np.abs(alpha0[0])  # Ensure positive
        if alpha0 < 0.1:  # Avoid very small values
            return 1e6
        # Dirichlet variance formula: Var[X_k] = μ_k(1 - μ_k) / (α₀ + 1)
        predicted_var = np.mean(mu * (1 - mu) / (alpha0 + 1))
        return (predicted_var - target_var) ** 2
    
    # Initial guess: start with α₀ = 10 (moderate concentration)
    alpha0_init = [10.0]
    res = minimize(loss, alpha0_init, method='Nelder-Mead')
    alpha0_fit = max(0.1, res.x[0])  # Ensure minimum positive value
    
    # Compute final concentration parameters
    # Ensure alpha parameters are strictly positive
    eps = 1e-6
    alpha_fit = np.maximum(mu * alpha0_fit, eps)
    
    # Draw from Dirichlet
    draws = dirichlet.rvs(alpha_fit, size=n_draws)
    # Convert to percentages
    draws_pct = draws * 100
    return draws_pct



### 7. District swing coefficient calculation

def swing_coef(df_ep_trans_agg):
    """
    Computes district-level swing coefficients based on 2024 EP election deviations from national average.
    
    For each district and party, calculates:
    Δ(d, k) = EP2024_district(d, k) − EP2024_national(k)
    
    This captures the stable geographic structure showing which districts are friendlier 
    or more hostile to each party relative to the national average.
    
    Args:
        df_ep_trans_agg: DataFrame with 2024 EP results (Year, Megye_No, Megye, OEVK, Party6, Votes)
    
    Returns:
        DataFrame with columns: Megye_No, Megye, OEVK, Fidesz, Tisza, MiHazánk, DK, MKKP, Other
    """
    # Filter for 2024 EP election only
    df_2024 = df_ep_trans_agg[df_ep_trans_agg['Year'] == 2024].copy()
    
    # Calculate national totals and national share for each party
    total_national_votes = df_2024['Votes'].sum()
    national_shares = df_2024.groupby('Party6')['Votes'].sum() / total_national_votes * 100
    
    # Calculate district totals
    df_2024['District_Total_Votes'] = df_2024.groupby(['Megye_No', 'OEVK'])['Votes'].transform('sum')
    
    # Calculate district share for each party
    df_2024['District_Share'] = df_2024['Votes'] / df_2024['District_Total_Votes'] * 100
    
    # Calculate deviation (district share - national share)
    df_2024['Delta'] = df_2024.apply(
        lambda row: row['District_Share'] - national_shares.get(row['Party6'], 0.0),
        axis=1
    )
    
    # Pivot to wide format: one row per district, columns for each party
    swing_df = df_2024.pivot_table(
        index=['Megye_No', 'Megye', 'OEVK'],
        columns='Party6',
        values='Delta',
        aggfunc='first'
    ).reset_index()
    
    # Ensure all 6 parties as columns
    expected_parties = ['Fidesz', 'Tisza', 'MiHazánk', 'DK', 'MKKP', 'Other']
    for party in expected_parties:
        if party not in swing_df.columns:
            swing_df[party] = 0.0
    
    # Select final columns in correct order
    cols_to_select = ['Megye_No', 'Megye', 'OEVK'] + expected_parties
    swing_df = swing_df[cols_to_select]
    
    # Sort by district
    swing_df = swing_df.sort_values(['Megye_No', 'OEVK']).reset_index(drop=True)
    return swing_df



### 8. Calibrate district noise (sigma_d)

def calibrate_sigma_d(df_list, df_oevk, df_ep_trans_agg):
    """
    Calibrate the district-level noise parameter σ_d by measuring within-election variance in 2022.
    
    Approach:
        1. Compute swing coefficients from 2024 EP election (baseline for geographic structure)
        2. Get 2022 actual national results as the forecast baseline
        3. Project 2022 district results using swing model with σ_d = 0 (no noise)
        4. For each district and party, compute residual = actual 2022 share - predicted 2022 share
        5. Take standard deviation of residuals to estimate true district-level noise
    
    This approach measures genuine within-election district variance around the swing model, 
    not between-election party-system change.
    
    Args:
        df_list: DataFrame with national election results (Year, Party6, Vote_Perc columns)
        df_oevk: DataFrame with district results (Year, Megye_No, OEVK, Party6, OEVK_Votes columns)
        df_ep_trans_agg: DataFrame with 2019 EP results (Year, Megye_No, OEVK, Party6, Votes columns)
    
    Returns:
        float: Estimated σ_d (standard deviation of within-election district residuals)
    """
    parties = ['Fidesz', 'Tisza', 'MiHazánk', 'DK', 'MKKP', 'Other']
    
    # Step 1: Compute swing coefficients from 2024 EP baseline
    swing_df = swing_coef(df_ep_trans_agg)
    
    # Step 2: Get 2022 actual national results
    df_2022_national = df_list[df_list['Year'] == 2022].copy()
    
    # Convert to percentages
    total_votes_2022 = df_2022_national['Party_List_Votes'].sum()
    national_2022 = {}
    for party in parties:
        party_data = df_2022_national[df_2022_national['Party6'] == party]
        if not party_data.empty:
            votes = party_data['Party_List_Votes'].values[0]
            national_2022[party] = (votes / total_votes_2022) * 100
        else:
            national_2022[party] = 0.0
    
    # Step 3: Project 2022 district results using swing model with no noise
    # This gives: projected = national_2022 + swing(d, k)
    district_projection = OEVK_projection(national_2022, swing_df, sigma_d=0.0, random_state=42)
    
    # Step 4: Get actual 2022 results
    df_2022_oevk = df_oevk[df_oevk['Year'] == 2022].copy()
    
    # Aggregate to percentages by district
    df_2022_oevk['Total_Votes'] = df_2022_oevk.groupby(['Megye_No', 'OEVK'])['OEVK_Votes'].transform('sum')
    df_2022_oevk['Vote_Share'] = df_2022_oevk['OEVK_Votes'] / df_2022_oevk['Total_Votes'] * 100
    
    # Step 5: Calculate within-election residuals (actual - predicted)
    residuals = []
    
    for idx, proj_row in district_projection.iterrows():
        megye_no = proj_row['Megye_No']
        oevk = proj_row['OEVK']
        
        # For each party, compare projection to actual
        for party in parties:
            projected_share = proj_row[party]
            
            # Get actual 2022 share for this district and party
            actual_rows = df_2022_oevk[
                (df_2022_oevk['Megye_No'] == megye_no) & 
                (df_2022_oevk['OEVK'] == oevk) & 
                (df_2022_oevk['Party6'] == party)
            ]
            
            if not actual_rows.empty:
                actual_share = actual_rows['Vote_Share'].values[0]
                residual = actual_share - projected_share
                residuals.append(residual)
    
    # Step 6: Estimate σ_d as standard deviation of within-election residuals
    if len(residuals) > 0:
        sigma_d_estimated = np.std(residuals)
    else:
        sigma_d_estimated = 1.0  # Default fallback
    
    return sigma_d_estimated




### 9. District-level projection function

def OEVK_projection(national_draw, swing_df, sigma_d=None, random_state=None):
    """
    Project district results for a given national (listás) vote share draw.
    Args:
        national_draw: dict or pd.Series, party vote shares (keys: party names, values: vote share in %)
        swing_df: DataFrame, district swing coefficients (columns: Megye_No, Megye, OEVK, parties...)
        sigma_d: float, district noise stddev (if None, defaults to 1.0)
        random_state: int or np.random.RandomState, for reproducibility
    Returns:
        DataFrame with columns: Megye_No, Megye, OEVK, parties..., Winner
    """
    parties = ['Fidesz', 'Tisza', 'MiHazánk', 'DK', 'MKKP', 'Other']
    if sigma_d is None:
        sigma_d = 1.0
    rng = np.random.default_rng(random_state)
    results = []
    for idx, row in swing_df.iterrows():
        district_votes = {}
        for party in parties:
            pi_k = national_draw[party] if party in national_draw else 0.0
            delta = row[party] if party in row else 0.0
            eps = rng.normal(0, sigma_d)
            vote = pi_k + delta + eps
            district_votes[party] = max(0, vote)
        # Renormalize so sum = 1 (or 100%)
        total = sum(district_votes.values())
        if total > 0:
            for party in parties:
                district_votes[party] = district_votes[party] / total * 100
        else:
            for party in parties:
                district_votes[party] = 0.0
        winner = max(district_votes, key=lambda p: district_votes[p])
        results.append({
            'Megye_No': row['Megye_No'],
            'Megye': row['Megye'],
            'OEVK': row['OEVK'],
            **{party: district_votes[party] for party in parties},
            'Winner': winner
        })
    return pd.DataFrame(results)



### 10. Simulation loop

def simulation(df_distr, swing_df, df_list=None, df_oevk=None, df_ep_trans_agg=None, sigma_d=1.0, n_sim=1000, year=2026, participation=None, random_state=None):
    """
    Run election simulation n_sim times using Dirichlet distribution for national vote shares.
    
    Process per simulation:
        1. Draw national vote shares from Dirichlet distribution (parameterized to match means μ(k) and variances σ(k)²)
        2. Compute district results: vote(d,k) = π(k) + Δ(d,k) + noise
        3. Determine 106 SMD winners
        4. Compute fragment votes and allocate 93 list seats via d'Hondt
        5. Record total seats per party
    
    Args:
        df_distr: DataFrame with national vote distributions (Party, Mu, Sigma columns)
        swing_df: DataFrame with district swing coefficients
        df_list: DataFrame with national election results (for valid votes)
        df_oevk: DataFrame with district results (for valid votes in 2014, 2018, 2022)
        df_ep_trans_agg: DataFrame with district results (for valid votes in 2019, 2024)
        sigma_d: float, district noise stddev
        n_sim: int, number of simulations
        year: int, election year (default 2026)
        participation: float or None, election participation rate (0.0 to 1.0)
                      If None, uses 2022 Sum_of_Valid_Votes from df_oevk
                      If set (e.g., 0.75), uses Sum_of_Voters * participation for each OEVK
        random_state: int or np.random.Generator
    Returns:
        tuple of (summary_df, oevk_means_df, oevk_vote_counts_df, national_summary_df):
        - summary_df: DataFrame with Party, Mean seats, 95% CI, P(majority), P(supermajority)
        - oevk_means_df: DataFrame with OEVK-level mean vote percentages (Megye_No, Megye, OEVK, 
                         Fidesz, Tisza, MiHazánk, DK, MKKP, Other, Winner)
        - oevk_vote_counts_df: DataFrame with OEVK-level mean vote counts (Megye_No, Megye, OEVK, 
                               Fidesz, Tisza, MiHazánk, DK, MKKP, Other, Winner)
        - national_summary_df: DataFrame with national (listás) vote share statistics (Party, Mean, Median, 95% CI)
    """
    from seat_allocation import seat_simulated
    parties = ['Fidesz', 'Tisza', 'MiHazánk', 'DK', 'MKKP', 'Other']
    rng = np.random.default_rng(random_state)
    seat_results = {party: [] for party in parties}
    
    # Store OEVK-level results from each simulation
    oevk_results_all = []
    
    # Store national vote shares from each simulation
    national_draws_all = []
    
    # Pre-generate all national vote shares from Dirichlet distribution
    # (parameterized to match means μ(k) and variances σ(k)²)
    dirichlet_draws = correl_parties(df_distr, n_draws=n_sim)  # Shape: (n_sim, n_parties)
    
    for s in range(n_sim):
        # 1. Get the pre-generated national vote shares
        national_draw = {}
        for i, party in enumerate(parties):
            national_draw[party] = dirichlet_draws[s, i]
        
        # Store national draw for later statistics
        national_draws_all.append(national_draw.copy())
        
        # 2. Compute district results: vote(d,k) = π(k) + Δ(d,k) + noise
        district_projection = OEVK_projection(national_draw, swing_df, sigma_d=sigma_d, random_state=rng)
        
        # Store OEVK-level results (district_projection contains vote counts or percentages per party per district)
        oevk_results_all.append(district_projection.copy())
        
        # 3, 4, 5. Determine SMD winners, allocate list seats, and record total seats
        parliament_seats = seat_simulated(district_projection, df_distr, df_list=df_list, df_oevk=df_oevk, df_ep_trans_agg=df_ep_trans_agg, year=year)
        for party in parties:
            seat_results[party].append(parliament_seats[party].values[0])
    
    # Aggregate results: compute summary statistics across simulations
    summary_data = {}
    for party in parties:
        seats = np.array(seat_results[party])
        mean_seats = np.mean(seats)
        ci_low = np.percentile(seats, 2.5)
        ci_high = np.percentile(seats, 97.5)
        p_majority = np.mean(seats >= 100)
        p_supermajority = np.mean(seats >= 133)
        summary_data[party] = {
            'mean': mean_seats,
            'ci_low': int(round(ci_low)),
            'ci_high': int(round(ci_high)),
            'p_majority': f"{int(round(p_majority*100))}%",
            'p_supermajority': f"{int(round(p_supermajority*100))}%"
        }
    
    # Use largest remainder method (Hamilton's method) to ensure total = 199 seats
    # 1. Start with floor of each party's mean
    allocated_seats = {party: int(np.floor(summary_data[party]['mean'])) for party in parties}
    total_allocated = sum(allocated_seats.values())
    
    # 2. Distribute remaining seats to parties with largest fractional parts
    remaining_seats = 199 - total_allocated
    if remaining_seats > 0:
        # Compute fractional parts for each party
        fractional_parts = {party: summary_data[party]['mean'] - int(np.floor(summary_data[party]['mean'])) 
                           for party in parties}
        # Sort by fractional part (descending) and allocate remaining seats
        for party in sorted(fractional_parts, key=lambda p: fractional_parts[p], reverse=True)[:remaining_seats]:
            allocated_seats[party] += 1
    
    # 3. Build summary with allocated seats that sum to 199
    summary = []
    for party in parties:
        summary.append({
            'Party': party,
            'Mean seats': allocated_seats[party],
            '95% CI': f"{summary_data[party]['ci_low']}–{summary_data[party]['ci_high']}",
            'P(majority)': summary_data[party]['p_majority'],
            'P(supermajority)': summary_data[party]['p_supermajority']
        })
    
    # Compute OEVK-level means across all simulations
    # Combine all OEVK results and compute mean vote percentages per party per OEVK
    if oevk_results_all:
        oevk_concat = pd.concat(oevk_results_all, ignore_index=False)
        
        # Group by Megye_No, Megye, OEVK and compute mean for each party
        if 'Megye_No' in oevk_concat.columns and 'OEVK' in oevk_concat.columns:
            groupby_cols = ['Megye_No', 'Megye', 'OEVK'] if 'Megye' in oevk_concat.columns else ['Megye_No', 'OEVK']
            oevk_means = oevk_concat.groupby(groupby_cols, as_index=False)[parties].mean()
            
            # Compute Winner column (party with largest mean vote percentage)
            oevk_means['Winner'] = oevk_means[parties].idxmax(axis=1)
            
            # Reorder columns
            if 'Megye' in oevk_means.columns:
                oevk_means = oevk_means[['Megye_No', 'Megye', 'OEVK'] + parties + ['Winner']]
            else:
                oevk_means = oevk_means[['Megye_No', 'OEVK'] + parties + ['Winner']]
        else:
            oevk_means = pd.DataFrame()
    else:
        oevk_means = pd.DataFrame()
    
    # Create vote count version: multiply percentages by valid votes for each OEVK
    oevk_vote_counts_df = oevk_means.copy() if not oevk_means.empty else pd.DataFrame()
    
    if not oevk_vote_counts_df.empty and df_oevk is not None and 'Sum_of_Valid_Votes' in df_oevk.columns:
        # Determine which year's data to use for valid votes
        if participation is None:
            # Use 2022 Sum_of_Valid_Votes by default
            df_oevk_target = df_oevk[df_oevk['Year'] == 2022][['Megye_No', 'Megye', 'OEVK', 'Sum_of_Valid_Votes']].drop_duplicates()
        else:
            # Use Sum_of_Voters * participation for 2022
            df_oevk_2022 = df_oevk[df_oevk['Year'] == 2022][['Megye_No', 'Megye', 'OEVK', 'Sum_of_Voters']].drop_duplicates()
            df_oevk_target = df_oevk_2022.copy()
            df_oevk_target['Sum_of_Valid_Votes'] = (df_oevk_target['Sum_of_Voters'] * participation).round(0).astype(int)
            df_oevk_target = df_oevk_target[['Megye_No', 'Megye', 'OEVK', 'Sum_of_Valid_Votes']]
        
        # Merge valid votes into oevk_vote_counts_df
        oevk_vote_counts_df = oevk_vote_counts_df.merge(
            df_oevk_target,
            on=['Megye_No', 'Megye', 'OEVK'] if 'Megye' in oevk_vote_counts_df.columns else ['Megye_No', 'OEVK'],
            how='left'
        )
        
        # Convert vote percentages to vote counts
        for party in parties:
            oevk_vote_counts_df[party] = (oevk_vote_counts_df[party] / 100.0) * oevk_vote_counts_df['Sum_of_Valid_Votes']
            oevk_vote_counts_df[party] = oevk_vote_counts_df[party].round(0).astype(int)
        
        # Drop the Sum_of_Valid_Votes column
        oevk_vote_counts_df = oevk_vote_counts_df.drop(columns=['Sum_of_Valid_Votes'])
    
    # Compute national vote share statistics
    national_summary = []
    for party in parties:
        votes = np.array([d[party] for d in national_draws_all])
        national_summary.append({
            'Party': party,
            'Mean_%': np.mean(votes),
            'Median_%': np.median(votes),
            'Std_%': np.std(votes),
            '95% CI Low_%': np.percentile(votes, 2.5),
            '95% CI High_%': np.percentile(votes, 97.5)
        })
    
    national_summary_df = pd.DataFrame(national_summary)
    
    return pd.DataFrame(summary), oevk_means, oevk_vote_counts_df, national_summary_df




### 11. Backtesting on 2022 election

def backtesting(df_polls, df_list, df_oevk, df_ep_trans_agg, df_seats=None, sigma_poll=None, sigma_d=1.0, n_sim=1000, random_state=None):
    """
    Backtest the entire pipeline on 2022 election.
    
    Process:
        1. Filter polls to pre-2022 only
        2. Estimate pollster bias from 2014 and 2018 elections only
        3. Compute pollster quality from pre-2022 data
        4. Generate polling average for 2022
        5. Build forecast distribution
        6. Use 2019 EP as district swing baseline
        7. Run full simulation
        8. Extract actual 2022 results
        9. Compare simulated distribution to actual results
        10. Check if actual outcome falls within 95% interval
    
    Args:
        df_polls: DataFrame with poll data (Adatgazda, Választás, Party columns)
        df_list: DataFrame with national results (Year, Party6, Vote_Perc columns)
        df_oevk: DataFrame with district results (Year, Megye_No, OEVK, Party6, OEVK_Votes columns)
        df_ep_trans_agg: DataFrame with 2019 EP results
        sigma_poll: float, industry-wide polling error (if None, computed from data)
        sigma_d: float, district noise parameter (default 1.0)
        n_sim: int, number of simulations
        random_state: int or np.random.Generator
    
    Returns:
        Tuple of (simulated_results_df, actual_results_df, comparison_df)
        - simulated_results_df: Summary statistics from simulation (mean, 95% CI, probabilities)
        - actual_results_df: Actual 2022 seat counts by party
        - comparison_df: Comparison table showing if actual falls within 95% CI
    """
    from seat_allocation import seat_simulated
    
    parties = ['Fidesz', 'Tisza', 'MiHazánk', 'DK', 'MKKP', 'Other']
    
    # Step 1: Filter polls to pre-2022 only
    df_polls_pre2022 = df_polls[df_polls['Választás'] < 2022].copy()
    
    # Step 2: Estimate pollster bias from 2014 and 2018 elections ONLY
    df_list_bias = df_list[df_list['Year'].isin([2014, 2018])].copy()
    df_bias = pollster_bias(
        df_polls_pre2022,
        df_list_bias,
        pollster_col='Adatgazda',
        party_cols=['Other', 'Fidesz', 'DK', 'MKKP', 'MiHazánk', 'Tisza'],
        year_col='Választás',
        result_year_col='Year',
        result_party_col='Party6',
        result_vote_col='Vote_Perc'
    )
    
    # Step 3: Compute pollster quality from pre-2022 data
    df_quality = pollster_quality(
        df_polls_pre2022,
        df_list,
        pollster_col='Adatgazda',
        party_col='Party6',
        year_col='Választás',
        result_year_col='Year',
        result_party_col='Party6',
        result_vote_col='Vote_Perc'
    )
    
    # Step 4: Compute σ_poll from historical polling errors using pollster_sigma()
    # Always compute using pre-2022 polls (do not rely on caller-supplied value)
    sigma_poll_df = pollster_sigma(
        df_polls_pre2022,
        df_list,
        pollster_col='Adatgazda',
        party_col='Party6',
        year_col='Választás',
        result_year_col='Year',
        result_party_col='Party6',
        result_vote_col='Vote_Perc'
    )
    sigma_poll_value = sigma_poll_df['sigma_poll'].values[0]
    
    # Step 5: Generate polling average for 2022
    df_avg = polling_avg(
        df_polls,
        df_bias,
        df_quality,
        election_date="2022-04-03",  # Actual 2022 election date
        lambda_decay=0.03,
        party_list=None
    )
    
    # Step 6: Build forecast distribution
    # Build forecast distribution without producing plots
    df_distr = forecast_distr(
        df_avg,
        sigma_poll_value,
        df_polls,
        election_date="2022-04-03",
        alpha=0.1,
        party_list=None,
    )

    # Step 7: Compute swing coefficients using 2024 EP baseline
    swing_df = swing_coef(df_ep_trans_agg)
    
    # Step 8: Run full simulation for 2022
    rng = np.random.default_rng(random_state)
    seat_results = {party: [] for party in parties}
    
    for s in range(n_sim):
        # Draw national vote shares from Dirichlet
        dirichlet_draw = correl_parties(df_distr, n_draws=1)
        national_draw = {}
        for i, party in enumerate(parties):
            national_draw[party] = dirichlet_draw[0, i]
        
        # Project to districts
        district_projection = OEVK_projection(national_draw, swing_df, sigma_d=sigma_d, random_state=rng)
        
        # Allocate seats
        parliament_seats = seat_simulated(district_projection, df_distr, df_list=df_list, df_oevk=df_oevk, df_ep_trans_agg=df_ep_trans_agg, year=2022)
        for party in parties:
            seat_results[party].append(parliament_seats[party].values[0])
    
    # Step 9: Compute summary statistics from simulation
    sim_summary = []
    sim_seat_distributions = {}
    for party in parties:
        seats = np.array(seat_results[party])
        sim_seat_distributions[party] = seats
        mean_seats = np.mean(seats)
        ci_low = np.percentile(seats, 2.5)
        ci_high = np.percentile(seats, 97.5)
        p_majority = np.mean(seats >= 100)
        p_supermajority = np.mean(seats >= 133)
        sim_summary.append({
            'Party': party,
            'Simulated Mean Seats': int(round(mean_seats)),
            'CI Low': int(round(ci_low)),
            'CI High': int(round(ci_high)),
            'P(majority)': f"{int(round(p_majority*100))}%",
            'P(supermajority)': f"{int(round(p_supermajority*100))}%"
        })
    simulated_results = pd.DataFrame(sim_summary)
    
    # Step 10: Extract actual 2022 seat results
    # Prefer `df_seats` (table with Year, Party, Seats) if provided; otherwise construct from OEVK
    if df_seats is not None:
        df_seats_2022 = df_seats[df_seats['Year'] == 2022].copy()
        # Expect columns ['Year','Party','Seats'] — convert to long format if already so
        if set(['Party', 'Seats']).issubset(df_seats_2022.columns):
            actual_seats_long = df_seats_2022[['Party', 'Seats']].copy()
        else:
            # Fallback to previous approach if columns differ
            df_seats_2022 = df_seats_2022.rename(columns={df_seats_2022.columns[0]: 'Party', df_seats_2022.columns[1]: 'Seats'})
            actual_seats_long = df_seats_2022[['Party', 'Seats']].copy()
        # Also build a wide DataFrame expected by other code (single-row with party columns)
        try:
            actual_seats_wide = actual_seats_long.set_index('Party').T
            # ensure integer seats
            for p in parties:
                if p not in actual_seats_wide.columns:
                    actual_seats_wide[p] = 0
            # Keep column order
            actual_seats_wide = actual_seats_wide[parties]
            actual_seats_2022 = actual_seats_wide.reset_index(drop=True)
        except Exception:
            actual_seats_2022 = pd.DataFrame([ {p: int(actual_seats_long[actual_seats_long['Party']==p]['Seats'].iloc[0]) if (actual_seats_long['Party']==p).any() else 0 for p in parties } ])
    else:
        # Get actual 2022 rows from df_oevk and compute district level shares
        df_2022_oevk = df_oevk[df_oevk['Year'] == 2022].copy()
        df_2022_oevk['Total_Votes'] = df_2022_oevk.groupby(['Megye_No', 'OEVK'])['OEVK_Votes'].transform('sum')
        df_2022_oevk['Vote_Share'] = df_2022_oevk['OEVK_Votes'] / df_2022_oevk['Total_Votes'] * 100

        pivot = df_2022_oevk.pivot_table(
            index=['Megye_No', 'Megye', 'OEVK'],
            columns='Party6',
            values='Vote_Share',
            fill_value=0
        ).reset_index()

        # Ensure all expected party columns exist
        for party in parties:
            if party not in pivot.columns:
                pivot[party] = 0.0

        # Determine Winner per district (party with highest vote share)
        pivot['Winner'] = pivot[parties].idxmax(axis=1)

        cols = ['Megye_No', 'Megye', 'OEVK'] + parties + ['Winner']
        district_projection_actual = pivot[cols]

        actual_seats_2022 = seat_simulated(district_projection_actual, df_distr, df_list=df_list, df_oevk=df_oevk, df_ep_trans_agg=df_ep_trans_agg, year=2022)

        # Convert to long format
        try:
            actual_seats_long = actual_seats_2022.melt(var_name='Party', value_name='Seats')[['Party','Seats']]
        except Exception:
            actual_seats_long = pd.DataFrame([{ 'Party': p, 'Seats': int(actual_seats_2022[p].iloc[0]) if p in actual_seats_2022.columns else 0 } for p in parties])

    # Convert seat_simulated output (one-row wide dataframe with party columns)
    # to a long format DataFrame with columns ['Party', 'Seats'] for easier comparison
    try:
        # If seat_simulated returned a single-row dataframe with party columns
        actual_seats_long = actual_seats_2022.melt(var_name='Party', value_name='Seats')[[ 'Party', 'Seats' ]]
    except Exception:
        # Fallback: build from parties list
        actual_seats_long = pd.DataFrame([{ 'Party': p, 'Seats': int(actual_seats_2022[p].iloc[0]) if p in actual_seats_2022.columns else 0 } for p in parties])
    
    # Step 11: Create comparison table
    comparison = []
    for party in parties:
        sim_row = simulated_results[simulated_results['Party'] == party].iloc[0]
        actual_row = actual_seats_long[actual_seats_long['Party'] == party]

        if not actual_row.empty:
            actual_seats = int(actual_row['Seats'].values[0])
        else:
            actual_seats = 0
        
        ci_low = sim_row['CI Low']
        ci_high = sim_row['CI High']
        within_ci = (actual_seats >= ci_low) and (actual_seats <= ci_high)
        
        comparison.append({
            'Party': party,
            'Simulated Mean': sim_row['Simulated Mean Seats'],
            '95% CI': f"{ci_low}–{ci_high}",
            'Actual Seats': actual_seats,
            'Within 95% CI': 'Yes' if within_ci else 'No'
        })
    
    comparison_df = pd.DataFrame(comparison)

    # Return only the final comparison table (party, simulated mean, 95% CI, actual seats,
    # and a Within 95% CI flag) as requested.
    return comparison_df



