import numpy as np
import pandas as pd

def seat_simulated(district_projection, df_distr, df_list=None, df_oevk=None, df_ep_trans_agg=None, year=2022):
	"""
	Simulate Hungarian Parliament seat allocation using actual vote counts (not percentages).
	
	Args:
		district_projection: DataFrame with OEVK results (Winner column, party vote columns as percentages)
		df_distr: DataFrame with national list votes (Party, Mu columns) - used as fallback
		df_list: DataFrame with national election results (Party_List_Votes, Valid_Votes columns by year)
		df_oevk: DataFrame with district results (Sum_of_Valid_Votes by OEVK for 2014, 2018, 2022)
		df_ep_trans_agg: DataFrame with district results (Sum_of_Valid_Votes by OEVK for 2019, 2024)
		year: int, the election year (default 2022)
	Returns:
		DataFrame: one row, columns are parties, values are total seats
	"""
	parties = ['Fidesz', 'Tisza', 'MiHazánk', 'DK', 'MKKP', 'Other']
	
	# 1. SMD seats (106): count wins per party
	smd_wins = district_projection['Winner'].value_counts().reindex(parties, fill_value=0)
	
	# 2. National list votes (in actual counts)
	if df_list is not None:
		year_data = df_list[df_list['Year'] == year]
		if not year_data.empty:
			# Get total valid votes for the year
			# Determine total valid votes robustly (different source frames may use different column names)
			if 'Valid_Votes' in year_data.columns and year_data['Valid_Votes'].notna().any():
				total_valid_votes = year_data['Valid_Votes'].dropna().iloc[0]
			elif 'Voters' in year_data.columns and year_data['Voters'].notna().any():
				total_valid_votes = year_data['Voters'].dropna().iloc[0]
			elif 'Party_List_Votes' in year_data.columns and year_data['Party_List_Votes'].notna().any():
				# Sum of party list votes across parties equals total list votes; use as fallback
				total_valid_votes = year_data['Party_List_Votes'].sum()
			else:
				# Last-resort fallback: estimate from df_oevk if available
				total_valid_votes = None

			# Determine party column explicitly and get party list votes (prefer actual counts when available)
			party_col = 'Party6' if 'Party6' in year_data.columns else ('Party' if 'Party' in year_data.columns else None)
			list_votes = {}
			for party in parties:
				if party_col is not None:
					party_data = year_data[year_data[party_col] == party]
				else:
					party_data = pd.DataFrame()
				if not party_data.empty and 'Party_List_Votes' in party_data.columns:
					list_votes[party] = party_data['Party_List_Votes'].iloc[0]
				else:
					list_votes[party] = np.nan
		else:
			# No df_list rows for this year - mark list_votes as NaN to be filled from fallbacks below
			list_votes = {party: np.nan for party in parties}
	else:
		# No df_list provided — mark list_votes as NaN to be filled from fallbacks below
		list_votes = {party: np.nan for party in parties}
	
	# 3. OEVK losing candidates votes and surplus votes (in actual counts)
	# district_projection has vote percentages; convert to actual votes
	losing_votes = {party: 0.0 for party in parties}
	surplus_votes = {party: 0.0 for party in parties}
	
	for idx, row in district_projection.iterrows():
		# Get the district's Megye_No and OEVK to look up total valid votes
		megye_no = row.get('Megye_No')
		oevk = row.get('OEVK')
		
		# Look up total valid votes for this district
		district_valid_votes = None
		# Prefer exact match for the requested year in df_oevk
		if df_oevk is not None and 'Sum_of_Valid_Votes' in df_oevk.columns:
			district_data = df_oevk[(df_oevk['Megye_No'] == megye_no) & (df_oevk['OEVK'] == oevk)]
			# Try same-year first
			district_year_match = district_data[district_data['Year'] == year]
			if not district_year_match.empty:
				district_valid_votes = district_year_match['Sum_of_Valid_Votes'].iloc[0]
			elif not district_data.empty:
				# Fall back to average across years for this district
				district_valid_votes = district_data['Sum_of_Valid_Votes'].mean()
		
		# If still None, try EP aggregated frame
		if (district_valid_votes is None or district_valid_votes == 0) and df_ep_trans_agg is not None and 'Sum_of_Valid_Votes' in df_ep_trans_agg.columns:
			district_data = df_ep_trans_agg[(df_ep_trans_agg['Megye_No'] == megye_no) & (df_ep_trans_agg['OEVK'] == oevk)]
			district_year_match = district_data[district_data['Year'] == year]
			if not district_year_match.empty:
				district_valid_votes = district_year_match['Sum_of_Valid_Votes'].iloc[0]
			elif not district_data.empty:
				district_valid_votes = district_data['Sum_of_Valid_Votes'].mean()
		
		# If not found, fall back to average district size across available data
		if district_valid_votes is None or district_valid_votes == 0:
			# Try global mean from df_oevk
			avg_per_district = None
			if df_oevk is not None and 'Sum_of_Valid_Votes' in df_oevk.columns:
				try:
					avg_per_district = df_oevk.groupby(['Megye_No','OEVK'])['Sum_of_Valid_Votes'].mean().mean()
				except Exception:
					avg_per_district = None
			# else try ep frame
			if (avg_per_district is None or np.isnan(avg_per_district) or avg_per_district == 0) and df_ep_trans_agg is not None and 'Sum_of_Valid_Votes' in df_ep_trans_agg.columns:
				try:
					avg_per_district = df_ep_trans_agg.groupby(['Megye_No','OEVK'])['Sum_of_Valid_Votes'].mean().mean()
				except Exception:
					avg_per_district = None
			if avg_per_district is not None and not np.isnan(avg_per_district) and avg_per_district > 0:
				district_valid_votes = avg_per_district
				print(f"Warning: district {megye_no}-{oevk} missing valid votes; using global avg {district_valid_votes:.1f} as fallback.")
			else:
				# Last-resort fallback to 1 to avoid silent skipping
				district_valid_votes = 1.0
				print(f"Warning: district {megye_no}-{oevk} missing valid votes; using fallback value 1.")
		
		# Convert vote percentages to actual vote counts
		party_votes = {}
		for party in parties:
			vote_share_pct = row.get(party, 0)
			party_votes[party] = (vote_share_pct / 100.0) * district_valid_votes
		
		# Find winner and runner-up
		sorted_parties = sorted(party_votes.items(), key=lambda x: x[1], reverse=True)
		winner = sorted_parties[0][0]
		winner_votes = sorted_parties[0][1]
		runner_up_votes = sorted_parties[1][1] if len(sorted_parties) > 1 else 0
		
		# Surplus for winner: use winner - runner_up - 1 (one-vote margin), floor at 0
		surplus = max(winner_votes - runner_up_votes - 1, 0)
		surplus_votes[winner] += surplus
		
		# Losing votes (all votes not cast for the winner)
		for party in parties:
			if party != winner:
				losing_votes[party] += party_votes[party]
	
	# 4. Effective list total (actual vote counts)
	# If some list_votes are NaN (missing df_list), try to estimate counts from df_distr Mu percentages
	if any(pd.isna(list_votes[p]) for p in parties):
		# Estimate total_valid_votes if not available yet
		if 'total_valid_votes' not in locals() or total_valid_votes is None:
			# Try to estimate from df_oevk (sum of Sum_of_Valid_Votes for requested year)
			if df_oevk is not None and 'Sum_of_Valid_Votes' in df_oevk.columns:
				df_year = df_oevk[df_oevk['Year'] == year]
				if not df_year.empty:
					total_valid_votes = df_year['Sum_of_Valid_Votes'].sum()
				else:
					# fallback: average per district * number districts
					n_districts = df_oevk[['Megye_No','OEVK']].drop_duplicates().shape[0]
					avg = df_oevk.groupby(['Megye_No','OEVK'])['Sum_of_Valid_Votes'].mean().mean()
					total_valid_votes = avg * n_districts
			elif df_ep_trans_agg is not None and 'Sum_of_Valid_Votes' in df_ep_trans_agg.columns:
				df_year = df_ep_trans_agg[df_ep_trans_agg['Year'] == year]
				if not df_year.empty:
					total_valid_votes = df_year['Sum_of_Valid_Votes'].sum()
				else:
					n_districts = df_ep_trans_agg[['Megye_No','OEVK']].drop_duplicates().shape[0]
					avg = df_ep_trans_agg.groupby(['Megye_No','OEVK'])['Sum_of_Valid_Votes'].mean().mean()
					total_valid_votes = avg * n_districts
			else:
				# As a last resort, set total_valid_votes to 1 (will make list_votes fractional but preserves ranking)
				total_valid_votes = 1.0

		# Fill list_votes from df_distr Mu percentages (Mu is percentage)
		# Build a Mu lookup map once for efficiency
		mu_map = {}
		if df_distr is not None and 'Party' in df_distr.columns and 'Mu' in df_distr.columns:
			try:
				mu_map = df_distr.set_index('Party')['Mu'].to_dict()
			except Exception:
				mu_map = {}
		for party in parties:
			if pd.isna(list_votes[party]):
				mu = mu_map.get(party, 0.0)
				list_votes[party] = (mu / 100.0) * total_valid_votes

	# 5. Apply 5% threshold to RAW LIST VOTES (before adding SMD fragments)
	total_list_votes = sum(list_votes.values())
	
	if total_list_votes > 0:
		# Threshold: 5% of raw national list votes
		eligible = [party for party in parties if list_votes[party] >= 0.05 * total_list_votes]
	else:
		eligible = parties  # Fallback: all parties eligible if no list votes
	
	# 6. Calculate effective list votes (only used for d'Hondt allocation, not for threshold)
	effective_list = {party: list_votes[party] + losing_votes[party] + surplus_votes[party] for party in parties}
	total_list = sum(effective_list.values())
	
	# 7. d'Hondt allocation for 93 seats (only eligible parties compete)
	list_seats = {party: 0 for party in parties}
	for _ in range(93):
		quotients = {party: effective_list[party] / (list_seats[party] + 1) if party in eligible else 0 for party in parties}
		winner = max(quotients, key=quotients.get)
		list_seats[winner] += 1
	
	# 8. Total seats (ensure integer counts)
	total_seats = {party: int(round(smd_wins.get(party, 0) + list_seats.get(party, 0))) for party in parties}
	
	# Output DataFrame
	return pd.DataFrame([total_seats], columns=parties)

