import pandas as pd
from collections import Counter, defaultdict

# ------------------------- 
# Objective 
# -------------------------

def build_objective(
    model, 
    score_vars, 
    balance_penalties=None, 
    hard_day_penalties=None, 
    diverse_penalties=None, 
    role_pref_penalties=None, 
    nf_day_pref_penalties=None,
    wr_penalties=None,
    spacing_penalties=None,
    balance_weight=10, 
    hard_day_weight=5, 
    diverse_weight=5, 
    role_pref_weight=10,
    nf_day_pref_weight=10,
    wr_pref_weight=10,
    spacing_weight=5
):
    """ 
    Build the optimization objective using resident scores. 
    Objective:
    - Maximize total resident scores (weekday=1, weekend=2, +2 weekend round bonus).
    - Subtract weighted penalties for imbalance (balance_penalties).
    - Subtract weighted penalties for hard-day violations (hard_day_penalties).
    - Subtract weighted penalties for lack of role diversity (diverse_penalties).
    """
    
    terms = []
    
    # Base objective: maximize total scores
    terms.extend(score_vars.values())
    
    # Balance penalties
    if balance_penalties:
        terms.extend([-balance_weight * p for p in balance_penalties])
    
    # Hard day penalties
    if hard_day_penalties:
        terms.extend([-hard_day_weight * p for p in hard_day_penalties])
    
    # Diverse rotation penalties
    if diverse_penalties:
        terms.extend([-diverse_weight * p for p in diverse_penalties])
    
    # Role preferences penalties
    if role_pref_penalties:
        terms.extend([-role_pref_weight * p for p in role_pref_penalties])

    # NF day preference 
    if nf_day_pref_penalties:
        terms.extend([-nf_day_pref_weight * p for p in nf_day_pref_penalties])

    # WR preference 
    if wr_penalties:
        terms.extend([-wr_pref_weight * p for p in wr_penalties])

    if spacing_penalties:
        terms.extend([-spacing_weight * p for p in spacing_penalties])

    # Final objective
    model.Maximize(sum(terms))

import pandas as pd
from collections import defaultdict

import pandas as pd
from collections import defaultdict

def extract_schedule(
    solver, 
    assign, 
    days, 
    roles, 
    residents, 
    wr_residents, 
    weekend_rounds_df,
    ns_residents, 
    night_counts, 
    score_vars, 
    max_shifts, 
    max_points, 
    nf_calendar_df,
    resident_year
):
    """ 
    Extract the solved schedule and compute per-resident scores. 

    Rules:
    - R1 residents: never assigned WR/EW.
    - Seniors: only Friday counts as WR.
    - Other years: Friday and Saturday both count as WR.
    - WR counts are grouped by weekend (Fri+Sat = one WR).
    
    Returns:
        schedule_df: DataFrame with daily assignments (day + NF roles + WR).
        scores_df: DataFrame with per-resident totals, score, and WR info.
    """
    
    schedule = []
    day_shift_counts = {r: 0 for r in residents}

    # --- Build the daily schedule from solver ---
    for d in days:
        row = {"date": d.date(), "day": d.strftime('%a')}
        for role in roles:
            for r in residents:
                if solver.Value(assign[(d, role, r)]) == 1:
                    row[role] = r
                    day_shift_counts[r] += 1
        schedule.append(row)

    # Add NS shifts
    if ns_residents is not None and not ns_residents.empty:
        for _, row in ns_residents.iterrows():
            r = row["name"]
            day_shift_counts[r] += 1  

    schedule_df = pd.DataFrame(schedule)

    # --- Merge NF calendar if provided ---
    if nf_calendar_df is not None and not nf_calendar_df.empty:
        schedule_df["date"] = pd.to_datetime(schedule_df["date"]).dt.date
        nf_calendar_df = nf_calendar_df.copy()
        nf_calendar_df["date"] = pd.to_datetime(nf_calendar_df["date"]).dt.date
        if "day" in nf_calendar_df.columns:
            nf_calendar_df = nf_calendar_df.drop(columns=["day"])
        schedule_df = schedule_df.merge(nf_calendar_df, on="date", how="left")

    # --- WR assignment logic using weekend_rounds_df ---
    wr_counts = {r: 0 for r in residents}
    wr_column_map = defaultdict(list)

    if weekend_rounds_df is not None and not weekend_rounds_df.empty:
        # Normalize dates
        weekend_rounds_df = weekend_rounds_df.copy()
        weekend_rounds_df["date"] = pd.to_datetime(weekend_rounds_df["date"]).dt.date

        # Count unique WR dates per resident
        wr_counts.update(
            weekend_rounds_df.groupby("name")["date"].nunique().to_dict()
        )

        # Build mapping: date -> list of residents
        for d, group in weekend_rounds_df.groupby("date"):
            wr_column_map[d] = group["name"].tolist()

    # Add WR column to schedule_df (comma-separated names if multiple)
    schedule_df["wr"] = schedule_df["date"].apply(
        lambda d: ", ".join(wr_column_map[d]) if d in wr_column_map else ""
    )

    # --- Build per-resident summary ---
    scores_df = pd.DataFrame([{
        "Name": r,
        "Total Shifts": day_shift_counts[r],
        "Score": solver.Value(score_vars[r]),
        "Max Shifts": max_shifts[r],
        "Max Points": max_points[r],
        "WR Count": wr_counts.get(r, 0),
        "NF Resident": "Yes" if night_counts[r] > 2 else "No",
        "NS Resident": "Yes" if r in set(ns_residents["name"]) else "No"
    } for r in residents])
    
    return schedule_df, scores_df