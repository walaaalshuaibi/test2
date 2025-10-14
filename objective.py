import pandas as pd
from collections import Counter

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
    balance_weight=10, 
    hard_day_weight=5, 
    diverse_weight=5, 
    role_pref_weight=10,
    nf_day_pref_weight=10
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

    # Final objective
    model.Maximize(sum(terms))

def extract_schedule(
    solver, 
    assign, 
    days, 
    roles, 
    residents, 
    wr_residents, 
    ns_residents, 
    night_counts, 
    score_vars, 
    max_shifts, 
    max_points, 
    nf_calendar_df
):
    """ 
    Extract the solved schedule and compute per-resident scores. 
    Returns:
        schedule_df: DataFrame with daily assignments (day + NF roles).
        scores_df: DataFrame with per-resident totals, score, and WR info.
    """
    
    schedule = []
    
    # Track how many shifts each resident worked
    day_shift_counts = {r: 0 for r in residents}
    
    # Build the daily schedule from solver
    for d in days:
        row = {"date": d.date(), "day": d.strftime('%a')}
        
        for role in roles:
            for r in residents:
                if solver.Value(assign[(d, role, r)]) == 1:
                    row[role] = r
                    day_shift_counts[r] += 1
        
        schedule.append(row)
    
    schedule_df = pd.DataFrame(schedule)
    
    # Merge NF calendar if provided
    if nf_calendar_df is not None and not nf_calendar_df.empty:
        # Ensure Date is aligned (convert both to datetime.date)
        schedule_df["date"] = pd.to_datetime(schedule_df["date"]).dt.date
        nf_calendar_df = nf_calendar_df.copy()
        nf_calendar_df["date"] = pd.to_datetime(nf_calendar_df["date"]).dt.date
        
        # Drop duplicate Day column from NF calendar if present
        if "day" in nf_calendar_df.columns:
            nf_calendar_df = nf_calendar_df.drop(columns=["day"])
        
        # Merge side by side
        schedule_df = schedule_df.merge(nf_calendar_df, on="date", how="left")
    
    # Add WR (Weekend Round) column if EW Day exists
    if 'day' in schedule_df.columns and 'ew day' in schedule_df.columns:
        schedule_df['wr'] = schedule_df.apply(
            lambda row: row['ew day'] if row['day'] == 'Fri' else '', 
            axis=1
        )
        # --- Count how many WRs each resident has ---
        wr_counts = (
            schedule_df['wr']
            .value_counts()
            .to_dict()
        )
    else:
        wr_counts = {r: 0 for r in residents}
    
    # Build per-resident summary (scores_df)
    scores_df = pd.DataFrame([{
        "Resident": r,
        "Total Shifts": day_shift_counts[r],
        "Score": solver.Value(score_vars[r]),
        "Max Shifts": max_shifts[r],
        "Max Points": max_points[r],
        "WR Resident": wr_counts.get(r, 0), 
        "NF Resident": "Yes" if night_counts[r] > 2 else "No",
        "NS Resident": "Yes" if r in set(ns_residents["resident"]) else "No"
    } for r in residents])
    
    return schedule_df, scores_df