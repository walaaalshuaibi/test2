import pandas as pd
from collections import Counter, defaultdict
import helper

# ------------------------- 
# Objective 
# -------------------------

def build_objective(
    model, 
    score_vars, 
    non_nf_balance_penalties=None,
    nf_balance_penalties=None, 
    hard_day_penalties=None, 
    diverse_penalties=None, 
    role_pref_penalties=None, 
    nf_day_pref_penalties=None,
    wr_penalties=None,
    spacing_penalties=None,
    spacing_fairness_penalties=None,
    weekend_vs_tues_thurs_penalties=None,
    non_nf_balance_weight = 1000000,
    nf_balance_weight = 10000,
    hard_day_weight=5, 
    diverse_weight=5, 
    role_pref_weight=10,
    nf_day_pref_weight=10,
    wr_pref_weight=10,
    spacing_weight=5,
    spacing_fairness_weight=5,
    weekend_vs_tues_thurs_weight=5
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
    #terms.extend(score_vars.values())
    
    # Balance penalties
    if non_nf_balance_penalties:
        terms.extend([non_nf_balance_weight * p for p in non_nf_balance_penalties])

    if nf_balance_penalties:
        terms.extend([nf_balance_weight * p for p in nf_balance_penalties])
    
    # Hard day penalties
    if hard_day_penalties:
        terms.extend([hard_day_weight * p for p in hard_day_penalties])
    
    # Diverse rotation penalties
    if diverse_penalties:
        terms.extend([diverse_weight * p for p in diverse_penalties])
    
    # Role preferences penalties
    if role_pref_penalties:
        terms.extend([role_pref_weight * p for p in role_pref_penalties])

    # NF day preference 
    if nf_day_pref_penalties:
        terms.extend([nf_day_pref_weight * p for p in nf_day_pref_penalties])

    # WR preference 
    if wr_penalties:
        terms.extend([wr_pref_weight * p for p in wr_penalties])

    if spacing_penalties:
        terms.extend([spacing_weight * p for p in spacing_penalties])

    if spacing_fairness_penalties:
        terms.extend([spacing_fairness_weight * p for p in spacing_fairness_penalties])
    
    if weekend_vs_tues_thurs_penalties:
        terms.extend([weekend_vs_tues_thurs_weight * p for p in weekend_vs_tues_thurs_penalties])

    # Final objective
    model.Minimize(sum(terms))

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
    resident_levels,
    limited_shift_residents
):
    """ 
    Extract solved schedule, compute per-resident scores, spacing stats, per-role counts,
    weekday counts (weekend Fri+Sat, Thursday, Tuesday) and Year.

    Returns:
        schedule_df, scores_df
    """
    schedule = []
    day_shift_counts = {r: 0 for r in residents}

    # Initialize role counts per resident
    role_counts = {r: {role: 0 for role in roles} for r in residents}

    # Weekday-specific counters per resident
    weekend_counts_per_res = {r: 0 for r in residents}   # Friday or Saturday
    thursday_counts_per_res = {r: 0 for r in residents}
    tuesday_counts_per_res = {r: 0 for r in residents}

    # --- Build the daily schedule from solver ---
    for d in days:
        date_val = d.date() if hasattr(d, "date") else d
        weekday_str = pd.to_datetime(date_val).strftime('%a')  # 'Mon','Tue',...,'Sun'
        row = {"date": date_val, "day": weekday_str}
        for role in roles:
            row[role] = ""
            for r in residents:
                try:
                    if solver.Value(assign[(d, role, r)]) == 1:
                        row[role] = r
                        day_shift_counts[r] += 1
                        role_counts[r][role] += 1
                        if weekday_str in ("Fri", "Sat"):
                            weekend_counts_per_res[r] += 1
                        if weekday_str == "Thu":
                            thursday_counts_per_res[r] += 1
                        if weekday_str == "Tue":
                            tuesday_counts_per_res[r] += 1
                except Exception:
                    pass
        schedule.append(row)

    # --- Normalize ns_residents into set of names / iterable rows and add to counts ---
    ns_names = set()
    ns_rows_iter = []
    if ns_residents is None:
        ns_names = set()
        ns_rows_iter = []
    elif isinstance(ns_residents, pd.DataFrame):
        if "name" in ns_residents.columns and not ns_residents.empty:
            ns_names = set(ns_residents["name"].tolist())
            ns_rows_iter = ns_residents.itertuples(index=False)
    elif isinstance(ns_residents, dict):
        ns_names = set(ns_residents.keys())
        ns_rows_iter = [(None, {"name": n}) for n in ns_names]
    else:
        try:
            # list of names
            if all(isinstance(x, str) for x in ns_residents):
                ns_names = set(ns_residents)
                ns_rows_iter = [(None, {"name": n}) for n in ns_names]
            else:
                names = []
                rows = []
                for item in ns_residents:
                    if isinstance(item, dict) and "name" in item:
                        names.append(item["name"])
                        rows.append(item)
                ns_names = set(names)
                ns_rows_iter = rows
        except Exception:
            ns_names = set()
            ns_rows_iter = []

    # Add NS shifts (increment total shift counts)
    if ns_rows_iter:
        for entry in ns_rows_iter:
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], dict):
                r = entry[1].get("name")
            elif hasattr(entry, "name"):
                r = getattr(entry, "name", None)
            elif isinstance(entry, dict):
                r = entry.get("name")
            else:
                r = entry if isinstance(entry, str) else None

            if r and r in day_shift_counts:
                day_shift_counts[r] += 1
    else:
        for r in ns_names:
            if r in day_shift_counts:
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
        weekend_rounds_df = weekend_rounds_df.copy()
        weekend_rounds_df["date"] = pd.to_datetime(weekend_rounds_df["date"]).dt.date
        wr_counts.update(weekend_rounds_df.groupby("name")["date"].nunique().to_dict())
        for d, group in weekend_rounds_df.groupby("date"):
            wr_column_map[d] = group["name"].tolist()
    schedule_df["wr"] = schedule_df["date"].apply(lambda d: ", ".join(wr_column_map[d]) if d in wr_column_map else "")

    # --- Compute spacing stats per resident (gaps in days between consecutive assigned days) ---
    # --- Compute spacing stats using ALL assigned sources ---
    per_resident_spacing = {}
    all_gaps = []

    # --- Compute per-resident spacing ---
    for r in residents:
        assigned_dates = helper.get_all_assigned_dates(
            r=r,
            solver=solver,
            assign=assign,
            days=days,
            roles=roles,
            ns_df=ns_residents,
            wr_df=weekend_rounds_df,
            nf_calendar_df=nf_calendar_df,
            extra_preassigned=None
        )

        gaps = [(assigned_dates[i] - assigned_dates[i-1]).days for i in range(1, len(assigned_dates))]

        per_resident_spacing[r] = {
            "gaps": gaps,
            "num_gaps": len(gaps),
            "avg_gap": (sum(gaps) / len(gaps) if gaps else None),
            "min_gap": min(gaps) if gaps else None,
            "max_gap": max(gaps) if gaps else None
        }

        all_gaps.extend(gaps)

    # --- Compute overall spacing ---
    spacing_overall = {
        "num_gaps": len(all_gaps),
        "avg_gap": (sum(all_gaps) / len(all_gaps)) if all_gaps else None,
        "min_gap": (min(all_gaps) if all_gaps else None),
        "max_gap": (max(all_gaps) if all_gaps else None)
    }

    # --- Build per-resident summary ---
    scores_rows = []
    for r in residents:
        spacing = per_resident_spacing.get(r, {})

        if limited_shift_residents and r in limited_shift_residents:
            effective_max_shifts = limited_shift_residents[r]
        else:
            effective_max_shifts = max_shifts.get(r, None) if isinstance(max_shifts, dict) else max_shifts

        row = {
            "Name": r,
            "Total Shifts": day_shift_counts.get(r, 0),
            "Score": solver.Value(score_vars[r]) if r in score_vars else None,
            "Max Shifts": effective_max_shifts,
            "Max Points": max_points.get(r, None) if isinstance(max_points, dict) else max_points,
            "WR Count": wr_counts.get(r, 0),
            "NF Resident": "Yes" if night_counts.get(r, 0) > 2 else "No",
            "NS Resident": "Yes" if r in ns_names else "No"
            # # Individual spacing
            # , "spacing_avg": spacing.get("avg_gap"),
            # "spacing_min": spacing.get("min_gap"),
            # "spacing_max": spacing.get("max_gap"),
            # "spacing_gaps_count": spacing.get("num_gaps", 0),
            # # Overall spacing
            # "spacing_overall_avg": spacing_overall["avg_gap"],
            # "spacing_overall_min": spacing_overall["min_gap"],
            # "spacing_overall_max": spacing_overall["max_gap"],
            # "spacing_overall_gaps_count": spacing_overall["num_gaps"],
            # "weekend_shifts": weekend_counts_per_res.get(r, 0),
            # "thursday_shifts": thursday_counts_per_res.get(r, 0),
            # "tuesday_shifts": tuesday_counts_per_res.get(r, 0),
            # "Year": resident_levels.get(r)
        }

        for role in roles:
            col_name = f"role_{role}_count"
            row[col_name] = role_counts.get(r, {}).get(role, 0)

        scores_rows.append(row)

    scores_df = pd.DataFrame(scores_rows)


    return schedule_df, scores_df

