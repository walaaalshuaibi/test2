import pandas as pd
from collections import Counter, defaultdict
import helper

# ------------------------- 
# Objective 
# -------------------------

def build_objective(
    model,

    # Balance
    non_nf_balance_penalties=None,
    non_nf_balance_weight=1,
    nf_balance_penalties=None,
    nf_balance_weight=1,

    # Spacing (Soft)
    spacing_nonnf_soft_penalties=None,
    non_nf_spacing_weight=1,
    spacing_nf_soft_penalties=None,
    nf_spacing_weight=1,
    spacing_ns_soft_penalties=None,
    ns_spacing_weight=1,

    # Hard days
    hard_nonhard_penalties=None,
    hard_nonhard_weight=1,
    hard_diversity_penalties=None,
    hard_diversity_weight=1,
    hard_max_penalties=None,
    hard_max_weight=1,

    # Roles
    diverse_role_penalties=None,
    diverse_role_weight=1,
    diverse_days_penalties=None,
    diverse_days_weight=1,
    role_pref_penalties=None,
    role_pref_weight=1,
    nf_day_pref_penalties=None,
    nf_day_pref_weight=1,
    wr_penalties=None,
    wr_pref_weight=1,

    night_thursday_penalties=None,
    night_thursday_weight=1
):
    """
    Build the optimization objective from penalty components.
    Objective minimizes weighted penalties.
    """
    terms = []

    # --- Balance penalties ---
    if non_nf_balance_penalties:
        terms.extend([non_nf_balance_weight * p for p in non_nf_balance_penalties])

    if nf_balance_penalties:
        terms.extend([nf_balance_weight * p for p in nf_balance_penalties])

    # --- Spacing penalties ---
    if spacing_nonnf_soft_penalties:
        terms.extend([non_nf_spacing_weight * p for p in spacing_nonnf_soft_penalties])

    if spacing_nf_soft_penalties:
        terms.extend([nf_spacing_weight * p for p in spacing_nf_soft_penalties])

    if spacing_ns_soft_penalties:
        terms.extend([ns_spacing_weight * p for p in spacing_ns_soft_penalties])

    # --- Hard day penalties ---
    if hard_nonhard_penalties:
        terms.extend([hard_nonhard_weight * p for p in hard_nonhard_penalties])

    if hard_diversity_penalties:
        terms.extend([hard_diversity_weight * p for p in hard_diversity_penalties])

    if hard_max_penalties:
        terms.extend([hard_max_weight * p for p in hard_max_penalties])

    # --- Role penalties ---
    if diverse_role_penalties:
        terms.extend([diverse_role_weight * p for p in diverse_role_penalties])
    
    if diverse_days_penalties:
        terms.extend([diverse_days_weight * p for p in diverse_days_penalties])

    if role_pref_penalties:
        terms.extend([role_pref_weight * p for p in role_pref_penalties])

    if nf_day_pref_penalties:
        terms.extend([nf_day_pref_weight * p for p in nf_day_pref_penalties])

    if wr_penalties:
        terms.extend([wr_pref_weight * p for p in wr_penalties])

    if night_thursday_penalties:
        terms.extend([night_thursday_weight * p for p in wr_penalties])

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
    Also adds a 'WEEKEND ROLE' column for each resident.
    
    Returns:
        schedule_df, scores_df
    """
    from collections import defaultdict
    import pandas as pd

    # --- Initialize counts ---
    schedule = []
    day_shift_counts = {r: 0 for r in residents}
    role_counts = {r: {role: 0 for role in roles} for r in residents}
    weekend_counts_per_res = {r: 0 for r in residents}
    thursday_counts_per_res = {r: 0 for r in residents}
    tuesday_counts_per_res = {r: 0 for r in residents}

    # --- Build daily schedule from solver ---
    for d in days:
        date_val = pd.to_datetime(d).date()  # normalize to date
        weekday_str = date_val.strftime('%a')
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

    schedule_df = pd.DataFrame(schedule)

    # --- NS resident normalization ---
    ns_names = set()
    ns_rows_iter = []
    if ns_residents is not None:
        if isinstance(ns_residents, pd.DataFrame) and "name" in ns_residents.columns:
            ns_names = set(ns_residents["name"].tolist())
            ns_rows_iter = ns_residents.itertuples(index=False)
        elif isinstance(ns_residents, dict):
            ns_names = set(ns_residents.keys())
            ns_rows_iter = [(None, {"name": n}) for n in ns_names]
        elif isinstance(ns_residents, list) and all(isinstance(x, str) for x in ns_residents):
            ns_names = set(ns_residents)
            ns_rows_iter = [(None, {"name": n}) for n in ns_names]

    # Add NS shifts
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

    for r in ns_names:
        if r in day_shift_counts:
            day_shift_counts[r] += 1

    # --- Merge NF calendar if provided ---
    if nf_calendar_df is not None and not nf_calendar_df.empty:
        schedule_df["date"] = pd.to_datetime(schedule_df["date"]).dt.date
        nf_calendar_df = nf_calendar_df.copy()
        nf_calendar_df["date"] = pd.to_datetime(nf_calendar_df["date"]).dt.date
        if "day" in nf_calendar_df.columns:
            nf_calendar_df = nf_calendar_df.drop(columns=["day"])
        schedule_df = schedule_df.merge(nf_calendar_df, on="date", how="left")

    # --- WR assignment logic ---
    wr_counts = {r: 0 for r in residents}
    wr_column_map = defaultdict(list)
    if weekend_rounds_df is not None and not weekend_rounds_df.empty:
        weekend_rounds_df = weekend_rounds_df.copy()
        weekend_rounds_df["date"] = pd.to_datetime(weekend_rounds_df["date"]).dt.date
        wr_counts.update(weekend_rounds_df.groupby("name")["date"].nunique().to_dict())
        for d, group in weekend_rounds_df.groupby("date"):
            wr_column_map[d] = group["name"].tolist()
    schedule_df["wr"] = schedule_df["date"].apply(lambda d: ", ".join(wr_column_map[d]) if d in wr_column_map else "")

    # --- Compute spacing stats per resident ---
    per_resident_spacing = {}
    all_gaps = []
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

    spacing_overall = {
        "num_gaps": len(all_gaps),
        "avg_gap": (sum(all_gaps) / len(all_gaps)) if all_gaps else None,
        "min_gap": (min(all_gaps) if all_gaps else None),
        "max_gap": (max(all_gaps) if all_gaps else None)
    }

    # --- Compute weekend roles per resident ---
    weekend_roles_per_res = {r: set() for r in residents}
    weekend_days_set = {pd.to_datetime(d).date() for d in days if pd.to_datetime(d).strftime('%a') in ['Fri', 'Sat']}
    for _, row in schedule_df.iterrows():
        d = pd.to_datetime(row['date']).date()
        if d in weekend_days_set:
            for role in roles:
                r = row.get(role)
                if r and r in residents:
                    weekend_roles_per_res[r].add(role)

    # --- Build per-resident summary ---
    scores_rows = []
    for r in residents:
        spacing = per_resident_spacing.get(r, {})

        # Get all assigned dates for this resident
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

        # Convert to weekday names and remove duplicates
        assigned_weekdays = sorted({d.strftime("%a") for d in assigned_dates}, 
                                key=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(x))

        effective_max_shifts = (limited_shift_residents[r] if limited_shift_residents and r in limited_shift_residents 
                                else (max_shifts.get(r) if isinstance(max_shifts, dict) else max_shifts))
        row = {
            "Name": r,
            "Total Shifts": len(assigned_dates),
            "Score": solver.Value(score_vars[r]) if r in score_vars else None,
            "Max Shifts": effective_max_shifts,
            "Max Points": max_points.get(r) if isinstance(max_points, dict) else max_points,
            "WR Count": wr_counts.get(r, 0),
            "NF Resident": "Yes" if night_counts.get(r, 0) > 2 else "No",
            "NS Resident": "Yes" if r in ns_names else "No",
            # "Assigned Days": ", ".join(assigned_weekdays),  # <-- NEW: weekdays
            # "spacing_avg": spacing.get("avg_gap"),
            # "spacing_min": spacing.get("min_gap"),
            # "spacing_max": spacing.get("max_gap"),
            # "spacing_gaps_count": spacing.get("num_gaps", 0),
            # "spacing_overall_avg": spacing_overall["avg_gap"],
            # "spacing_overall_min": spacing_overall["min_gap"],
            # "spacing_overall_max": spacing_overall["max_gap"],
            # "spacing_overall_gaps_count": spacing_overall["num_gaps"],
            # "weekend_shifts": weekend_counts_per_res.get(r, 0),
            # "thursday_shifts": thursday_counts_per_res.get(r, 0),
            # "tuesday_shifts": tuesday_counts_per_res.get(r, 0),
            # "Year": resident_levels.get(r),
            # "WEEKEND ROLE": ", ".join(sorted(weekend_roles_per_res.get(r, [])))
        }
        for role in roles:
            row[f"role_{role}_count"] = role_counts.get(r, {}).get(role, 0)
        scores_rows.append(row)

    scores_df = pd.DataFrame(scores_rows)

    return schedule_df, scores_df
