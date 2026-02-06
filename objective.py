import helper

# ------------------------- 
# Objective 
# -------------------------

def build_objective(
    model,
        
        # Score Balance
        non_nf_balance_penalties=None,
        non_nf_balance_weight=1,
        nf_balance_penalties=None,
        nf_balance_weight=1,

        # Spacing Balance
        spacing_nonnf_soft_penalties=None,
        non_nf_spacing_weight=1,
        spacing_nf_soft_penalties=None,
        nf_spacing_weight=1,
        spacing_ns_soft_penalties=None,
        ns_spacing_weight=1,

        # Days
        diverse_days_penalties=None,
        diverse_days_weight=1,
        wr_penalties=None,
        wr_pref_weight=1,
        nf_day_pref_penalties=None,
        nf_day_pref_weight=1,

        # Hard Days
        hard_nonhard_weight=1,
        hard_diversity_penalties=None,
        hard_diversity_weight=1,
        hard_max_penalties=None,
        hard_max_weight=1,
        thursday_penalties=None,
        thursday_weight=1,
        tuesday_penalties=None,
        tuesday_weight=1,
        weekend_penalties=None,
        weekend_weight=1,

        # Roles
        diverse_role_penalties=None,
        diverse_role_weight=1,
        role_pref_penalties=None,
        role_pref_weight=1
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

    if hard_diversity_penalties:
        terms.extend([hard_diversity_weight * p for p in hard_diversity_penalties])

    if hard_max_penalties:
        terms.extend([hard_max_weight * p for p in hard_max_penalties])

    if thursday_penalties:
        terms.extend([thursday_weight * p for p in thursday_penalties])

    if tuesday_penalties:
        terms.extend([tuesday_weight * p for p in tuesday_penalties])

    if weekend_penalties:
        terms.extend([weekend_weight * p for p in weekend_penalties])

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

    model.Minimize(sum(terms))

def minimize_and_fix(model, solver, penalties):
    if penalties is None:
        return  # ✅ THIS IS REQUIRED

    obj = penalties
    model.Minimize(obj)
    solver.Solve(model)

    best = solver.Value(penalties) 
    model.Add(obj == best)

def extract_schedule(
    solver=None, 
    assign=None, 
    days=None, 
    roles=None, 
    residents=None, 
    wr_residents=None,
    ns_residents=None, 
    night_counts=None, 
    wr_counts=None,
    score_vars=None, 
    max_shifts=None, 
    max_points=None, 
    nf_calendar_df=None,
    resident_levels=None,
    limited_shift_residents=None,
    schedule_dict=None
):
    """
    Extract solved schedule and compute per-resident statistics including:
    - Total shifts
    - Role counts
    - Weekday counts
    - Hard day counts (Tue, Thu, Weekend)
    - WR / NS / NF info
    - Spacing statistics

    Works with either:
    - solver (CpSolver)
    - schedule_dict {(d, role, r): 0/1} from hybrid/local search
    """

    from collections import defaultdict
    import pandas as pd

    # -----------------------------
    # Initialization
    # -----------------------------
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hard_days = {"Tue", "Thu", "Fri", "Sat"}

    schedule_dict = schedule_dict or {}

    schedule = []

    day_shift_counts = {r: 0 for r in residents}
    role_counts = {r: {role: 0 for role in roles} for r in residents}

    weekday_counts_per_res = {
        r: {wd: 0 for wd in weekday_order}
        for r in residents
    }

    hard_day_counts_per_res = {r: 0 for r in residents}

    # -----------------------------
    # Build schedule from solver or dict
    # -----------------------------
    for d in days:
        date_val = pd.to_datetime(d).date()
        weekday_str = date_val.strftime("%a")

        row = {"date": date_val, "day": weekday_str}

        for role in roles:
            row[role] = ""
            for r in residents:
                try:
                    # Determine if resident r is assigned
                    if solver is not None:
                        assigned = solver.Value(assign[(d, role, r)]) == 1
                    else:
                        assigned = schedule_dict.get((d, role, r), 0) == 1

                    if assigned:
                        row[role] = r
                        day_shift_counts[r] += 1
                        role_counts[r][role] += 1
                        weekday_counts_per_res[r][weekday_str] += 1

                        if weekday_str in hard_days:
                            hard_day_counts_per_res[r] += 1
                except Exception:
                    pass

        schedule.append(row)

    schedule_df = pd.DataFrame(schedule)

    # -----------------------------
    # Normalize NS residents
    # -----------------------------
    ns_names = set()
    if ns_residents is not None:
        if isinstance(ns_residents, pd.DataFrame) and "name" in ns_residents.columns:
            ns_names = set(ns_residents["name"])
        elif isinstance(ns_residents, dict):
            ns_names = set(ns_residents.keys())
        elif isinstance(ns_residents, list):
            ns_names = set(ns_residents)

    # -----------------------------
    # Merge NF calendar
    # -----------------------------
    if nf_calendar_df is not None and not nf_calendar_df.empty:
        schedule_df["date"] = pd.to_datetime(schedule_df["date"]).dt.date
        nf_calendar_df = nf_calendar_df.copy()
        nf_calendar_df["date"] = pd.to_datetime(nf_calendar_df["date"]).dt.date
        if "day" in nf_calendar_df.columns:
            nf_calendar_df = nf_calendar_df.drop(columns=["day"])
        schedule_df = schedule_df.merge(nf_calendar_df, on="date", how="left")

    # -----------------------------
    # WR logic
    # -----------------------------
    wr_counts = {r: 0 for r in residents}
    wr_column_map = defaultdict(list)

    if wr_residents is not None and not wr_residents.empty:
        wr_residents = wr_residents.copy()
        wr_residents["date"] = pd.to_datetime(wr_residents["date"]).dt.date
        wr_counts.update(
            wr_residents.groupby("name")["date"].nunique().to_dict()
        )
        for d, g in wr_residents.groupby("date"):
            wr_column_map[d] = g["name"].tolist()

    schedule_df["wr"] = schedule_df["date"].apply(
        lambda d: ", ".join(wr_column_map[d]) if d in wr_column_map else ""
    )

    # -----------------------------
    # Spacing statistics
    # -----------------------------
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
            wr_df=wr_residents,
            nf_calendar_df=nf_calendar_df,
            extra_preassigned=None,
            #schedule_dict=schedule_dict
        )

        gaps = [
            (assigned_dates[i] - assigned_dates[i - 1]).days
            for i in range(1, len(assigned_dates))
        ]

        per_resident_spacing[r] = {
            "num_gaps": len(gaps),
            "avg_gap": sum(gaps) / len(gaps) if gaps else None,
            "min_gap": min(gaps) if gaps else None,
            "max_gap": max(gaps) if gaps else None
        }

        all_gaps.extend(gaps)

    spacing_overall = {
        "num_gaps": len(all_gaps),
        "avg_gap": sum(all_gaps) / len(all_gaps) if all_gaps else None,
        "min_gap": min(all_gaps) if all_gaps else None,
        "max_gap": max(all_gaps) if all_gaps else None
    }

    # -----------------------------
    # Weekend roles
    # -----------------------------
    weekend_roles_per_res = {r: set() for r in residents}
    weekend_days = {
        pd.to_datetime(d).date()
        for d in days
        if pd.to_datetime(d).strftime("%a") in ["Fri", "Sat"]
    }

    for _, row in schedule_df.iterrows():
        d = row["date"]
        if d in weekend_days:
            for role in roles:
                r = row.get(role)
                if r in residents:
                    weekend_roles_per_res[r].add(role)

    ns_role_map = {}
    if ns_residents is not None:
        if isinstance(ns_residents, pd.DataFrame) and "name" in ns_residents.columns and "role" in ns_residents.columns:
            # Group all roles per resident as a comma-separated string
            ns_role_map = ns_residents.groupby("name")["role"].apply(lambda x: ", ".join(sorted(x))).to_dict()
        elif isinstance(ns_residents, dict):
            ns_role_map = ns_residents

    # -----------------------------
    # Build scores dataframe
    # -----------------------------
    scores_rows = []

    for r in residents:
        spacing = per_resident_spacing[r]

        effective_max_shifts = (
            limited_shift_residents[r]
            if limited_shift_residents and r in limited_shift_residents
            else (max_shifts.get(r) if isinstance(max_shifts, dict) else max_shifts)
        )

        row = {
            "Name": r,
            "Total Shifts": day_shift_counts[r],
            "Hard Day Shifts": hard_day_counts_per_res[r],
            "Score": solver.Value(score_vars[r]) if solver is not None and score_vars and r in score_vars else schedule_dict.get(("score", r), None),
            "Max Shifts": effective_max_shifts,
            "Max Points": max_points.get(r) if isinstance(max_points, dict) else max_points,
            "WR Count": wr_counts.get(r, 0),
            "NF Resident": "Yes" if night_counts.get(r, 0) > 2 else "No",
            "NS Resident": "Yes" if r in ns_names else "No",
            "Year": resident_levels.get(r),
            "WEEKEND ROLE": ", ".join(sorted(weekend_roles_per_res[r])),
            "spacing_avg": spacing["avg_gap"],
            "spacing_min": spacing["min_gap"],
            "spacing_max": spacing["max_gap"],
            "spacing_gaps_count": spacing["num_gaps"],
            "spacing_overall_avg": spacing_overall["avg_gap"],
            "spacing_overall_min": spacing_overall["min_gap"],
            "spacing_overall_max": spacing_overall["max_gap"],
            "spacing_overall_gaps_count": spacing_overall["num_gaps"],
            "NS Role": ns_role_map.get(r, "")
        }

        for wd in weekday_order:
            row[f"{wd}_shifts"] = weekday_counts_per_res[r][wd]

        for role in roles:
            row[f"role_{role}_count"] = role_counts[r][role]

        scores_rows.append(row)

    scores_df = pd.DataFrame(scores_rows)

    return schedule_df, scores_df
