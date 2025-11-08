import pandas as pd
import helper

def build_nf_calendar(residents_df, start_date, nf_cols=None):
    """ 
    Build an NF calendar from residents_df and expand into NF1..NF{n_slots} columns.
    """
    
    if nf_cols is None:
        nf_cols = ["nf1", "nf2", "nf3"]
    
    # --- Step 1: Collect NF assignments per date ---
    calendar = {}
    for _, row in residents_df.iterrows():
        for d in helper.expand_dates(row["nf"], base_year=start_date.year):
            # Ensure d is a pd.Timestamp
            d_ts = pd.Timestamp(d)
            calendar.setdefault(d_ts, []).append(row["name"])
    
    # --- Step 2: Determine full date range ---
    start_date = pd.Timestamp(start_date)
    last_date = max(calendar.keys()) if calendar else start_date
    last_date = pd.Timestamp(last_date)
    all_dates = [start_date + pd.Timedelta(days=i) for i in range((last_date - start_date).days + 1)]
    
    # --- Step 3: Build base NF calendar DataFrame ---
    base_calendar = pd.DataFrame({
        "day": [d.strftime("%a") for d in all_dates],
        "date": all_dates,  # keep as Timestamp
        "nf": [", ".join(calendar.get(d, [])) for d in all_dates]
    })
    
    # --- Step 4: Expand into NF1..NF{n_slots} with rotation ---
    calendar_data = []
    for day_idx, row in base_calendar.iterrows():
        names = row["nf"].split(", ") if row["nf"] else []
        rotated_names = helper.rotate_list(names, day_idx)
        row_dict = {
            "day": row["day"],
            "date": row["date"]
        }
        for idx, col_name in enumerate(nf_cols):
            row_dict[col_name] = rotated_names[idx] if idx < len(rotated_names) else ""
        calendar_data.append(row_dict)
    
    return pd.DataFrame(calendar_data)

def add_nf_day_preferences_seniors(model, assign, roles, days, nf_residents, weight=3):
    """
    - Hard constraint: NF residents cannot be assigned to ER-1 role.
    - Soft penalty: discourage NF residents from being assigned to day shifts
      on non-preferred weekdays (anything other than Tue/Thu).
    """
    penalties = []
    preferred_days = {1, 3}  # Tuesday=1, Thursday=3 (Python weekday: Monday=0)

    for r in nf_residents:
        for d in days:
            weekday = pd.to_datetime(d).weekday()
            for role in roles:
                # HARD CONSTRAINT: no ER-1 shifts for NF residents
                if "er-1" in role.lower():
                    model.Add(assign[(d, role, r)] == 0)

                # SOFT PENALTY: non-Tue/Thu day shifts
                elif weekday not in preferred_days and "day" in role.lower():
                    penalty_var = model.NewIntVar(0, 1, f"nf_day_penalty_{d}_{role}_{r}")
                    model.Add(penalty_var == assign[(d, role, r)])
                    penalties.append(penalty_var)

    return penalties

def add_nf_day_preferences_juniors(model, assign, roles, days, nf_residents, weight=3):
    """
    - Hard constraints, NF residents should avoid Tuesday and Thursday
    """
    
    penalties = []
    avoid_days = {1, 3}  # Tuesday=1, Thursday=3 (Python weekday: Monday=0)

    for r in nf_residents:
        for d in days:
            weekday = pd.to_datetime(d).weekday()
            for role in roles:
                # HARD CONSTRAINT: no Tue/Thu for NF
                if weekday in avoid_days and "day" in role.lower():
                    penalty_var = model.NewIntVar(0, 1, f"nf_day_penalty_{d}_{role}_{r}")
                    model.Add(assign[(d, role, r)] == 0)
                    penalties.append(penalty_var)

    return penalties