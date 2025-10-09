import pandas as pd
from ortools.sat.python import cp_model
import random
from datetime import datetime, timedelta
import re

# -------------------------
# Helpers
# -------------------------
def standardize_dates(df, date_columns):
    """
    Convert columns to pd.Timestamp with datetime64[ns] dtype.
    Works on one or multiple columns.
    """
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.normalize()
    return df

def resident_score_expr(assign, days, roles, resident, weekend_days, weekend_rounds_df=None, ns_residents_df=None):
    """
    Build the linear expression for a resident's score:
      - 1 point per weekday shift
      - 2 points per weekend shift
      - +2 bonus if assigned to weekend rounds
    """
    # Base score: weighted by weekday/weekend
    expr = sum(
        assign[(d, role, resident)] * (2 if d in weekend_days else 1)
        for d in days for role in roles
    )

    # Weekend round bonus 
    if weekend_rounds_df is not None:
        wr_residents = set(weekend_rounds_df["Resident"].str.strip())
        if resident in wr_residents:
            expr += 2

    # NS bonus
    if ns_residents_df is not None and not ns_residents_df.empty:
        ns_residents = set(ns_residents_df["Resident"].str.strip())
        if resident in ns_residents:
            expr += 2

    return expr
    
def extract_shift_columns(): 
    # Extract the columns needed
    night_cols = ["ER-1 Night", "ER-2 Night", "EW Night"] 
    day_cols = ["ER-1 Day", "ER-2 Day", "EW Day"] 
    return night_cols, day_cols 

# Helper: expand date ranges from NF column
def expand_dates(date_range_str, year=2025):
    if not date_range_str or date_range_str.strip().lower() == "no":
        return []

    # Normalize dashes and spacing
    clean_str = date_range_str.replace("–", "-").replace("—", "-").strip()
    clean_str = re.sub(r"\s*-\s*", "-", clean_str)  # normalize spacing around dash

    # If single date (no dash at all)
    if "-" not in clean_str:
        return [datetime.strptime(clean_str + f" {year}", "%d %b %Y").date()]

    # Split into start and end
    start_str, end_str = clean_str.split("-", 1)
    start_str = start_str.strip()
    end_str = end_str.strip()

    # Parse start
    try:
        start_date = datetime.strptime(start_str + f" {year}", "%d %b %Y").date()
    except ValueError:
        end_parts = end_str.split()
        if len(end_parts) == 2:
            start_date = datetime.strptime(start_str + " " + end_parts[1] + f" {year}", "%d %b %Y").date()
        else:
            raise

    # Parse end
    try:
        end_date = datetime.strptime(end_str + f" {year}", "%d %b %Y").date()
    except ValueError:
        start_parts = start_str.split()
        if len(start_parts) == 2:
            end_date = datetime.strptime(end_str + " " + start_parts[1] + f" {year}", "%d %b %Y").date()
        else:
            raise

    # Ensure correct order
    if end_date < start_date:
        if end_date.month < start_date.month:
            end_date = end_date.replace(year=year, month=end_date.month)
        if end_date < start_date:
            end_date = end_date.replace(year=year+1)

    # Expand full range as date objects (no time)
    delta = (end_date - start_date).days
    return [start_date + timedelta(days=i) for i in range(delta + 1)]

def build_nf_calendar(seniors_df, start_date, nf_cols=None):
    """
    Build an NF calendar from seniors_df and expand into NF1..NF{n_slots} columns.
    """
    if nf_cols is None:
        nf_cols = ["NF1", "NF2", "NF3"]

    # --- Step 1: Collect NF assignments per date ---
    calendar = {}
    for _, row in seniors_df.iterrows():
        for d in expand_dates(row["NF"], year=2025):
            # Ensure d is a pd.Timestamp
            d_ts = pd.Timestamp(d)
            calendar.setdefault(d_ts, []).append(row["Name"])

    # --- Step 2: Determine full date range ---
    start_date = pd.Timestamp(start_date)
    last_date = max(calendar.keys()) if calendar else start_date
    last_date = pd.Timestamp(last_date)  # ensure Timestamp

    all_dates = [start_date + pd.Timedelta(days=i)
                 for i in range((last_date - start_date).days + 1)]

    # --- Step 3: Build base NF calendar DataFrame ---
    base_calendar = pd.DataFrame({
        "Day": [d.strftime("%a") for d in all_dates],
        "Date": all_dates,  # keep as Timestamp
        "NF": [", ".join(calendar.get(d, [])) for d in all_dates]
    })

    # --- Step 4: Expand into NF1..NF{n_slots} with rotation ---
    calendar_data = []
    for day_idx, row in base_calendar.iterrows():
        names = row["NF"].split(", ") if row["NF"] else []
        rotated_names = names[day_idx % len(names):] + names[:day_idx % len(names)] if names else []

        row_dict = {
            "Day": row["Day"],
            "Date": row["Date"]
        }

        for idx, col_name in enumerate(nf_cols):
            row_dict[col_name] = rotated_names[idx] if idx < len(rotated_names) else ""

        calendar_data.append(row_dict)

    return pd.DataFrame(calendar_data)

def fill_empty_nf_cells(
    non_nf_residents,
    nf_calendar_df,
    wr_residents,
    blackout_df=None,
    nf_cols=None
):
    """
    Fill empty NF cells in the calendar with unused residents,
    ensuring they are not on vacation or blackout on that date.
    """
    if nf_cols is None:
        nf_cols = ["NF1", "NF2", "NF3"]

    # --- Step 1: Build resident pool ---
    available_residents = [r for r in non_nf_residents if r not in wr_residents]

    # Track already used residents
    #used_residents = set()
    #for col in nf_cols:
    #    used_residents.update([name for name in nf_calendar_df[col] if name])

    # Available pool
    #available_residents = [r for r in all_residents if r not in used_residents]
    random.shuffle(available_residents)

    # --- Step 2: Build lookup sets for blackout ---
    blackout_lookup = set()
    if blackout_df is not None and not blackout_df.empty:
        for _, row in blackout_df.iterrows():
            blackout_lookup.add((row["Resident"], pd.to_datetime(row["Date"])))

    # --- Step 3: Fill missing NF slots ---
    filled_records = []

    for col in nf_cols:
        for idx, val in nf_calendar_df[col].items():
            if val == "" and available_residents:
                date = pd.to_datetime(nf_calendar_df.at[idx, "Date"])
                assigned = None
                attempts = 0
                while available_residents and assigned is None and attempts < 100:
                    candidate = available_residents.pop(0)
                    if (candidate, date) not in blackout_lookup:
                        assigned = candidate
                        nf_calendar_df.at[idx, col] = candidate
                        filled_records.append({"Date": date, "Resident": candidate})
                    else:
                        # Put them back at the end of the pool
                        available_residents.append(candidate)
                    attempts += 1

    # --- Step 4: Build NS residents DataFrame ---
    ns_residents = pd.DataFrame(filled_records)
    ns_residents = standardize_dates(ns_residents, ["Date"])

    return nf_calendar_df, ns_residents

def prepare_data(residents_df, start_date, num_weeks=4):
    """
    Prepare scheduling inputs from the residents DataFrame.
    """    
    # Residents list
    residents = residents_df['Name'].tolist()
    
    
    # Days in scheduling period
    start_date = pd.to_datetime(start_date)
    days = pd.date_range(start_date, start_date + pd.Timedelta(weeks=num_weeks) - pd.Timedelta(days=1))
    weekend_days = set(d for d in days if d.strftime('%a') in ['Fri', 'Sat'])
    
    # Roles
    nf_roles, roles = extract_shift_columns()

    # NF Calendar
    nf_calendar_df = build_nf_calendar(residents_df, start_date, nf_cols=nf_roles)
    nf_calendar_df = standardize_dates(nf_calendar_df, ["Date"])
    
    # Initialize blackouts and counters
    nf_blackout = {r: set() for r in residents}
    wr_blackout = {r: set() for r in residents}
    vacation_blackout = {r: set() for r in residents}
    night_counts = {r: 0 for r in residents}
    
    vacation_records = []
    weekend_rounds_records = []
    
    for _, row in residents_df.iterrows():
        r = row['Name']
        
        # NF blackout (night shifts)
        if row['NF'] != 'No':
            for rng in row['NF'].split("\n"):
                for d in expand_dates(rng, year=2025):
                    night_counts[r] += 1
                    nf_blackout[r].update(pd.date_range(d - pd.Timedelta(days=5), d + pd.Timedelta(days=5)))
        
        # Vacation blackout
        if row['Leave'] != 'No':
            for rng in row['Leave'].split("\n"):
                dates = expand_dates(rng, year=2025)
                if not dates:
                    continue
        
                # If the vacation block is exactly 5 days long
                if len(dates) == 5:
                    for d in dates:
                        extended = [
                            d - pd.Timedelta(days=2),
                            d - pd.Timedelta(days=1),
                            d,
                            d + pd.Timedelta(days=1),
                            d + pd.Timedelta(days=2)
                        ]
                        vacation_blackout[r].update(extended)
                        vacation_records.append({"Resident": r, "Date": d})
                else:
                    # Only blackout the actual vacation days
                    vacation_blackout[r].update(dates)
                    for d in dates:
                        vacation_records.append({"Resident": r, "Date": d})
        
        # Weekend rounds
        if row['Weekend round'] != 'No':
            for rng in row['Weekend round'].split("\n"):
                dates = expand_dates(rng, year=2025)
                if not dates:
                    continue

                # Take only the first date in the range
                wr_date_ts = pd.Timestamp(dates[0]).normalize()

                # Build blackout range (±2 days) excluding the WR date
                blackout_range = pd.date_range(
                    wr_date_ts - pd.Timedelta(days=2),
                    wr_date_ts + pd.Timedelta(days=2),
                    freq="D"
                ).difference(pd.DatetimeIndex([wr_date_ts]))

                # Update resident's WR blackout set
                wr_blackout[r].update(blackout_range)

                # --- Remove WR date from all blackouts so it can be assigned ---
                for b in [nf_blackout, vacation_blackout, wr_blackout]:
                    b[r].discard(wr_date_ts)

                # Record the WR assignment itself
                weekend_rounds_records.append({
                    "Resident": r,
                    "Date": wr_date_ts
                })
    
    vacation_df = pd.DataFrame(vacation_records)
    vacation_df = standardize_dates(vacation_df, ["Date"])

    weekend_rounds_df = pd.DataFrame(weekend_rounds_records)
    weekend_rounds_df = standardize_dates(weekend_rounds_df, ["Date"])

    wr_residents = set(weekend_rounds_df["Resident"]) if not weekend_rounds_df.empty else set()

    # Split Residents to NF or Not
    nf_residents = {r for r, c in night_counts.items() if c > 2}
    non_nf_residents = set(residents) - nf_residents

    # Build combined blackout dict 
    combined_blackout = {
        r: nf_blackout.get(r, set()) | wr_blackout.get(r, set()) | vacation_blackout.get(r, set())
        for r in set(nf_blackout) | set(wr_blackout) | set(vacation_blackout)
    }
    
    blackout_records = [{"Resident": r, "Date": d} for r, dates in combined_blackout.items() for d in dates]
    combined_blackout_df = pd.DataFrame(blackout_records) if blackout_records else pd.DataFrame(columns=["Resident", "Date"])
    combined_blackout_df = standardize_dates(combined_blackout_df, ["Date"])

    # Fill NF table with NS
    filled_nf_calendar_df, ns_residents = fill_empty_nf_cells(
        non_nf_residents,
        nf_calendar_df,
        wr_residents=wr_residents,
        blackout_df=combined_blackout_df,
        nf_cols=nf_roles
    )

    # NS blackout (1 day before and after)
    ns_blackout = {r: set() for r in residents}
    if ns_residents is not None and not ns_residents.empty:
        for _, row in ns_residents.iterrows():
            r = row["Resident"]
            ns_date = pd.to_datetime(row["Date"])
            ns_blackout[r].update([ns_date - pd.Timedelta(days=1), ns_date + pd.Timedelta(days=1)])
        
    return (
        residents,
        roles,
        days,
        weekend_days,
        nf_residents,
        non_nf_residents,
        wr_residents,
        nf_blackout,
        wr_blackout,
        ns_blackout,
        vacation_blackout,
        night_counts,
        vacation_df,
        weekend_rounds_df,
        ns_residents,
        filled_nf_calendar_df
    )

def calculate_max_limits(residents, nf_residents, wr_residents, night_counts):
    """
    Calculate maximum shifts, points, and weekend limits for each resident.
    """
    max_shifts, max_points, weekend_limits = {}, {}, {}

    for resident in residents:
        if resident in nf_residents:
            shifts, points, weekend = 1, 1, 0
        elif resident in wr_residents:
            shifts, points, weekend = 2, 6, 1
        else: 
            shifts, points, weekend = 5, 6, 2
            
        # Define Max Shifts, Max Points, Max Weekends
        max_shifts[resident] = shifts
        max_points[resident] = points
        weekend_limits[resident] = weekend

    return max_shifts, max_points, weekend_limits

# ------------------------- 
# Hard Constraints 
# ------------------------- 
def add_basic_constraints(model, assign, days, roles, residents): 
    """
    Core constraints:
      1. Every role must be filled each day.
      2. No resident can be assigned to more than one role per day.
    """
    # Each role is filled by exactly one resident per day
    for d in days: 
        for role in roles: 
            model.Add(sum(assign[(d, role, r)] for r in residents) == 1) 

    # Each resident works at most one role per day
    for d in days: 
        for r in residents: 
            model.Add(sum(assign[(d, role, r)] for role in roles) <= 1) 

def add_weekend_rounds_constraint(model, assign, roles, weekend_rounds_df): 
    """
    Add constraints to enforce weekend round assignments.

    For each row in weekend_rounds_df:
      - Ensure the specified resident is assigned to the 'EW Day' role
        on the given date.
      - Raise an error if 'EW Day' is not in the roles list.
    """
    # Ensure 'Date' column is in datetime format
    weekend_rounds_df['Date'] = pd.to_datetime(weekend_rounds_df['Date']) 

    # Iterate through each weekend round assignment
    for _, row in weekend_rounds_df.iterrows(): 
        resident = row['Resident'].strip() 
        date = row['Date'] 

        # Enforce that the resident is assigned to 'EW Day' on this date
        if "EW Day" in roles: 
            model.Add(assign[(date, "EW Day", resident)] == 1) 
        else: 
            # Defensive check: roles list must include 'EW Day'
            raise ValueError("Role 'EW Day' not found in roles list.")

def add_blackout_constraints(model, assign, roles, blackout_dict):
    """
    Prevent residents from being scheduled on blackout dates.

    Args:
        blackout_dict (dict): {resident: set of blackout dates}
    """
    for resident, blackout_days in blackout_dict.items():
        for day in blackout_days:
            for role in roles:
                if (day, role, resident) in assign:
                    model.Add(assign[(day, role, resident)] == 0)

def add_shift_cap_constraints(
    model, assign, days, roles, residents,
    max_shifts, max_points, weekend_days, weekend_limits, ns_residents_df=None):
    """
    Add constraints to enforce per-resident limits:
      1. Minimum and maximum total shifts
      2. Minimum and maximum weighted points (weekends count double)
      3. Maximum weekend shifts
    """

    for r in residents:
        total_shifts = sum(assign[(d, role, r)] for d in days for role in roles)
        total_points = resident_score_expr(assign, days, roles, r, weekend_days, ns_residents_df)
        weekend_shifts = sum(assign[(d, role, r)] for d in weekend_days for role in roles)

        # --- Minimums ---
        model.Add(total_shifts >= 1)   # NF residents must get at least 1 shift
        model.Add(total_points >= 1)   # NF residents must get at least 1 point

        # --- Maximums ---
        model.Add(total_shifts <= max_shifts[r])
        model.Add(total_points <= max_points[r])
        model.Add(weekend_shifts <= weekend_limits[r])

def add_no_consecutive_groups_constraint(model, assign, groups, roles, residents, label="group"):
    """
    Generic constraint: prevent residents from being scheduled in consecutive groups.
    """
    for r in residents:
        for i in range(len(groups) - 1):
            group1_days = groups[i]
            group2_days = groups[i + 1]

            # BoolVar: did the resident work in group i?
            g1_worked = model.NewBoolVar(f"{label}_{i}_worked_{r}")
            model.Add(sum(assign[(d, role, r)] for d in group1_days for role in roles) >= 1).OnlyEnforceIf(g1_worked)
            model.Add(sum(assign[(d, role, r)] for d in group1_days for role in roles) == 0).OnlyEnforceIf(g1_worked.Not())

            # BoolVar: did the resident work in group i+1?
            g2_worked = model.NewBoolVar(f"{label}_{i+1}_worked_{r}")
            model.Add(sum(assign[(d, role, r)] for d in group2_days for role in roles) >= 1).OnlyEnforceIf(g2_worked)
            model.Add(sum(assign[(d, role, r)] for d in group2_days for role in roles) == 0).OnlyEnforceIf(g2_worked.Not())

            # Constraint: cannot work both consecutive groups
            model.AddBoolOr([g1_worked.Not(), g2_worked.Not()])

# ------------------------- 
# Soft Constraints 
# ------------------------- 
def diverse_rotation_penalty(model, assign, days, roles, residents):
    """
    Soft constraint version of diverse rotation:
      - If a resident works 2+ shifts of one role but 0 of another role,
        add a penalty.
    Returns:
        penalties (list of IntVar): penalty variables to include in objective.
    """
    penalties = []

    for r in residents:
        # Count how many times this resident is assigned to each role
        role_counts = {}
        for role in roles:
            role_counts[role] = model.NewIntVar(0, len(days), f"{r}_{role}_count")
            model.Add(role_counts[role] == sum(assign[(d, role, r)] for d in days))

        # For each pair of roles, add a penalty if resident overuses one and ignores the other
        for role in roles:
            for other_role in roles:
                if role == other_role:
                    continue

                # Boolean: did this resident do 2+ of 'role'?
                role_used_twice = model.NewBoolVar(f"{r}_{role}_used_twice")
                model.Add(role_counts[role] >= 2).OnlyEnforceIf(role_used_twice)
                model.Add(role_counts[role] < 2).OnlyEnforceIf(role_used_twice.Not())

                # Boolean: did this resident do at least 1 of 'other_role'?
                other_role_used = model.NewBoolVar(f"{r}_{other_role}_used")
                model.Add(role_counts[other_role] >= 1).OnlyEnforceIf(other_role_used)
                model.Add(role_counts[other_role] == 0).OnlyEnforceIf(other_role_used.Not())

                # Penalty: violation if role_used_twice AND NOT other_role_used
                violation = model.NewBoolVar(f"{r}_{role}_vs_{other_role}_violation")
                model.AddBoolAnd([role_used_twice, other_role_used.Not()]).OnlyEnforceIf(violation)
                model.AddBoolOr([role_used_twice.Not(), other_role_used]).OnlyEnforceIf(violation.Not())

                penalties.append(violation)

    return penalties

def add_limited_shifts_constraint(model, assign, roles, days, limit_dict): 
    """
    Add constraints to enforce custom shift limits for specific residents.
    """
    if not limit_dict: 
        return 

    for r, max_shifts in limit_dict.items(): 
        # Ensure resident r is not assigned to more than max_shifts total
        model.Add(sum(assign[(d, role, r)] for d in days for role in roles) <= max_shifts)

def add_off_days_constraint(model, assign, roles, off_days): 
    """
    Add constraints to block residents from being scheduled on specific days.
    """
    # Ensure 'Date' column is datetime
    off_days["Date"] = pd.to_datetime(off_days["Date"]) 

    for _, row in off_days.iterrows(): 
        r, d = row["Resident"], row["Date"] 
        # Block this resident from all roles on this date
        for role in roles: 
            if (d, role, r) in assign: 
                model.Add(assign[(d, role, r)] == 0) 

def add_on_days_constraint(model, assign, roles, on_days): 
    """
    Add constraints to force residents to be scheduled on specific days.
    """
    for _, row in on_days.iterrows(): 
        r, d = row["Resident"], row["Date"] 
        # Resident must be assigned to exactly one role on this date
        model.Add(sum(assign[(d, role, r)] for role in roles) == 1)

# ------------------------- 
# Fairness Scoring 
# ------------------------- 
def add_score_balance_constraint(
    model,
    assign,
    days,
    roles,
    residents,
    weekend_days,
    nf_residents,
    weekend_rounds_df,
    ns_residents_df,
    night_counts=None,
    vacation_df=None,
    limited_shift_residents=None,
    off_days=None
):
    """
    Add fairness/balance constraints across residents.

    Rules:
      - Compute a "score" for each resident:
          * 1 point per weekday shift
          * 2 points per weekend shift
          * +2 bonus if assigned to weekend rounds
      - Exclude constrained residents (vacation, limited shifts, off-days) from hard balancing.
      - Hard balance: non-constrained residents should have similar scores.
      - NF residents are balanced against each other as one group.
      - Soft balance: constrained residents are softly compared against hard residents.
    """

    # Weekend round bonus residents
    wr_residents = set(weekend_rounds_df["Resident"].str.strip())

    # Score variables
    score_vars = {r: model.NewIntVar(0, 500, f"score_{r}") for r in residents}
    for r in residents:
        model.Add(
            score_vars[r] == resident_score_expr(
                assign, days, roles, r, weekend_days, weekend_rounds_df, ns_residents_df
            )
        )

    # Exclusions
    excluded = set()
    if vacation_df is not None:
        excluded.update(vacation_df["Resident"].unique())
    if limited_shift_residents is not None:
        excluded.update(limited_shift_residents.keys())
    if off_days is not None:
        excluded.update(off_days["Resident"].unique())

    balance_penalties = []

    # -------------------------
    # Hard balance: unconstrained residents
    # -------------------------
    hard_residents = [r for r in residents if r not in excluded]

    # Split into NF and non-NF
    nf_hard = [r for r in hard_residents if r in nf_residents]
    non_nf_hard = [r for r in hard_residents if r not in nf_residents]

    # Balance non-NF hard residents pairwise
    for i in range(len(non_nf_hard)):
        for j in range(i + 1, len(non_nf_hard)):
            r1, r2 = non_nf_hard[i], non_nf_hard[j]
            diff = model.NewIntVar(-500, 500, f"score_diff_{r1}_{r2}")
            model.Add(diff == score_vars[r1] - score_vars[r2])
            abs_diff = model.NewIntVar(0, 500, f"abs_score_diff_{r1}_{r2}")
            model.AddAbsEquality(abs_diff, diff)
            balance_penalties.append(abs_diff)

    # Balance NF hard residents pairwise
    for i in range(len(nf_hard)):
        for j in range(i + 1, len(nf_hard)):
            r1, r2 = nf_hard[i], nf_hard[j]
            diff = model.NewIntVar(-500, 500, f"nf_score_diff_{r1}_{r2}")
            model.Add(diff == score_vars[r1] - score_vars[r2])
            abs_diff = model.NewIntVar(0, 500, f"nf_abs_score_diff_{r1}_{r2}")
            model.AddAbsEquality(abs_diff, diff)
            balance_penalties.append(abs_diff)

    # -------------------------
    # Soft balance: constrained residents
    # -------------------------
    soft_residents = list(excluded)
    for r in soft_residents:
        for peer in hard_residents:
            diff = model.NewIntVar(-500, 500, f"soft_diff_{r}_{peer}")
            model.Add(diff == score_vars[r] - score_vars[peer])
            abs_diff = model.NewIntVar(0, 500, f"soft_abs_diff_{r}_{peer}")
            model.AddAbsEquality(abs_diff, diff)
            balance_penalties.append(abs_diff)

    return balance_penalties, score_vars

def tuesday_thursday_fairness_penalty(model, assign, days, roles, residents):
    """
    Soft fairness constraint:
    Penalize residents who are assigned to too many Tuesday/Thursday shifts.
    - 0 or 1 Tue/Thu shifts → no penalty
    - 2+ Tue/Thu shifts → penalty = (count - 1)
    Returns a list of penalty variables to be included in the objective.
    """

    penalty_vars = []

    # Identify all Tuesday and Thursday dates in the schedule
    hard_days = [d for d in days if d.strftime('%a') in ['Tue', 'Thu']]

    for r in residents:
        # -------------------------
        # Count how many Tue/Thu shifts this resident has
        # -------------------------
        hard_day_count = model.NewIntVar(0, len(hard_days), f"{r}_hard_day_count")
        model.Add(
            hard_day_count == sum(assign[(d, role, r)] for d in hard_days for role in roles)
        )

        # -------------------------
        # Boolean flag: does this resident have 2 or more Tue/Thu shifts?
        # -------------------------
        has_excess_hard_day = model.NewBoolVar(f"{r}_has_excess_hard_day")
        model.Add(hard_day_count >= 2).OnlyEnforceIf(has_excess_hard_day)
        model.Add(hard_day_count < 2).OnlyEnforceIf(has_excess_hard_day.Not())

        # -------------------------
        # Penalty variable:
        # - If resident has 0 or 1 Tue/Thu shifts → penalty = 0
        # - If resident has 2+ → penalty = hard_day_count - 1
        # -------------------------
        excess_hard_day = model.NewIntVar(0, len(hard_days), f"{r}_excess_hard_day")
        model.Add(excess_hard_day == hard_day_count - 1).OnlyEnforceIf(has_excess_hard_day)
        model.Add(excess_hard_day == 0).OnlyEnforceIf(has_excess_hard_day.Not())

        # Collect penalty variable for use in the objective
        penalty_vars.append(excess_hard_day)

    return penalty_vars

# ------------------------- 
# Objective
# ------------------------- 
def build_objective(
    model,
    score_vars,
    balance_penalties=None,
    hard_day_penalties=None,
    diverse_penalties=None,
    balance_weight=10,
    hard_day_weight=5,
    diverse_weight=5
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

    # Final objective
    #model.Minimize(0)
    model.Maximize(sum(terms))

# ------------------------- 
# Output (Detailed schedule + scores_df) 
# ------------------------- 
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
        scores_df:   DataFrame with per-resident totals, score, and WR info.
    """

    schedule = []
    # Track how many shifts each resident worked
    day_shift_counts = {r: 0 for r in residents}

    # Build the daily schedule from solver
    for d in days:
        row = {"Date": d.date(), "Day": d.strftime('%a')}
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
        schedule_df["Date"] = pd.to_datetime(schedule_df["Date"]).dt.normalize()
        nf_calendar_df = nf_calendar_df.copy()
        if nf_calendar_df is not None and not nf_calendar_df.empty:
            nf_calendar_df = standardize_dates(nf_calendar_df, ["Date"])

        # Drop duplicate Day column from NF calendar if present
        if "Day" in nf_calendar_df.columns:
            nf_calendar_df = nf_calendar_df.drop(columns=["Day"])

        # Merge side by side
        schedule_df = schedule_df.merge(nf_calendar_df, on="Date", how="left")

    # Add WR (Weekend Round) column if EW Day exists
    if 'Day' in schedule_df.columns and 'EW Day' in schedule_df.columns:
        schedule_df['WR'] = schedule_df.apply(
            lambda row: row['EW Day'] if row['Day'] == 'Fri' else '', axis=1
        )

    # Build per-resident summary (scores_df)
    scores_df = pd.DataFrame([{
        "Resident": r,
        "Total Shifts": day_shift_counts[r],
        "Score": solver.Value(score_vars[r]),
        "Max Shifts": max_shifts[r],
        "Max Points": max_points[r],
        "WR Resident": r in wr_residents,   # Boolean flag
        "NF Resident": night_counts[r] > 2,
        "NS Resident": r in set(ns_residents["Resident"])
    } for r in residents])

    return schedule_df, scores_df

def schedule_with_ortools_full_modular(
    seniors_df,         
    start_date,
    num_weeks=4,
    limited_shift_residents=None,
    off_days=None,
    on_days=None
):
    """
    Build and solve a resident scheduling problem using OR-Tools CP-SAT.

    Returns:
        schedule_df (pd.DataFrame): Final schedule with assignments.
        scores_df (pd.DataFrame): Score breakdown for fairness and constraints.
    """

    # -----------------------------------------------------
    # 1. Prepare data and limits
    # -----------------------------------------------------
    seniors_df.columns = seniors_df.columns.str.strip()
    (
        residents, roles, days, weekend_days,
        nf_residents, non_nf_residents, wr_residents,
        nf_blackout, wr_blackout, ns_blackout, vacation_blackout,
        night_counts, vacation_df,
        weekend_rounds_df, ns_residents, nf_calendar_df
    ) = prepare_data(seniors_df, start_date, num_weeks=num_weeks)
    
    # Per-resident caps
    max_shifts, max_points, weekend_limits = calculate_max_limits(
        residents, nf_residents, wr_residents, night_counts
    )

    # Shuffle & sort residents by night count (adds randomness for fairness)
    residents = sorted(
        residents,
        key=lambda r: (night_counts.get(r, 0), random.random())
    )

    # -----------------------------------------------------
    # 2. Define model and decision variables
    # -----------------------------------------------------
    model = cp_model.CpModel()

    # Binary variable: assign[d, role, r] = 1 if resident r works role on day d
    assign = {
        (d, role, r): model.NewBoolVar(f"assign_{d.date()}_{role}_{r}")
        for d in days for role in roles for r in residents
    }

    # -----------------------------------------------------
    # 3. Hard constraints (must always hold)
    # -----------------------------------------------------
    add_basic_constraints(model, assign, days, roles, residents)

    # Weekend rounds coverage and blackout rules
    add_weekend_rounds_constraint(model, assign, roles, weekend_rounds_df)
    add_blackout_constraints(model, assign, roles, nf_blackout)  # NF blackout
    add_blackout_constraints(model, assign, roles, wr_blackout)  # WR blackout
    add_blackout_constraints(model, assign, roles, ns_blackout)  # NS blackout
    add_blackout_constraints(model, assign, roles, vacation_blackout) # Vacation blackout
    
    # Caps: total shifts, points, weekend limits (refactor internally to use resident_score_expr)
    add_shift_cap_constraints(
        model, assign, days, roles, residents, max_shifts, 
        max_points, weekend_days, weekend_limits, ns_residents_df=ns_residents
    )
    
    # -----------------------------------------------------
    # 3b. No consecutive groups (days + Fri/Sat weekends)
    # -----------------------------------------------------
    # Each day is its own group (no consecutive days)
    day_groups = [[d] for d in days]
    add_no_consecutive_groups_constraint(
        model, assign, day_groups, roles, residents, label="day"
    )

    # Build weekend groups as explicit Friday–Saturday pairs
    weekend_groups = []
    days_sorted = sorted(days)
    for i in range(len(days_sorted) - 1):
        d1, d2 = days_sorted[i], days_sorted[i + 1]
        if d1.weekday() == 4 and d2.weekday() == 5:  # 4=Fri, 5=Sat
            weekend_groups.append([d1, d2])

    add_no_consecutive_groups_constraint(
        model, assign, weekend_groups, roles, residents, label="weekend"
    )

    # -----------------------------------------------------
    # 4. Soft constraints (preferences, optional rules)
    # -----------------------------------------------------
    if limited_shift_residents is not None:
        add_limited_shifts_constraint(model, assign, days, roles, limited_shift_residents)
    if off_days is not None:
        add_off_days_constraint(model, assign, roles, off_days)
    if on_days is not None:
        add_on_days_constraint(model, assign, roles, on_days)

    # -----------------------------------------------------
    # 5. Fairness scoring + objective
    # -----------------------------------------------------
    balance_penalties, score_vars = add_score_balance_constraint(
        model, assign, days, roles, residents,
        weekend_days, nf_residents, weekend_rounds_df, ns_residents,
        night_counts, vacation_df, limited_shift_residents, off_days
    )

    # Tue/Thu fairness penalty (soft)
    hard_day_penalties = tuesday_thursday_fairness_penalty(model, assign, days, roles, residents)

    diverse_penalties = diverse_rotation_penalty(model, assign, days, roles, residents)

    # Build objective
    build_objective(
        model,
        score_vars=score_vars,
        balance_penalties=balance_penalties,
        hard_day_penalties=hard_day_penalties,
        diverse_penalties=diverse_penalties,
        balance_weight=10,
        hard_day_weight=5,
        diverse_weight=10   
    )

    # -----------------------------------------------------
    # 6. Solve the model
    # -----------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 200
    solver.parameters.random_seed = 42
    solver.parameters.num_search_workers = 8
    solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH

    status = solver.Solve(model)
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise RuntimeError("No feasible solution found")

    # -----------------------------------------------------
    # 7. Extract results
    # -----------------------------------------------------
    return extract_schedule(
        solver, assign, days, roles, residents,
        wr_residents, ns_residents, night_counts, score_vars, max_shifts, max_points, nf_calendar_df
    )