import pandas as pd

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
        wr_residents = set(weekend_rounds_df["resident"].str.strip())
        if resident in wr_residents:
            expr += 2
    
    # NS bonus
    if ns_residents_df is not None and not ns_residents_df.empty:
        ns_residents = set(ns_residents_df["resident"].str.strip())
        if resident in ns_residents:
            expr += 2
            
    return expr

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
    - Ensure the specified resident is assigned to the 'EW Day' role on the given date.
    - Raise an error if 'EW Day' is not in the roles list.
    """
    
    # Ensure 'Date' column is in datetime format
    weekend_rounds_df['date'] = pd.to_datetime(weekend_rounds_df['date']).dt.normalize()
    
    # Iterate through each weekend round assignment
    for _, row in weekend_rounds_df.iterrows():
        resident = row['resident'].strip()
        date = row['date']
        
        # Enforce that the resident is assigned to 'EW Day' on this date
        if "ew day" in roles:
            model.Add(assign[(date, "ew day", resident)] == 1)
        else:
            # Defensive check: roles list must include 'EW Day'
            raise ValueError("role 'ew day' not found in roles list.")

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
    max_shifts, max_points, weekend_days, weekend_limits, 
    ns_residents_df=None
):
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
        model.Add(total_shifts >= 0)  # NF can have no day shifts so why min >= 1
        model.Add(total_points >= 0)  # NF residents must get at least 1 point
        
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
    off_days["date"] = pd.to_datetime(off_days["date"])
    
    for _, row in off_days.iterrows():
        r, d = row["resident"], row["date"]
        
        # Block this resident from all roles on this date
        for role in roles:
            if (d, role, r) in assign:
                model.Add(assign[(d, role, r)] == 0)

def add_on_days_constraint(model, assign, roles, on_days):
    """ 
    Add constraints to force residents to be scheduled on specific days. 
    """
    
    for _, row in on_days.iterrows():
        r, d = row["resident"], row["date"]
        
        # Resident must be assigned to exactly one role on this date
        model.Add(sum(assign[(d, role, r)] for role in roles) == 1)

