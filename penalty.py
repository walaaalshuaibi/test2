import constraints
import pandas as pd

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
    
    # Score variables
    score_vars = {r: model.NewIntVar(0, 500, f"score_{r}") for r in residents}
    
    for r in residents:
        model.Add(score_vars[r] == constraints.resident_score_expr(
            assign, days, roles, r, weekend_days, weekend_rounds_df, ns_residents_df
        ))
    
    # Exclusions
    excluded = set()
    if vacation_df is not None:
        excluded.update(vacation_df["resident"].unique())
    if limited_shift_residents is not None:
        excluded.update(limited_shift_residents.keys())
    if off_days is not None:
        excluded.update(off_days["resident"].unique())
    
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
    
    soft_residents = list(excluded)  # WIP is this necessary?
    
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
    Soft fairness constraint: Penalize residents who are assigned to too many Tuesday/Thursday shifts.
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
        model.Add(hard_day_count == sum(assign[(d, role, r)] for d in hard_days for role in roles))
        
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

def diverse_rotation_penalty(model, assign, days, roles, residents):
    """ 
    Soft constraint version of diverse rotation: 
    - If a resident works 2+ shifts of one role but 0 of another role, add a penalty.
    
    Returns: penalties (list of IntVar): penalty variables to include in objective.
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

def add_role_preferences_by_level(model, assign, roles, days, residents, resident_levels, nf_residents, weight=3):
    """ 
    Encourage R4 residents to prefer NS and ER-2 roles on weekends, 
    and R3 residents to prefer ER-1 and EW roles on weekends. 
    Applies soft penalties for not assigning preferred roles on weekend days. 
    """
    
    penalties = []
    
    for r in residents:
        level = resident_levels.get(r, "").strip().upper()
        
        for d in days:
            d_dt = pd.to_datetime(d) if isinstance(d, str) else d
            is_weekend = d_dt.weekday() in [4, 5]  # Friday or Saturday
            
            for role in roles:
                role_lower = role.strip().lower()
                
                # General weekend preferences
                if is_weekend:
                    if level == "R4" and role_lower in ["er-2 day", "er-2 night"]:
                        penalty_var = model.NewIntVar(0, 1, f"r4_pref_{d}_{role}_{r}")
                        model.Add(assign[(d, role, r)] == 0).OnlyEnforceIf(penalty_var)
                        model.Add(assign[(d, role, r)] == 1).OnlyEnforceIf(penalty_var.Not())
                        penalties.append(penalty_var)
                    
                    if level == "R3" and role_lower in ["er-1 day", "ew day", "ew night"]:
                        penalty_var = model.NewIntVar(0, 1, f"r3_pref_{d}_{role}_{r}")
                        model.Add(assign[(d, role, r)] == 0).OnlyEnforceIf(penalty_var)
                        model.Add(assign[(d, role, r)] == 1).OnlyEnforceIf(penalty_var.Not())
                        penalties.append(penalty_var)
                
                # NF-specific weekend preference
                if r in nf_residents and is_weekend:
                    if level == "R4" and role_lower in ["er-2 day", "er-2 night"]:
                        penalty_var = model.NewIntVar(0, 1, f"r4_nf_pref_{d}_{role}_{r}")
                        model.Add(assign[(d, role, r)] == 0).OnlyEnforceIf(penalty_var)
                        model.Add(assign[(d, role, r)] == 1).OnlyEnforceIf(penalty_var.Not())
                        penalties.append(penalty_var)
                    
                    if level == "R3" and role_lower in ["er-1 day", "ew day", "ew night"]:
                        penalty_var = model.NewIntVar(0, 1, f"r3_nf_pref_{d}_{role}_{r}")
                        model.Add(assign[(d, role, r)] == 0).OnlyEnforceIf(penalty_var)
                        model.Add(assign[(d, role, r)] == 1).OnlyEnforceIf(penalty_var.Not())
                        penalties.append(penalty_var)
    
    return penalties

def add_nf_day_preferences(model, assign, roles, days, nf_residents, weight=3):
    """
    Soft penalty: discourage NF residents from being assigned to day shifts
    on non-preferred weekdays (anything other than Tue/Thu).
    """
    penalties = []
    preferred_days = {1, 3}  # Tuesday=1, Thursday=3

    for r in nf_residents:
        for d in days:
            weekday = pd.to_datetime(d).weekday()
            if weekday not in preferred_days:
                for role in roles:
                    if "day" in role.lower():  # only day shifts
                        # penalty_var = 1 if resident is assigned here, else 0
                        penalty_var = model.NewIntVar(0, 1, f"nf_day_penalty_{d}_{role}_{r}")
                        model.Add(penalty_var == assign[(d, role, r)])
                        penalties.append(penalty_var)

    return penalties
