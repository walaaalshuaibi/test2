from collections import defaultdict
import pandas as pd
import helper

def extract_shift_columns():
    # Extract the columns needed
    nf_roles = ["ER-1 Night", "ER-2 Night", "EW Night"]
    day_roles = ["ER-1 Day", "ER-2 Day", "EW Day"]
    return nf_roles, day_roles

def add_basic_constraints(model, assign, days, roles, residents):
    """ 
    Core constraints:
    1. Every role must be filled each day.
    2. No resident can be assigned to more than one role per day. 
    """
    
    # Each role is filled by exactly one resident per day
    for d in days:
        for role in roles:
            vars_to_sum = [assign[(d, role, r)] for r in residents if (d, role, r) in assign]
            if vars_to_sum:
                model.Add(sum(vars_to_sum) == 1)

    # Each resident works at most one role per day
    for d in days:
        for r in residents:
            vars_to_sum = [assign[(d, role, r)] for role in roles if (d, role, r) in assign]
            if vars_to_sum:
                model.Add(sum(vars_to_sum) <= 1)


# ------------------------- 
# Hard Constraints 
# -------------------------
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
    score_vars):
    """ 
    Add constraints to enforce per-resident limits: 
    1. Minimum and maximum total shifts 
    2. Minimum and maximum weighted points (weekends count double) 
    3. Maximum weekend shifts 
    """
    
    for r in residents:
        total_shifts = sum(assign[(d, role, r)] for d in days for role in roles)
        weekend_shifts = sum(assign[(d, role, r)] for d in weekend_days for role in roles)
        
        # --- Minimums ---
        model.Add(total_shifts >= 0) 
        model.Add(score_vars[r] >= 0)  
        
        # --- Maximums ---
        model.Add(total_shifts <= max_shifts[r])
        model.Add(score_vars[r] <= max_points[r])
        model.Add(weekend_shifts <= weekend_limits[r])

def add_no_consecutive_weekend_constraint(model, assign, roles, residents, weekend_days):
    """
    Prevent residents from working two consecutive weekends.
    
    weekend_days: set of date objects that are Friday or Saturday.
    If a resident works ANY shift on weekend W,
    they cannot work ANY shift on weekend W+1.
    """

    # Convert weekend days into grouped weekends (by ISO week)
    weekend_by_week = {}
    for d in weekend_days:
        week = d.isocalendar().week
        weekend_by_week.setdefault(week, []).append(d)

    sorted_weeks = sorted(weekend_by_week.keys())

    for r in residents:
        for i in range(len(sorted_weeks) - 1):

            w1 = sorted_weeks[i]
            w2 = sorted_weeks[i + 1]

            # BoolVars: did this resident work in weekend w1 or w2?
            worked_w1 = model.NewBoolVar(f"wknd_{w1}_worked_{r}")
            worked_w2 = model.NewBoolVar(f"wknd_{w2}_worked_{r}")

            # Weekend W1 assign vars
            wknd1_vars = [
                assign[(d, role, r)]
                for d in weekend_by_week[w1]
                for role in roles
            ]

            # Weekend W2 assign vars
            wknd2_vars = [
                assign[(d, role, r)]
                for d in weekend_by_week[w2]
                for role in roles
            ]

            # worked_w1 = OR(all roles in weekend 1)
            model.AddMaxEquality(worked_w1, wknd1_vars)

            # worked_w2 = OR(all roles in weekend 2)
            model.AddMaxEquality(worked_w2, wknd2_vars)

            # Rule: If resident works W1 → cannot work W2
            model.Add(worked_w2 == 0).OnlyEnforceIf(worked_w1)
                    
def add_cooldown_constraints(model, assign, days, roles, residents, cooldown=3):
    """
    Hard constraint: if a resident works on day d, they cannot work for the next `cooldown` days.
    """
    for r in residents:
        for i, d in enumerate(days):
            # BoolVar: resident works on day d
            works_d = model.NewBoolVar(f"{r}_works_{d}")
            model.Add(sum(assign[(d, role, r)] for role in roles) == 1).OnlyEnforceIf(works_d)
            model.Add(sum(assign[(d, role, r)] for role in roles) == 0).OnlyEnforceIf(works_d.Not())

            # Block the next `cooldown` days
            for offset in range(1, cooldown + 1):
                if i + offset < len(days):
                    next_d = days[i + offset]
                    works_next = model.NewBoolVar(f"{r}_works_{next_d}")
                    model.Add(sum(assign[(next_d, role, r)] for role in roles) == 1).OnlyEnforceIf(works_next)
                    model.Add(sum(assign[(next_d, role, r)] for role in roles) == 0).OnlyEnforceIf(works_next.Not())

                    # Hard constraint: if works_d then not works_next
                    model.AddImplication(works_d, works_next.Not())

# ------------------------- 
# Soft Constraints 
# -------------------------
def add_limited_shifts_constraint(model, assign, days, roles, limit_dict, max_shifts):
    """ 
    Add constraints to enforce custom shift limits for specific residents,
    and update the max_shifts for those residents to the new limited shifts.
    """
    
    if not limit_dict:
        return
    
    for r, limited_shift in limit_dict.items():
        # Add constraint: resident r cannot have more than limited_shift total
        model.Add(sum(assign[(d, role, r)] for d in days for role in roles) <= limited_shift)
        
        # Update the max_shifts dictionary for resident r
        max_shifts[r] = limited_shift


def add_off_days_constraint(model, assign, roles, off_days):
    """ 
    Add constraints to block residents from being scheduled on specific days. 
    """
    
    # Ensure 'Date' column is datetime
    off_days["date"] = pd.to_datetime(off_days["date"])
    
    for _, row in off_days.iterrows():
        r, d = row["name"], row["date"]
        
        # Block this resident from all roles on this date
        for role in roles:
            if (d, role, r) in assign:
                model.Add(assign[(d, role, r)] == 0)

def add_on_days_constraint(model, assign, roles, on_days):
    """ 
    Add constraints to force residents to be scheduled on specific days. 
    """
    
    for _, row in on_days.iterrows():
        r, d = row["name"], row["date"]
        
        # Resident must be assigned to exactly one role on this date
        model.Add(sum(assign[(d, role, r)] for role in roles) == 1)



# ------------------------- 
# Penalty
# -------------------------
def add_score_balance_soft_penalties(model, score_vars, residents, max_points, threshold=1):
    penalties = []

    # Convert set → list so we can index
    res_list = list(residents)

    for i in range(len(res_list)):
        for j in range(i + 1, len(res_list)):
            r1, r2 = res_list[i], res_list[j]

            # Absolute difference variable
            diff = model.NewIntVar(0, max_points[r1], f"score_diff_{r1}_{r2}")

            # Bool: r1 score >= r2 score
            b = model.NewBoolVar(f"r1_ge_r2_{r1}_{r2}")

            # diff = score[r1] - score[r2] when b = True
            model.Add(diff == score_vars[r1] - score_vars[r2]).OnlyEnforceIf(b)

            # diff = score[r2] - score[r1] when b = False
            model.Add(diff == score_vars[r2] - score_vars[r1]).OnlyEnforceIf(b.Not())

            # Now detect if difference exceeds allowed threshold
            over = model.NewBoolVar(f"over_{r1}_{r2}")
            model.Add(diff >= threshold + 1).OnlyEnforceIf(over)
            model.Add(diff <= threshold).OnlyEnforceIf(over.Not())

            penalties.append(over)

    return penalties

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
        # Count how many Tue/Thu shifts this resident has         
        hard_day_count = model.NewIntVar(0, len(hard_days), f"{r}_hard_day_count")
        model.Add(hard_day_count == sum(assign[(d, role, r)] for d in hard_days for role in roles))
        
        # Boolean flag: does this resident have 2 or more Tue/Thu shifts?         
        has_excess_hard_day = model.NewBoolVar(f"{r}_has_excess_hard_day")
        model.Add(hard_day_count >= 2).OnlyEnforceIf(has_excess_hard_day)
        model.Add(hard_day_count < 2).OnlyEnforceIf(has_excess_hard_day.Not())
        
        # Penalty variable: 
        # - If resident has 0 or 1 Tue/Thu shifts → penalty = 0 
        # - If resident has 2+ → penalty = hard_day_count - 1         
        excess_hard_day = model.NewIntVar(0, len(hard_days), f"{r}_excess_hard_day")
        model.Add(excess_hard_day == hard_day_count - 1).OnlyEnforceIf(has_excess_hard_day)
        model.Add(excess_hard_day == 0).OnlyEnforceIf(has_excess_hard_day.Not())
        
        # Collect penalty variable for use in the objective
        penalty_vars.append(excess_hard_day)
    
    return penalty_vars

def balanced_rotation_penalty(model, assign, days, roles, residents):
    """
    Soft constraint: penalize imbalance in role distribution.
    Differences between role counts should ideally be <= 1.
    Larger differences incur penalties.
    """
    penalties = []

    for r in residents:
        # Count shifts per role
        role_counts = {}
        for role in roles:
            role_counts[role] = model.NewIntVar(0, len(days), f"{r}_{role}_count")
            model.Add(role_counts[role] == sum(assign[(d, role, r)] for d in days))

        # Penalize differences greater than 1
        for i, role in enumerate(roles):
            for j, other_role in enumerate(roles):
                if j <= i:
                    continue

                # Difference between two role counts
                diff = model.NewIntVar(-len(days), len(days), f"{r}_diff_{role}_{other_role}")
                model.Add(diff == role_counts[role] - role_counts[other_role])

                # Absolute difference
                abs_diff = model.NewIntVar(0, len(days), f"{r}_absdiff_{role}_{other_role}")
                model.Add(abs_diff >= diff)
                model.Add(abs_diff >= -diff)

                # Excess beyond ±1 tolerance
                excess = model.NewIntVar(0, len(days), f"{r}_excess_{role}_{other_role}")

                # Boolean: imbalance exceeds 1
                imbalance = model.NewBoolVar(f"{r}_imbalance_{role}_{other_role}")
                model.Add(abs_diff > 1).OnlyEnforceIf(imbalance)
                model.Add(abs_diff <= 1).OnlyEnforceIf(imbalance.Not())

                # Link excess to imbalance
                model.Add(excess == abs_diff - 1).OnlyEnforceIf(imbalance)
                model.Add(excess == 0).OnlyEnforceIf(imbalance.Not())

                penalties.append(excess)

    return penalties

def add_role_preferences_by_level(model, assign, roles, days, residents, resident_levels, nf_residents):

    penalties = []

    # Preferred weekend roles
    preferred = {
        "R4": ["er-2 day"],
        "R3": ["er-1 day","ew day"]
    }

    for r in residents:
        level = resident_levels.get(r, "").strip().upper()

        for d in days:
            d_dt = pd.to_datetime(d) if isinstance(d, str) else d
            is_weekend = d_dt.weekday() in [4, 5]  # Fri/Sat

            if not is_weekend:
                continue

            for role in roles:
                role_lower = role.strip().lower()

                # Determine if this role is preferred
                is_preferred = role_lower in preferred.get(level, [])

                # Create penalty: 1 if assigned to NON-preferred role
                penalty = model.NewIntVar(0, 1, f"pref_penalty_{r}_{d}_{role}")

                # If assigned AND role is not preferred → penalty = 1
                if is_preferred:
                    model.Add(assign[(d, role, r)] == 1).OnlyEnforceIf(penalty.Not())
                    #model.Add(assign[(d, role, r)] == 0).OnlyEnforceIf(penalty)
                else:
                    model.Add(assign[(d, role, r)] == 1).OnlyEnforceIf(penalty)
                    #model.Add(assign[(d, role, r)] == 0).OnlyEnforceIf(penalty.Not())

                penalties.append(penalty)

    return penalties


def add_minimum_spacing_soft_constraint(
    model, assign, days, roles, residents, max_shifts,
    ns_df=None, wr_df=None, nf_calendar_df=None, extra_preassigned=None
):
    """
    Soft spacing constraint considering all assignments (solver + preassigned).
    Penalizes residents for being assigned too close together.

    Returns:
        penalties: list of IntVars representing spacing penalties
    """

    penalties = []

    sorted_days = sorted(days)

    for r in residents:
        # Compute min_gap per resident
        min_gap_r = max(1, 30 // max_shifts[r])

        # Get all assigned dates (preassigned + solver)
        all_assigned = helper.get_all_assigned_dates(
            r=r,
            solver=model,
            assign=assign,
            days=days,
            roles=roles,
            ns_df=ns_df,
            wr_df=wr_df,
            nf_calendar_df=nf_calendar_df,
            extra_preassigned=extra_preassigned
        )

        all_assigned = sorted(all_assigned)

        # ----------------------
        # 1) Pairwise proportional penalties
        # ----------------------
        for i in range(len(all_assigned)):
            for j in range(i + 1, len(all_assigned)):
                d1, d2 = all_assigned[i], all_assigned[j]
                gap = (d2 - d1).days

                if gap < min_gap_r:
                    # Determine if either day is preassigned
                    preassigned_days = set()
                    if ns_df is not None:
                        preassigned_days.update(pd.to_datetime(ns_df.loc[ns_df["name"] == r, "date"]).dt.date)
                    if wr_df is not None:
                        preassigned_days.update(pd.to_datetime(wr_df.loc[wr_df["name"] == r, "date"]).dt.date)
                    if nf_calendar_df is not None:
                        preassigned_days.update(pd.to_datetime(nf_calendar_df.loc[nf_calendar_df["name"] == r, "date"]).dt.date)
                    if extra_preassigned:
                        if isinstance(extra_preassigned, dict):
                            preassigned_days.update([pd.to_datetime(d).date() for d in extra_preassigned.get(r, [])])
                        elif isinstance(extra_preassigned, list):
                            for item in extra_preassigned:
                                if isinstance(item, tuple) and item[0] == r:
                                    preassigned_days.add(pd.to_datetime(item[1]).date())

                    if d1 in preassigned_days and d2 in preassigned_days:
                        continue  # skip both preassigned
                    elif d1 in preassigned_days or d2 in preassigned_days:
                        # penalize only solver-controlled day
                        solver_day = d1 if d2 in preassigned_days else d2
                        for role in roles:
                            penalties.append(assign[(solver_day, role, r)])
                    else:
                        # Both solver-controlled → proportional penalty
                        penalty_var = model.NewIntVar(1, min_gap_r - 1, f"penalty_{r}_{d1}_{d2}")
                        model.Add(penalty_var == min_gap_r - gap)
                        penalties.append(penalty_var)

        # ----------------------
        # 2) Sliding window max-1 enforcement
        # ----------------------
        for i in range(len(sorted_days) - min_gap_r + 1):
            window_days = sorted_days[i:i + min_gap_r]
            window_vars = [assign[(d, role, r)] for d in window_days for role in roles]
            if not window_vars:
                continue
            window_count = model.NewIntVar(0, len(window_vars), f"window_count_{r}_{i}")
            model.Add(window_count == sum(window_vars))
            overassign = model.NewIntVar(0, len(window_vars), f"overassign_{r}_{i}")
            model.AddMaxEquality(overassign, [window_count - 1, 0])
            penalties.append(overassign)

    return penalties

def add_fairness_soft_constraint(
    model, assign, days, roles, residents, max_shifts,
    ns_df=None, wr_df=None, nf_calendar_df=None, extra_preassigned=None
):
    """
    Fairness soft constraint:
    - Compare residents with same max_shifts.
    - Penalize differences in total assignments (solver + preassigned).
    """

    penalties = []

    # Group residents by max_shifts
    groups = defaultdict(list)
    for r in residents:
        groups[max_shifts[r]].append(r)

    # Compute total assignments per resident as IntVar + preassigned count
    total_assignments = {}
    for r in residents:
        # Solver-controlled assignments
        solver_assign_var = model.NewIntVar(0, len(days)*len(roles), f"total_solver_{r}")
        model.Add(solver_assign_var == sum(assign[(d, role, r)] for d in days for role in roles))

        # Preassigned count
        preassigned_count = len(helper.get_all_assigned_dates(
            r=r, solver=None, assign=assign, days=[], roles=[],
            ns_df=ns_df, wr_df=wr_df, nf_calendar_df=nf_calendar_df,
            extra_preassigned=extra_preassigned
        ))

        # Total = solver + preassigned (constant added)
        total_assignments[r] = solver_assign_var + preassigned_count

    # Add fairness penalties within each group
    for k, group in groups.items():
        if len(group) <= 1:
            continue

        for i in range(len(group)):
            for j in range(i+1, len(group)):
                r1, r2 = group[i], group[j]
                diff_var = model.NewIntVar(0, len(days)*len(roles), f"fair_diff_{r1}_{r2}")
                model.AddAbsEquality(diff_var, total_assignments[r1] - total_assignments[r2])
                penalties.append(diff_var)

    return penalties


def weekend_vs_tue_thu_penalty(model, assign, days, roles, residents, weekend_days, threshold=2):
    """
    Soft constraint: if a resident has >= `threshold` weekend shifts,
    then each Tue/Thu assignment for that resident produces a penalty.

    Args:
        model: cp_model.CpModel
        assign: dict[(day, role, resident)] -> BoolVar
        days: iterable of pd.Timestamp (all days in schedule)
        roles: list of role names (strings)
        residents: list of resident names (strings)
        weekend_days: iterable (set/list) of pd.Timestamp that are considered weekend
        threshold: int, weekend-shift count threshold (default 2)

    Returns:
        penalties: list of IntVar (0..1) — include these in the objective.
    """
    penalty_vars = []
    # Precompute Tue/Thu days
    tue_thu_days = [d for d in days if d.strftime("%a") in ["Tue", "Thu"]]

    for r in residents:
        # --- weekend count var ---
        weekend_count = model.NewIntVar(0, len(weekend_days), f"weekend_count_{r}")
        weekend_sum_terms = []
        for d in weekend_days:
            for role in roles:
                key = (d, role, r)
                if key in assign:
                    weekend_sum_terms.append(assign[key])
        if weekend_sum_terms:
            model.Add(weekend_count == sum(weekend_sum_terms))
        else:
            model.Add(weekend_count == 0)

        # Boolean: resident has many weekends (>= threshold)
        has_many_weekends = model.NewBoolVar(f"has_many_weekends_{r}")
        model.Add(weekend_count >= threshold).OnlyEnforceIf(has_many_weekends)
        model.Add(weekend_count < threshold).OnlyEnforceIf(has_many_weekends.Not())

        # --- For each Tue/Thu assignment, create a violation var = assign AND has_many_weekends ---
        for d in tue_thu_days:
            for role in roles:
                key = (d, role, r)
                if key not in assign:
                    continue
                assigned_var = assign[key]  # BoolVar

                # violation bool = assigned_var AND has_many_weekends
                violation = model.NewBoolVar(f"violation_{r}_{d.date()}_{role}")
                model.AddBoolAnd([assigned_var, has_many_weekends]).OnlyEnforceIf(violation)
                model.AddBoolOr([assigned_var.Not(), has_many_weekends.Not()]).OnlyEnforceIf(violation.Not())

                # Penalty variable (0..1), equal to violation
                penalty = model.NewIntVar(0, 1, f"pen_{r}_{d.date()}_{role}")
                model.Add(penalty == violation)

                penalty_vars.append(penalty)

    return penalty_vars

