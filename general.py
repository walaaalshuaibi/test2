from collections import defaultdict
import pandas as pd
import helper

def extract_shift_columns(resident_year):
    # Extract the columns needed
    if resident_year == 'r1':
        nf_roles = ["ER-1 Night", "EW Night", "ER-2 Night"]
    else:
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

def add_no_thursday_after_weekend_constraint(model, assign, days, roles, residents, weekend_days, weekend_rounds_df):
    """
    Hard constraint:
    - If a resident has 2 or more weekend shifts (or weekend rounds),
      then they cannot be assigned to any role on Thursday.
    
    Args:
        model: CpModel
        assign: dict[(day, role, resident)] -> BoolVar
        days: list of pd.Timestamp
        roles: list of roles
        residents: list of resident names
        weekend_days: list of pd.Timestamp that are weekends
        weekend_rounds_df: pd.DataFrame with columns ["date", "name"] for weekend rounds
    """
    thursdays = [d for d in days if d.strftime("%a") == "Thu"]

    # Precompute weekend round counts per resident
    weekend_rounds_count = {}
    if weekend_rounds_df is not None and not weekend_rounds_df.empty:
        weekend_rounds_count = weekend_rounds_df.groupby("name")["date"].nunique().to_dict()
    
    for r in residents:
        # Collect all weekend shift variables for this resident
        weekend_assign_vars = [
            assign[(d, role, r)]
            for d in weekend_days
            for role in roles
            if (d, role, r) in assign
        ]
        
        # Count of weekend rounds (fixed preassigned)
        rounds = weekend_rounds_count.get(r, 0)
        
        if not weekend_assign_vars:
            continue
        
        # Sum of weekend shifts as IntVar
        weekend_shifts_sum = model.NewIntVar(0, len(weekend_assign_vars), f"{r}_weekend_sum")
        model.Add(weekend_shifts_sum == sum(weekend_assign_vars))
        
        # Boolean: exceeds 1 weekend shift or 1 weekend round
        exceeds_two = model.NewBoolVar(f"{r}_exceeds_two_weekend")
        model.Add(weekend_shifts_sum + rounds >= 2).OnlyEnforceIf(exceeds_two)
        model.Add(weekend_shifts_sum + rounds < 2).OnlyEnforceIf(exceeds_two.Not())
        
        # For each Thursday, make sure no role is assigned if exceeds_two is True
        for thursday in thursdays:
            for role in roles:
                key = (thursday, role, r)
                if key in assign:
                    model.Add(assign[key] == 0).OnlyEnforceIf(exceeds_two)

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
    Soft fairness constraint: Penalize residents assigned to Tue/Thu/Fri/Sat too often.
    
    Rules:
    - 0 or 1 hard days → no penalty
    - 2+ hard days → penalty = count - 1
    """
    
    penalty_vars = []

    # Filter all Tue/Thu/Fri/Sat dates
    hard_days = [d for d in days if d.strftime('%a') in ['Tue', 'Thu', 'Fri', 'Sat']]

    for resident in residents:
        # Count number of hard day assignments for this resident
        hard_day_count = model.NewIntVar(0, len(hard_days) * len(roles), f"{resident}_hard_day_count")
        model.Add(hard_day_count == sum(assign[(d, role, resident)] for d in hard_days for role in roles))

        # Boolean flag for exceeding 1 hard day
        exceeds_one = model.NewBoolVar(f"{resident}_exceeds_one_hard_day")
        model.Add(hard_day_count >= 2).OnlyEnforceIf(exceeds_one)
        model.Add(hard_day_count < 2).OnlyEnforceIf(exceeds_one.Not())

        # Penalty variable: count - 1 if >1, else 0
        penalty = model.NewIntVar(0, len(hard_days) * len(roles), f"{resident}_hard_day_penalty")
        model.Add(penalty == hard_day_count - 1).OnlyEnforceIf(exceeds_one)
        model.Add(penalty == 0).OnlyEnforceIf(exceeds_one.Not())

        penalty_vars.append(penalty)

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
        "r4": ["er-2 day"],
        "r3": ["er-1 day","ew day"]
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
    model, assign, days, roles, residents, fixed_preassigned, min_gap=7
):
    penalties = []
    sorted_days = sorted(days)

    for r in residents:
        # 1) Fixed preassigned dates for this resident
        fixed_dates = sorted(fixed_preassigned.get(r, []))

        # -----------------------------------------------------------
        # A) Spacing between one solver date and one fixed date
        # -----------------------------------------------------------
        for d in sorted_days:
            for fd in fixed_dates:
                gap = abs((d - fd).days)
                if gap < min_gap:
                    # All roles that could assign this resident on day d
                    assign_vars = [
                        assign[(d, role, r)]
                        for role in roles
                        if (d, role, r) in assign
                    ]
                    if not assign_vars:
                        continue

                    # If ANY role assigns them on this day → violation
                    assigned_today = model.NewBoolVar(f"{r}_{d}_close_to_fixed")
                    model.AddBoolOr(assign_vars).OnlyEnforceIf(assigned_today)
                    model.Add(sum(assign_vars) == 0).OnlyEnforceIf(assigned_today.Not())

                    penalties.append(assigned_today)

        # -----------------------------------------------------------
        # B) Spacing between solver-assigned days
        # -----------------------------------------------------------
        for i in range(len(sorted_days)):
            for j in range(i+1, len(sorted_days)):
                d1 = sorted_days[i]
                d2 = sorted_days[j]

                gap = (d2 - d1).days
                if gap < min_gap:
                    # Collect vars for each day
                    vars_d1 = [
                        assign[(d1, role, r)]
                        for role in roles
                        if (d1, role, r) in assign
                    ]
                    vars_d2 = [
                        assign[(d2, role, r)]
                        for role in roles
                        if (d2, role, r) in assign
                    ]

                    if not vars_d1 or not vars_d2:
                        continue

                    any_d1 = model.NewBoolVar(f"{r}_{d1}_any")
                    any_d2 = model.NewBoolVar(f"{r}_{d2}_any")

                    model.AddBoolOr(vars_d1).OnlyEnforceIf(any_d1)
                    model.Add(sum(vars_d1) == 0).OnlyEnforceIf(any_d1.Not())

                    model.AddBoolOr(vars_d2).OnlyEnforceIf(any_d2)
                    model.Add(sum(vars_d2) == 0).OnlyEnforceIf(any_d2.Not())

                    violation = model.NewBoolVar(f"gap_violation_{r}_{d1}_{d2}")
                    model.AddBoolAnd([any_d1, any_d2]).OnlyEnforceIf(violation)
                    model.AddBoolOr([any_d1.Not(), any_d2.Not()]).OnlyEnforceIf(violation.Not())

                    penalties.append(violation)

    return penalties


def weekend_vs_tue_thu_penalty(model, assign, days, roles, residents, weekend_days, threshold=2):
    """
    Soft constraint: for weekend + Tue/Thu assignments, penalize consecutive assignments.
    If a resident has an assignment on one of these days, the next assignment on any of these
    days produces a penalty, to encourage fairness.

    Args:
        model: cp_model.CpModel
        assign: dict[(day, role, resident)] -> BoolVar
        days: list of pd.Timestamp (all schedule days, sorted)
        roles: list of role names
        residents: list of resident names
        weekend_days: list/set of pd.Timestamp considered weekend
        threshold_days: list of weekday strings to consider in addition to weekend (default Tue/Thu)

    Returns:
        penalty_vars: list of IntVar (0..1) — include in objective
    """
    threshold_days=["Tue","Thu"]
    penalty_vars = []

    # Combine all “critical days”
    critical_days = [d for d in days if d in weekend_days or d.strftime("%a") in threshold_days]
    critical_days = sorted(critical_days)  # make sure days are chronological

    for r in residents:
        # Keep track of the previous critical assignment
        prev_assigned = None

        for d in critical_days:
            for role in roles:
                key = (d, role, r)
                if key not in assign:
                    continue
                assigned_var = assign[key]

                if prev_assigned is None:
                    # First critical day → no penalty
                    prev_assigned = assigned_var
                    continue

                # Penalty if assigned_var AND previous was assigned
                violation = model.NewBoolVar(f"consec_pen_{r}_{d.date()}_{role}")
                model.AddBoolAnd([assigned_var, prev_assigned]).OnlyEnforceIf(violation)
                model.AddBoolOr([assigned_var.Not(), prev_assigned.Not()]).OnlyEnforceIf(violation.Not())

                # IntVar for penalty
                penalty = model.NewIntVar(0, 1, f"pen_{r}_{d.date()}_{role}")
                model.Add(penalty == violation)

                penalty_vars.append(penalty)

                # Update previous assigned for next iteration
                prev_assigned = assigned_var

    return penalty_vars

