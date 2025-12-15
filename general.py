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

def no_consecutive_weekends_constraint(model, assign, days, roles, residents):
    """
    Hard constraint:
    - If a resident works a Friday, they cannot work the following Fri or Sat
    - If a resident works a Saturday, they cannot work the following Fri or Sat
    """

    days_set = set(days)

    for r in residents:
        for d in days:
            dow = d.strftime("%a")

            if dow not in ["Fri", "Sat"]:
                continue

            # Determine which future days to block
            if dow == "Fri":
                blocked_days = [d + pd.Timedelta(days=7), d + pd.Timedelta(days=8)]
            else:  # Saturday
                blocked_days = [d + pd.Timedelta(days=6), d + pd.Timedelta(days=7)]

            for role in roles:
                if (d, role, r) not in assign:
                    continue

                for bd in blocked_days:
                    if bd not in days_set:
                        continue

                    for role2 in roles:
                        if (bd, role2, r) in assign:
                            # If assigned on this Fri/Sat → forbid next weekend
                            model.Add(assign[(bd, role2, r)] == 0)\
                                 .OnlyEnforceIf(assign[(d, role, r)])
                    
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

def hard_days_balance_and_diversity_penalty(model, assign, days, roles, residents):
    """
    Soft constraints for hard days:
    1️⃣ Hard vs non-hard balance: each resident should have a mix of hard and non-hard days.
    2️⃣ Hard day diversity: Tue, Thu, WE (Fri+Sat) should be roughly evenly distributed.
    Returns:
        balance_penalties, diversity_penalties
    """
    balance_penalties = []
    diversity_penalties = []

    # Define hard day categories
    categories = {
        "Tue": [d for d in days if d.strftime('%a') == "Tue"],
        "Thu": [d for d in days if d.strftime('%a') == "Thu"],
        "WE":  [d for d in days if d.strftime('%a') in ["Fri", "Sat"]],
    }

    for r in residents:
        # --- Count hard days and non-hard days ---
        hard_flags = [assign[(d, role, r)] for d in days for role in roles
                      if d.strftime('%a') in ['Tue','Thu','Fri','Sat'] and (d, role, r) in assign]
        non_hard_flags = [assign[(d, role, r)] for d in days for role in roles
                          if d.strftime('%a') not in ['Tue','Thu','Fri','Sat'] and (d, role, r) in assign]

        if hard_flags and non_hard_flags:
            hard_count = model.NewIntVar(0, len(hard_flags), f"{r}_hard_count")
            model.Add(hard_count == sum(hard_flags))

            non_hard_count = model.NewIntVar(0, len(non_hard_flags), f"{r}_non_hard_count")
            model.Add(non_hard_count == sum(non_hard_flags))

            # Penalty if too few non-hard days relative to hard days
            diff = model.NewIntVar(-len(days), len(days), f"{r}_hard_nonhard_diff")
            model.Add(diff == hard_count - non_hard_count)

            abs_diff = model.NewIntVar(0, len(days), f"{r}_hard_nonhard_absdiff")
            model.AddAbsEquality(abs_diff, diff)

            over = model.NewBoolVar(f"{r}_hard_nonhard_over")
            model.Add(abs_diff > 1).OnlyEnforceIf(over)
            model.Add(abs_diff <= 1).OnlyEnforceIf(over.Not())
            balance_penalties.append(over)

        # --- Hard day diversity ---
        cat_counts = {}
        for cat, cat_days in categories.items():
            flags = [assign[(d, role, r)] for d in cat_days for role in roles if (d, role, r) in assign]
            if flags:
                cat_count = model.NewIntVar(0, len(flags), f"{r}_{cat}_count")
                model.Add(cat_count == sum(flags))
                cat_counts[cat] = cat_count

        cat_list = list(cat_counts.keys())
        for i in range(len(cat_list)):
            for j in range(i+1, len(cat_list)):
                diff = model.NewIntVar(-len(days), len(days), f"{r}_hard_diff_{cat_list[i]}_{cat_list[j]}")
                model.Add(diff == cat_counts[cat_list[i]] - cat_counts[cat_list[j]])

                abs_diff = model.NewIntVar(0, len(days), f"{r}_hard_absdiff_{cat_list[i]}_{cat_list[j]}")
                model.AddAbsEquality(abs_diff, diff)

                over = model.NewBoolVar(f"{r}_hard_diversity_over_{cat_list[i]}_{cat_list[j]}")
                model.Add(abs_diff > 1).OnlyEnforceIf(over)
                model.Add(abs_diff <= 1).OnlyEnforceIf(over.Not())
                diversity_penalties.append(over)

    return balance_penalties, diversity_penalties

def balanced_rotation_penalty(model, assign, days, roles, residents):
    """
    Soft constraint: penalize imbalance in role distribution.
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


def consecutive_hard_days_penalty(model, assign, days, roles, residents, weekend_days, threshold=2):
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

def balanced_day_penalty(model, assign, days, roles, residents):
    """
    Soft constraint: penalize imbalance in day assignments per resident.
    Differences between day categories (Mon, Tue, Wed, Thu, WE, Sun) ideally <= 1.
    Friday+Saturday are considered a single "WE" category.
    Returns a single list of penalties.
    """
    penalties = []

    # Define day categories
    categories = {
        "Mon": [d for d in days if d.strftime("%a") == "Mon"],
        "Tue": [d for d in days if d.strftime("%a") == "Tue"],
        "Wed": [d for d in days if d.strftime("%a") == "Wed"],
        "Thu": [d for d in days if d.strftime("%a") == "Thu"],
        "WE":  [d for d in days if d.strftime("%a") in ["Fri", "Sat"]],
        "Sun": [d for d in days if d.strftime("%a") == "Sun"]
    }

    for r in residents:
        # Count assignments per category
        cat_counts = {}
        for cat, cat_days in categories.items():
            flags = [assign[(d, role, r)] for d in cat_days for role in roles if (d, role, r) in assign]
            if flags:
                count_var = model.NewIntVar(0, len(flags), f"{r}_{cat}_count")
                model.Add(count_var == sum(flags))
                cat_counts[cat] = count_var

        # Penalize pairwise differences > 1
        cat_list = list(cat_counts.keys())
        for i in range(len(cat_list)):
            for j in range(i + 1, len(cat_list)):
                c1, c2 = cat_list[i], cat_list[j]

                # Absolute difference using AddAbsEquality
                diff = model.NewIntVar(0, len(days), f"{r}_day_diff_{c1}_{c2}")
                model.AddAbsEquality(diff, cat_counts[c1] - cat_counts[c2])

                # Boolean: imbalance exceeds 1
                over = model.NewBoolVar(f"{r}_day_over_{c1}_{c2}")
                model.Add(diff > 1).OnlyEnforceIf(over)
                model.Add(diff <= 1).OnlyEnforceIf(over.Not())

                penalties.append(over)

    return penalties

# NEW HARD DAYS
# def hard_vs_nonhard_balance_penalty(model, assign, days, roles, residents):
#     """
#     Soft constraint: penalize if a resident has more than 1 hard day without any non-hard day.
#     Returns list of penalties (one per resident per condition).
#     """
#     penalties = []

#     for r in residents:
#         hard_flags = [assign[(d, role, r)] for d in days for role in roles
#                       if d.strftime('%a') in ['Tue','Thu','Fri','Sat'] and (d, role, r) in assign]
#         non_hard_flags = [assign[(d, role, r)] for d in days for role in roles
#                           if d.strftime('%a') not in ['Tue','Thu','Fri','Sat'] and (d, role, r) in assign]

#         if hard_flags and non_hard_flags:
#             hard_count = model.NewIntVar(0, len(hard_flags), f"{r}_hard_count")
#             model.Add(hard_count == sum(hard_flags))

#             non_hard_count = model.NewIntVar(0, len(non_hard_flags), f"{r}_non_hard_count")
#             model.Add(non_hard_count == sum(non_hard_flags))

#             diff = model.NewIntVar(-len(days), len(days), f"{r}_hard_nonhard_diff")
#             model.Add(diff == hard_count - non_hard_count)

#             abs_diff = model.NewIntVar(0, len(days), f"{r}_hard_nonhard_absdiff")
#             model.AddAbsEquality(abs_diff, diff)

#             over = model.NewBoolVar(f"{r}_hard_nonhard_over")
#             model.Add(abs_diff > 1).OnlyEnforceIf(over)
#             model.Add(abs_diff <= 1).OnlyEnforceIf(over.Not())
#             penalties.append(over)

#     return penalties

def hard_vs_nonhard_balance_constraint(model, assign, days, roles, residents):
    """
    Hard constraint:
    - If a resident has more than 1 hard day, they must also have at least 1 non-hard day.
    """

    for r in residents:
        # Flags for hard days
        hard_flags = [assign[(d, role, r)]
                      for d in days for role in roles
                      if d.strftime('%a') in ['Tue','Thu','Fri','Sat'] and (d, role, r) in assign]

        # Flags for non-hard days
        non_hard_flags = [assign[(d, role, r)]
                          for d in days for role in roles
                          if d.strftime('%a') not in ['Tue','Thu','Fri','Sat'] and (d, role, r) in assign]

        if hard_flags:
            hard_count = model.NewIntVar(0, len(hard_flags), f"{r}_hard_count")
            model.Add(hard_count == sum(hard_flags))

            # If there are possible non-hard slots, track them
            if non_hard_flags:
                non_hard_count = model.NewIntVar(0, len(non_hard_flags), f"{r}_non_hard_count")
                model.Add(non_hard_count == sum(non_hard_flags))
            else:
                # No non-hard slots → treat as always 0
                non_hard_count = model.NewIntVar(0, 0, f"{r}_non_hard_count")
                model.Add(non_hard_count == 0)

            # Boolean: hard_count > 1
            cond = model.NewBoolVar(f"{r}_hard_over1")
            model.Add(hard_count > 1).OnlyEnforceIf(cond)
            model.Add(hard_count <= 1).OnlyEnforceIf(cond.Not())

            # Enforce: if hard_count > 1 → non_hard_count ≥ 1
            model.Add(non_hard_count >= 1).OnlyEnforceIf(cond)

def hard_days_diversity_penalty(model, assign, days, roles, residents):
    """
    Soft constraint: penalize imbalance in distribution of hard days (Tue, Thu, WE) within each resident.
    Returns list of penalties (one per resident per pair of categories).
    """
    penalties = []

    categories = {
        "Tue": [d for d in days if d.strftime('%a') == "Tue"],
        "Thu": [d for d in days if d.strftime('%a') == "Thu"],
        "WE":  [d for d in days if d.strftime('%a') in ["Fri","Sat"]],
    }

    for r in residents:
        cat_counts = {}
        for cat, cat_days in categories.items():
            flags = [assign[(d, role, r)] for d in cat_days for role in roles if (d, role, r) in assign]
            if flags:
                count_var = model.NewIntVar(0, len(flags), f"{r}_{cat}_count")
                model.Add(count_var == sum(flags))
                cat_counts[cat] = count_var

        cat_list = list(cat_counts.keys())
        for i in range(len(cat_list)):
            for j in range(i+1, len(cat_list)):
                diff = model.NewIntVar(-len(days), len(days), f"{r}_hard_diff_{cat_list[i]}_{cat_list[j]}")
                model.Add(diff == cat_counts[cat_list[i]] - cat_counts[cat_list[j]])

                abs_diff = model.NewIntVar(0, len(days), f"{r}_hard_absdiff_{cat_list[i]}_{cat_list[j]}")
                model.AddAbsEquality(abs_diff, diff)

                over = model.NewBoolVar(f"{r}_hard_diversity_over_{cat_list[i]}_{cat_list[j]}")
                model.Add(abs_diff > 1).OnlyEnforceIf(over)
                model.Add(abs_diff <= 1).OnlyEnforceIf(over.Not())
                penalties.append(over)

    return penalties

def hard_days_max_penalty(model, assign, days, roles, residents, max_hard=3):
    """
    Soft constraint: penalize if a resident has more than max_hard hard days.
    Returns list of penalties (one per resident exceeding max_hard).
    """
    penalties = []

    for r in residents:
        hard_flags = [assign[(d, role, r)] for d in days for role in roles
                      if d.strftime('%a') in ['Tue','Thu','Fri','Sat'] and (d, role, r) in assign]
        if hard_flags:
            hard_count = model.NewIntVar(0, len(hard_flags), f"{r}_hard_count")
            model.Add(hard_count == sum(hard_flags))

            excess = model.NewIntVar(0, len(hard_flags), f"{r}_hard_excess")
            over = model.NewBoolVar(f"{r}_hard_over_max")
            model.Add(hard_count > max_hard).OnlyEnforceIf(over)
            model.Add(hard_count <= max_hard).OnlyEnforceIf(over.Not())
            model.Add(excess == hard_count - max_hard).OnlyEnforceIf(over)
            model.Add(excess == 0).OnlyEnforceIf(over.Not())
            penalties.append(excess)

    return penalties



