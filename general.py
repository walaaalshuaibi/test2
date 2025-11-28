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


def add_shift_cap_constraints( model, assign, days, roles, residents, max_shifts, max_points, weekend_days, weekend_limits, weekend_rounds_df=None, ns_residents_df=None ): 
    """ Add constraints to enforce per-resident limits: 
    1. Minimum and maximum total shifts 
    2. Minimum and maximum weighted points (weekends count double) 
    3. Maximum weekend shifts """ 
    for r in residents: 
        total_shifts = sum(assign[(d, role, r)] for d in days for role in roles) 
        total_points = helper.resident_score_expr(assign, days, roles, r, weekend_days, weekend_rounds_df, ns_residents_df) 
        weekend_shifts = sum(assign[(d, role, r)] for d in weekend_days for role in roles) 
        # --- Minimums --- 
        model.Add(total_shifts > 0) 
        model.Add(total_points > 0) 
        # --- Maximums --- 
        model.Add(total_shifts <= max_shifts[r]) 
        model.Add(total_points <= max_points[r]) 
        model.Add(weekend_shifts <= weekend_limits[r])

def add_no_consecutive_weekend_constraint(
    model,
    assign,
    groups,
    roles,
    residents,
    label="group",
    min_gap_groups=1,
    prevent_same_role=False,
):
    """ 
    Prevent residents from being scheduled in consecutive (or near-consecutive) groups.
    
    Args:
        model: OR-Tools CpModel
        assign: dict[(day, role, resident)] = BoolVar
        groups: list[list[day]] - each element is a list of days representing a group
        roles: list[str]
        residents: list[str]
        label: str - label for variable naming
        min_gap_groups: int - how many consecutive groups to block (default=1)
        prevent_same_role: bool - also block same role twice in a row (default=False)
    """
    
    for r in residents:
        # --- Prevent working within N consecutive groups ---
        for i in range(len(groups) - min_gap_groups):
            # mark if resident worked in each group
            group_vars = []
            for g in range(i, i + min_gap_groups + 1):
                group_days = groups[g]
                worked = model.NewBoolVar(f"{label}_{g}_worked_{r}")
                model.Add(sum(assign[(d, role, r)] for d in group_days for role in roles) >= 1).OnlyEnforceIf(worked)
                model.Add(sum(assign[(d, role, r)] for d in group_days for role in roles) == 0).OnlyEnforceIf(worked.Not())
                group_vars.append(worked)
            
            # resident cannot work in two of these consecutive groups
            # e.g. for min_gap_groups=1 → block group[i] and group[i+1]
            for j in range(len(group_vars) - 1):
                model.AddBoolOr([group_vars[j].Not(), group_vars[j + 1].Not()])

        # --- Prevent same role twice in a row ---
        if prevent_same_role:
            all_days = [d for g in groups for d in g]  # flatten groups
            all_days = sorted(all_days)  # ensure chronological order
            for i in range(len(all_days) - 1):
                d1, d2 = all_days[i], all_days[i + 1]
                for role in roles:
                    model.AddBoolOr([
                        assign[(d1, role, r)].Not(),
                        assign[(d2, role, r)].Not()
                    ])
                    
def add_cooldown_constraints(model, assign, days, roles, residents, cooldown=3):
    """
    Hard constraint: if a resident works on day d, they cannot work for the next `cooldown` days.
    """
    cooldown=cooldown+1
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
    - Penalties only apply when the difference > 1 (one-off is free).
    """
    
    # Score variables
    score_vars = {r: model.NewIntVar(0, 500, f"score_{r}") for r in residents}
    
    for r in residents:
        model.Add(score_vars[r] == helper.resident_score_expr(
            assign, days, roles, r, weekend_days, weekend_rounds_df, ns_residents_df
        ))
    
    # Exclusions
    excluded = set()
    # if vacation_df is not None and not vacation_df.empty:
    #     excluded.update(vacation_df["name"].unique())
    if limited_shift_residents is not None:
        excluded.update(limited_shift_residents.keys())
    # if off_days is not None and not off_days.empty:
    #     excluded.update(off_days["name"].unique())
    
    balance_penalties = []
    
    # Hard balance: unconstrained residents 
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
            penalty = model.NewIntVar(0, 500, f"balance_penalty_{r1}_{r2}")
            model.Add(penalty >= abs_diff - 1)
            balance_penalties.append(penalty)
    
    # Balance NF hard residents pairwise
    for i in range(len(nf_hard)):
        for j in range(i + 1, len(nf_hard)):
            r1, r2 = nf_hard[i], nf_hard[j]
            diff = model.NewIntVar(-500, 500, f"nf_score_diff_{r1}_{r2}")
            model.Add(diff == score_vars[r1] - score_vars[r2])
            abs_diff = model.NewIntVar(0, 500, f"nf_abs_score_diff_{r1}_{r2}")
            model.AddAbsEquality(abs_diff, diff)
            penalty = model.NewIntVar(0, 500, f"nf_balance_penalty_{r1}_{r2}")
            model.Add(penalty >= abs_diff - 1)
            balance_penalties.append(penalty)
    
    # Soft balance: constrained residents 
    soft_residents = list(excluded)
    
    for r in soft_residents:
        for peer in hard_residents:
            diff = model.NewIntVar(-500, 500, f"soft_diff_{r}_{peer}")
            model.Add(diff == score_vars[r] - score_vars[peer])
            abs_diff = model.NewIntVar(0, 500, f"soft_abs_diff_{r}_{peer}")
            model.AddAbsEquality(abs_diff, diff)
            penalty = model.NewIntVar(0, 500, f"soft_balance_penalty_{r}_{peer}")
            model.Add(penalty >= abs_diff - 1)
            balance_penalties.append(penalty)
    
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
    model, assign, days, roles, residents, max_shifts, fixed_preassigned
):
    """
    Soft minimum-spacing constraints.
    - Includes BOTH preassigned and solver assignments.
    - Penalizes assignments that occur too close together.
    """

    penalties = []
    sorted_days = sorted(days)

    # Convert preassigned map into: fixed[r] = {dates}
    fixed_map = {r: set() for r in residents}
    for (d, r) in fixed_preassigned:
        fixed_map[r].add(d)

    # -------------------------------
    # MAIN LOOP PER RESIDENT
    # -------------------------------
    for r in residents:

        min_gap_r = max(1, 30 // max_shifts[r])

        # Collect solver-controlled assignment variables grouped by day
        solver_map = {d: [] for d in sorted_days}
        for d in sorted_days:
            for role in roles:
                solver_map[d].append(assign[(d, role, r)])

        # -------------------------------------
        # 1) PAIRWISE penalty: (d1, d2) too close
        # -------------------------------------
        for i in range(len(sorted_days)):
            for j in range(i + 1, len(sorted_days)):
                d1, d2 = sorted_days[i], sorted_days[j]
                gap = (d2 - d1).days
                if gap >= min_gap_r:
                    continue  # safe distance → no penalty needed

                d1_fixed = d1 in fixed_map[r]
                d2_fixed = d2 in fixed_map[r]

                # Case A: both preassigned → ignore
                if d1_fixed and d2_fixed:
                    continue

                # Case B: one fixed, one solver → penalize solver assignments
                if d1_fixed and not d2_fixed:
                    penalties.extend(solver_map[d2])
                    continue
                if d2_fixed and not d1_fixed:
                    penalties.extend(solver_map[d1])
                    continue

                # Case C: both solver-controlled → add proportional penalty
                # penalty = (min_gap - gap) * count(d1) * count(d2)
                weight = (min_gap_r - gap)

                for v1 in solver_map[d1]:
                    for v2 in solver_map[d2]:
                        # product: v1 AND v2
                        both = model.NewBoolVar(f"pair_{r}_{d1}_{d2}")
                        model.AddBoolAnd([v1, v2]).OnlyEnforceIf(both)
                        model.Add(both == 0).OnlyEnforceIf(v1.Not())
                        model.Add(both == 0).OnlyEnforceIf(v2.Not())

                        # scaled penalty
                        p = model.NewIntVar(0, weight, f"penalty_{r}_{d1}_{d2}")
                        model.Add(p == weight * both)
                        penalties.append(p)

        # --------------------------------------
        # 2) Sliding window (max 1 shift inside window)
        # --------------------------------------
        for i in range(len(sorted_days) - min_gap_r + 1):
            window_days = sorted_days[i : i + min_gap_r]
            vars_in_window = []

            for d in window_days:
                vars_in_window.extend(solver_map[d])

            # also count fixed (static +1)
            fixed_count = sum(1 for d in window_days if d in fixed_map[r])

            # If fixed_count > 1, unavoidable → skip
            if fixed_count > 1:
                continue

            # window_count = solver_boxes + fixed_count
            if vars_in_window:
                window_sum = model.NewIntVar(0, len(vars_in_window), f"w_sum_{r}_{i}")
                model.Add(window_sum == sum(vars_in_window))

                total = model.NewIntVar(0, len(vars_in_window) + fixed_count, f"w_total_{r}_{i}")
                model.Add(total == window_sum + fixed_count)

                # Penalize (total − 1) if > 1
                over = model.NewIntVar(0, len(vars_in_window), f"w_over_{r}_{i}")
                model.AddMaxEquality(over, [total - 1, 0])
                penalties.append(over)

    return penalties




def add_spacing_fairness_soft_constraint(
    model, assign, days, roles, residents, max_shifts, fixed_preassigned
):
    """
    Soft constraint for spacing fairness.
    Penalizes short gaps between ANY two assigned dates
    (preassigned + solver assignments).
    """

    penalties = []

    # Precompute real date objects for your gap math
    real_dates = {d: d for d in days}

    # Build assigned_days as bool vars for solver assignments
    assigned_days = {}
    for r in residents:
        assigned_days[r] = {}
        for d in days:
            b = model.NewBoolVar(f"{r}_works_day_{d}")
            model.Add(b == sum(assign[(d, role, r)] for role in roles))
            assigned_days[r][d] = b

    # Convert fixed_preassigned to fast lookup
    fixed_lookup = {r: set(dates) for r, dates in fixed_preassigned.items()}

    # -------------------------------------------------------
    # Spacing fairness:
    # Penalize small gaps between ALL assigned days:
    # (Fixed preassigned + solver assignments)
    # -------------------------------------------------------
    for r in residents:
        # mark preassigned as always-assigned days
        pre = fixed_lookup.get(r, set())

        for i in range(len(days)):
            for j in range(i + 1, len(days)):

                d1 = days[i]
                d2 = days[j]
                gap = (real_dates[d2] - real_dates[d1]).days

                if gap <= 0:
                    continue

                # Detect assigned-by-solver
                v1 = assigned_days[r][d1]
                v2 = assigned_days[r][d2]

                # Detect preassigned (constant 1 if preassigned)
                p1 = 1 if d1 in pre else 0
                p2 = 1 if d2 in pre else 0

                # both assigned? (solver OR preassigned)
                both_assigned = model.NewBoolVar(f"spacing_both_{r}_{d1}_{d2}")

                # both_assigned = (v1 or p1) AND (v2 or p2)
                # Create booleans for (v1 or p1) and (v2 or p2)
                w1 = model.NewBoolVar(f"{r}_works_or_pre_d1_{d1}")
                w2 = model.NewBoolVar(f"{r}_works_or_pre_d2_{d2}")

                # w1 = max(v1, p1)
                model.Add(w1 >= v1)
                model.Add(w1 >= p1)
                model.Add(w1 <= v1 + p1)

                # w2 = max(v2, p2)
                model.Add(w2 >= v2)
                model.Add(w2 >= p2)
                model.Add(w2 <= v2 + p2)

                # both_assigned = w1 AND w2
                model.Add(both_assigned >= w1 + w2 - 1)
                model.Add(both_assigned <= w1)
                model.Add(both_assigned <= w2)

                # Weight: small gap → big penalty
                weight = int(1000 / gap)

                p = model.NewIntVar(0, weight, f"spacing_penalty_{r}_{d1}_{d2}")
                model.Add(p == weight * both_assigned)

                penalties.append(p)

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

