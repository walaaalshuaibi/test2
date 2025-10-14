import pandas as pd
import random
from ortools.sat.python import cp_model
import preprocess_data
import max_limits
import objective
import penalty
import constraints

def build_weekend_groups(days):
    days_sorted = sorted(days)
    return [
        [days_sorted[i], days_sorted[i + 1]]
        for i in range(len(days_sorted) - 1)
        if days_sorted[i].weekday() == 4 and days_sorted[i + 1].weekday() == 5
    ]

def schedule_with_ortools_full_modular(
    residents_df, 
    start_date, 
    num_weeks, 
    limited_shift_residents, 
    off_days, 
    on_days, 
    resident_max_limit, 
    nf_max_limit,
    resident_year
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
    
    residents_df.columns = residents_df.columns.str.strip().str.lower()
    
    (
        residents, 
        resident_level, 
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
        nf_calendar_df
    ) = preprocess_data.prepare_data(residents_df, start_date, num_weeks, resident_year)
    
    # Per-resident caps
    max_shifts, max_points, weekend_limits = max_limits.calculate_max_limits(
        residents, 
        nf_residents, 
        wr_residents, 
        night_counts,
        resident_max_limit, 
        nf_max_limit
    )
    
    # Shuffle & sort residents by night count (adds randomness for fairness)
    residents = sorted(
        residents, 
        key=lambda r: (night_counts.get(r, 0), random.random()))
    
    # -----------------------------------------------------
    # 2. Define model and decision variables
    # -----------------------------------------------------
  
    model = cp_model.CpModel()
    
    # Binary variable: assign[d, role, r] = 1 if resident r works role on day d
    assign = {
        (d, role, r): model.NewBoolVar(f"assign_{d.date()}_{role}_{r}")
        for d in days 
        for role in roles 
        for r in residents
    }
    
    # -----------------------------------------------------
    # 3. Hard constraints (must always hold)
    # -----------------------------------------------------
    
    constraints.add_basic_constraints(model, assign, days, roles, residents)
    
    # Weekend rounds coverage and blackout rules
    constraints.add_weekend_rounds_constraint(model, assign, roles, weekend_rounds_df)
    
    for blackout in [nf_blackout, wr_blackout, ns_blackout, vacation_blackout]:
        constraints.add_blackout_constraints(model, assign, roles, blackout)
    
    # Caps: total shifts, points, weekend limits
    constraints.add_shift_cap_constraints(
        model, 
        assign, 
        days, 
        roles, 
        residents, 
        max_shifts, 
        max_points, 
        weekend_days, 
        weekend_limits, 
        ns_residents_df=ns_residents
    )
    
    # -----------------------------------------------------
    # 3b. No consecutive groups (days + Fri/Sat weekends)
    # -----------------------------------------------------
    
    # Each day is its own group (no consecutive days)
    day_groups = [[d] for d in days]
    constraints.add_no_consecutive_groups_constraint(
        model, 
        assign, 
        day_groups, 
        roles, 
        residents, 
        label="day"
    )
    
    # Build weekend groups as explicit Friday–Saturday pairs
    weekend_groups = build_weekend_groups(days)
    constraints.add_no_consecutive_groups_constraint(
        model, 
        assign, 
        weekend_groups, 
        roles, 
        residents, 
        label="weekend"
    )
    
    # -----------------------------------------------------
    # 4. Soft constraints (preferences, optional rules)
    # -----------------------------------------------------
    
    if limited_shift_residents is not None:
        constraints.add_limited_shifts_constraint(model, assign, days, roles, limited_shift_residents)
    
    if off_days is not None:
        constraints.add_off_days_constraint(model, assign, roles, off_days)
    
    if on_days is not None:
        constraints.add_on_days_constraint(model, assign, roles, on_days)
    
    # Add soft preferences for R3/R4 role tendencies
    role_pref_penalties = penalty.add_role_preferences_by_level(
        model, 
        assign, 
        roles, 
        days, 
        residents, 
        resident_level, 
        nf_residents, 
        weight=3
    )

    # Add soft preference for NF day shifts (prefer Tue/Thu)
    nf_day_pref_penalties = penalty.add_nf_day_preferences(
        model,
        assign,
        roles,
        days,
        nf_residents,
        weight=3
    )
    
    # -----------------------------------------------------
    # 5. Fairness scoring + objective
    # -----------------------------------------------------
    
    balance_penalties, score_vars = penalty.add_score_balance_constraint(
        model, 
        assign, 
        days, 
        roles, 
        residents, 
        weekend_days, 
        nf_residents, 
        weekend_rounds_df, 
        ns_residents, 
        night_counts, 
        vacation_df, 
        limited_shift_residents, 
        off_days
    )
    
    # Tue/Thu fairness penalty (soft)
    hard_day_penalties = penalty.tuesday_thursday_fairness_penalty(model, assign, days, roles, residents)
    diverse_penalties = penalty.diverse_rotation_penalty(model, assign, days, roles, residents)
    
    # Build objective
    objective.build_objective(
        model, 
        score_vars=score_vars, 
        balance_penalties=balance_penalties, 
        hard_day_penalties=hard_day_penalties, 
        diverse_penalties=diverse_penalties, 
        role_pref_penalties=role_pref_penalties,
        nf_day_pref_penalties=nf_day_pref_penalties, 
        balance_weight=10, 
        hard_day_weight=5, 
        diverse_weight=10, 
        role_pref_weight=13
    )
    
    # -----------------------------------------------------
    # 6. Solve the model
    # -----------------------------------------------------
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.random_seed = 42
    solver.parameters.num_search_workers = 8
    solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
    
    status = solver.Solve(model)
    
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise RuntimeError("No feasible solution found")
    
    # -----------------------------------------------------
    # 7. Extract results
    # -----------------------------------------------------
    
    return objective.extract_schedule(
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
        nf_calendar_df,
        resident_year
    )

