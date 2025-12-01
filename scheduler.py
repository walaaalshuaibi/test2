import pandas as pd
import random
from ortools.sat.python import cp_model
import preprocess_data
import objective
import general
import helper
import wr
import nf
import blackout
import ns

def schedule_with_ortools_full_modular(
    residents_df,
    start_date,
    num_weeks,
    limited_shift_residents,
    off_days,
    on_days,
    resident_max_limit,
    nf_max_limit,
    resident_year,
    buffers,
    r2_cover,
    optional_rules=None,
    max_consecutive_days=2,
    preassigned_ns_df=None,
    preassigned_wr_df=None
):
    """ 
    Build and solve a resident scheduling problem using OR-Tools CP-SAT. 
    Returns:
        schedule_df (pd.DataFrame): Final schedule with assignments.
        scores_df (pd.DataFrame): Score breakdown for fairness and constraints.
    """
    
    # -----------------------------------------------------
    # 1. Prepare data and blackouts
    # -----------------------------------------------------

    # Assign variables
    (residents_df, residents, resident_max_limit, nf_max_limit, optional_rules, 
     resident_levels, start_date, days, weekend_days, nf_roles, day_roles ) = preprocess_data.prepare_data(
        residents_df,
        start_date,
        num_weeks,
        resident_year,
        nf_max_limit,
        optional_rules,
        resident_max_limit)
    
    # Build NF calendar
    nf_calendar_df = nf.build_nf_calendar(residents_df, start_date, buffers, nf_cols=nf_roles)

    # Blackout Dictionary
    (non_nf_residents, nf_residents, wr_residents, nf_blackout_lookup, combined_blackout_df, combined_blackout_dict,
     night_counts, weekend_rounds_df, vacation_df) = blackout.prepare_blackout(
        buffers, residents_df, start_date, resident_year, r2_cover, on_days, off_days, residents)
    
    # Per-resident caps
    max_shifts, max_points, weekend_limits = helper.calculate_max_limits(
        residents,
        nf_residents,
        resident_max_limit,
        nf_max_limit
    )


    # NS SCHEDULE
    filled_nf_calendar_df, ns_residents, updated_blackout = ns.fill_ns_cells(
        resident_year,
        non_nf_residents, 
        nf_residents,
        nf_max_limit,
        nf_calendar_df, 
        wr_residents=wr_residents, 
        resident_level=resident_levels, 
        blackout_df=combined_blackout_df, 
        nf_cols=nf_roles,
        nf_blackout_lookup=nf_blackout_lookup,
        preassigned_ns_df=preassigned_ns_df
    )

    # Add NS Blackout to Combined Blackout Dict
    blackout.ns_blackout_section(residents, ns_residents, optional_rules, nf_calendar_df, buffers, combined_blackout_dict)
    
    # Shuffle & sort residents by night count (adds randomness for fairness)
    residents = sorted(
        residents,
        key=lambda r: (night_counts.get(r, 0), random.random()))
    

    # -----------------------------------------------------
    # 2. Define model and decision variables
    # -----------------------------------------------------
  
    print("Starting model..........")
    model = cp_model.CpModel()
    
    # Binary variable: assign[d, role, r] = 1 if resident r works role on day d
    assign = {
        (d, role, r): model.NewBoolVar(f"assign_{d.date()}_{role}_{r}")
        for d in days
        for role in day_roles
        for r in residents}
    
    # Calculate scores
    score_vars = {r: model.NewIntVar(0, max_points[r], f"score_{r}") for r in residents}
    
    for r in residents:
        model.Add(score_vars[r] == helper.resident_score_expr(
            assign, days, day_roles, r, weekend_days, weekend_rounds_df, ns_residents
        ))
    
    # -----------------------------------------------------
    # 3. Hard constraints (must always hold)
    # -----------------------------------------------------
    
    # Assign 1 resident for each role in a single day
    general.add_basic_constraints(model, assign, days, day_roles, residents)
    
    # Weekend round assigned roles on the wr day
    if optional_rules["WR_assigned_to_EW"] == True:
        wr.add_weekend_rounds_constraint(model, assign, day_roles, weekend_rounds_df, 
                                         resident_year, preassigned_wr_df=preassigned_wr_df, combined_blackout_dict=combined_blackout_dict)
    
    # Blackout dates where there are no assignments 
    general.add_blackout_constraints(model, assign, day_roles, combined_blackout_dict)
    
    # Caps: total shifts, points, weekend limits
    general.add_shift_cap_constraints(model, assign, days, day_roles, residents, 
    max_shifts, max_points, weekend_days, weekend_limits, score_vars)

    # 3 Day cooldown after every shift
    general.add_cooldown_constraints(model, assign, days, day_roles, residents, cooldown=max_consecutive_days)

    # No 2 weekends 
    general.add_no_consecutive_weekend_constraint(model, assign, day_roles, residents, weekend_days)
    
    # -----------------------------------------------------
    # 4. Optional Rules
    # -----------------------------------------------------
    
    if limited_shift_residents is not None:
        general.add_limited_shifts_constraint(model, assign, days, day_roles, limited_shift_residents, max_shifts)
    
    if off_days is not None:
        general.add_off_days_constraint(model, assign, day_roles, off_days)
    
    if on_days is not None:
        general.add_on_days_constraint(model, assign, day_roles, on_days)

    # -----------------------------------------------------
    # 5. Penalties (Soft Constraints)
    # -----------------------------------------------------
    role_pref_penalties = []
    nf_day_pref_penalties = []
    
    if resident_year == "seniors":
        # Add soft preferences for R3/R4 role tendencies
        role_pref_penalties = general.add_role_preferences_by_level(
            model,
            assign,
            day_roles,
            days,
            residents,
            resident_levels,
            nf_residents
        )

        # Add soft preference for NF day shifts (prefer Tue/Thu)
        nf_day_pref_penalties = nf.add_nf_day_preferences_seniors(
            model,
            assign,
            day_roles,
            days,
            nf_residents
        )

    elif resident_year == "r1":
        # Add preference for NF day shifts (avoid Tue/Thu)
        nf_day_pref_penalties = nf.add_nf_day_preferences_juniors(
            model,
            assign,
            day_roles,
            days,
            nf_residents
        )

    # Non NF Balance
    non_nf_balance_penalties = general.add_score_balance_soft_penalties(model, score_vars, non_nf_residents, max_points)
    
    # NF Balance
    nf_balance_penalties = general.add_score_balance_soft_penalties(model, score_vars, nf_residents, max_points)
    
    # Maximize spacing constraint  
    fixed_preassigned = helper.build_fixed_preassigned(nf_calendar_df, ns_residents, weekend_rounds_df, preassigned_ns_df, preassigned_wr_df)
                                    
    spacing_soft_penalties = general.add_minimum_spacing_soft_constraint(model, assign, days, day_roles, residents, max_shifts, fixed_preassigned)
    spacing_fairness_penalties = general.add_fairness_soft_constraint(model, assign, days, day_roles, residents, max_shifts)


    # Tue/Thu fairness penalty (soft)
    hard_day_penalties = general.tuesday_thursday_fairness_penalty(model, assign, days, day_roles, residents)
    diverse_penalties = general.balanced_rotation_penalty(model, assign, days, day_roles, residents)

    wr_penalties = wr.add_wr_soft_constraints(model, assign, days, day_roles, weekend_rounds_df, weekend_days)

    weekend_vs_tues_thurs_penalties = general.weekend_vs_tue_thu_penalty(model, assign, days, day_roles, residents, weekend_days, threshold=2)
    
    # Build objective
    objective.build_objective(
        model,
        score_vars=score_vars,
        non_nf_balance_penalties=non_nf_balance_penalties,
        nf_balance_penalties=nf_balance_penalties,
        hard_day_penalties=hard_day_penalties,
        diverse_penalties=diverse_penalties,
        role_pref_penalties=role_pref_penalties,
        nf_day_pref_penalties=nf_day_pref_penalties,
        wr_penalties=wr_penalties,
        spacing_penalties=spacing_soft_penalties,
        spacing_fairness_penalties=spacing_fairness_penalties,
        weekend_vs_tues_thurs_penalties=weekend_vs_tues_thurs_penalties,
        non_nf_balance_weight = 1000000000,
        nf_balance_weight = 10000,
        hard_day_weight = 10,
        diverse_weight = 1000,
        role_pref_weight = 1,
        nf_day_pref_weight = 1,
        wr_pref_weight = 1,
        spacing_weight = 100000,
        spacing_fairness_weight = 10000,
        weekend_vs_tues_thurs_weight = 10
    )
    
    # -----------------------------------------------------
    # 7. Solve the model
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
    # 8. Extract results
    # -----------------------------------------------------
    
    return objective.extract_schedule(
        solver,
        assign,
        days,
        day_roles,
        residents,
        wr_residents,
        weekend_rounds_df,
        ns_residents,
        night_counts,
        score_vars,
        max_shifts,
        max_points,
        nf_calendar_df,
        resident_levels,
        limited_shift_residents
    )

