from collections import defaultdict
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
import streamlit as st
import new 

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
    max_consecutive_days=3,
    preassigned_ns_df=None,
    preassigned_wr_df=None):
    """ 
    Build and solve a resident scheduling problem using OR-Tools CP-SAT. 
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
    nf_calendar_df = nf.build_nf_calendar(residents_df, start_date, nf_cols=nf_roles)

    # Blackout and Grayouts Dictionary
    blackout_data = blackout.prepare_blackouts_and_grayouts(
        buffers=buffers,
        residents_df=residents_df,
        start_date=start_date,
        num_weeks=num_weeks,
        resident_year=resident_year,
        r2_cover=r2_cover,
        on_days=on_days,
        off_days=off_days,
        residents=residents
    )

    combined_blackout_df = blackout_data["blackouts"] # Fix: change name to combined
    soft_grayouts = blackout_data["grayouts"]
    night_counts = blackout_data["night_counts"]
    wr_residents = blackout_data["wr_residents"]
    nf_residents = blackout_data["nf_residents"]
    non_nf_residents = blackout_data["non_nf_residents"]

    # FIX: Use both gray and black for seniors, but for r1's. if gray override with ns
    # Build NS calendar
    nf_calendar_df, ns_residents, combined_blackout_df = ns.fill_ns_cells(
        resident_year,
        non_nf_residents, 
        nf_residents,
        nf_calendar_df, 
        wr_residents=wr_residents, 
        resident_level=resident_levels, 
        blackout_df=combined_blackout_df, 
        soft_grayouts=soft_grayouts,
        nf_cols=nf_roles,
        preassigned_ns_df=preassigned_ns_df)

    soft_grayouts = blackout.ns_grayout_section(ns_residents, optional_rules, nf_calendar_df, buffers, combined_blackout_df, soft_grayouts)

    # Per-resident caps # DELETE LATER
    max_shifts, max_points = helper.calculate_max_limits(
        residents,
        nf_residents,
        resident_max_limit,
        nf_max_limit,
        night_counts
    )
    
    # Count WR Assignmnets
    wr_counts = (
        wr_residents
        .assign(date=pd.to_datetime(wr_residents["date"]).dt.date)
        .groupby("name")["date"]
        .nunique()
        .to_dict()
    ) if wr_residents is not None and not wr_residents.empty else {}
            
    # Shuffle & sort residents by night count (adds randomness for fairness)
    residents = sorted(
        residents,
        key=lambda r: (night_counts.get(r, 0), random.random())
    )
        
    # -----------------------------------------------------
    # 7. Solve the model
    # -----------------------------------------------------
    last_feasible_solver = None

    # Update caps for this iteration 

    print("Starting model..........")
    model = cp_model.CpModel()
    
    # Binary variable: assign[d, role, r] = 1 if resident r works role on day d
    assign = {
        (d, role, r): model.NewBoolVar(f"assign_{d.date()}_{role}_{r}")
        for d in days
        for role in day_roles
        for r in residents}
    
    # Calculate scores # max_points[r]
    score_vars = {r: model.NewIntVar(0, max_points[r], f"score_{r}") for r in residents}

    for r in residents:
        model.Add(score_vars[r] == helper.resident_score_expr(
            assign, days, day_roles, r, weekend_days, wr_residents, ns_residents))
        
    # -----------------------------------------------------
    # 3. Hard constraints (must always hold)
    # -----------------------------------------------------
    # Shift and score cap constraint
    # general.add_shift_cap_constraints (model, assign, days, day_roles, residents, 
    # max_shifts, max_points, score_vars)

    # Assign 1 resident for each role in a single day
    general.add_basic_constraints(model, assign, days, day_roles, residents)
    
    # Weekend round assigned roles on the wr day
    if optional_rules["WR_assigned_to_EW"] == True:
        wr.add_weekend_rounds_constraint(model, assign, day_roles, wr_residents, 
                                        resident_year, preassigned_wr_df=preassigned_wr_df, combined_blackout_dict=combined_blackout_df, soft_grayouts=soft_grayouts)
    
    # Blackout dates where there are no assignments 
    general.add_blackout_constraints(model, assign, day_roles, combined_blackout_df, soft_grayouts)

    # 3 day cooldown period after every shift
    general.add_cooldown_constraints(model, assign, days, day_roles, residents, cooldown=max_consecutive_days+1)

    # No two consecutive weekends
    general.no_consecutive_weekends_constraint(model, assign, days, day_roles, residents)

    # 2 Weekends or 2 WR => No Thursday
    general.add_no_thursday_after_weekend_constraint(model, assign, days, day_roles, residents, weekend_days, wr_residents)

    # 2 Weekends or 2 WR => No more Weekends
    # if resident_year == "r1":
    #     wr.add_wr_hard_constraints(
    #         model,
    #         assign,
    #         days,
    #         day_roles,
    #         preassigned_wr_df,
    #         weekend_days)

    # Have at least 1 'easy' day if they have more than 1 day of 'hard' day
    general.hard_vs_nonhard_balance_constraint(model, assign, days, day_roles, residents)

    # NS can't have more Thursdays
    ns.night_thursday_hard_constraint(model, assign, days, day_roles, ns_residents)

    nf_day_pref_penalty = []
    if resident_year == "seniors":
        # NF must always be assigned thursdays and tuesdays
        nf_day_pref_penalty = nf.add_nf_day_preferences_seniors_soft(model, assign, day_roles, days, nf_residents)
    
    # -----------------------------------------------------
    # 4. Optional Rules
    # -----------------------------------------------------
    
    if limited_shift_residents is not None:
        general.add_limited_shifts_constraint(model, assign, days, day_roles, limited_shift_residents, max_shifts)
    
    if off_days is not None:
        general.add_off_days_constraint(model, assign, day_roles, off_days)
    
    if on_days is not None:
        # FIX: OVERRIDE GRAYOUTS
        general.add_on_days_constraint(model, assign, day_roles, on_days, soft_grayouts)

    # -----------------------------------------------------
    # 5. Penalties (Soft Constraints)
    # -----------------------------------------------------

    # ======= Score Balance =======
    # Non NF Score Balance
    non_nf_balance_penalties = general.add_score_balance_soft_penalties(model, score_vars, non_nf_residents, max_points)
    helper.assert_penalties("non_nf_balance", non_nf_balance_penalties)
    
    # NF Score Balance
    nf_balance_penalties = general.add_score_balance_soft_penalties(model, score_vars, nf_residents, max_points)
    helper.assert_penalties("nf_balance", nf_balance_penalties)
    
    # Pre assigned dates (input from the user)
    fixed_preassigned = helper.build_fixed_preassigned(nf_calendar_df, ns_residents, wr_residents, preassigned_ns_df, preassigned_wr_df)   

    # NF minimum spacing
    spacing_nf_soft_penalties = general.add_minimum_spacing_soft_constraint(model, assign, days, day_roles, nf_residents, fixed_preassigned, min_gap=9)
    helper.assert_penalties("spacing_nf_soft_penalties", spacing_nf_soft_penalties)

    # NS minimum spacing
    spacing_ns_soft_penalties = general.add_minimum_spacing_soft_constraint(model, assign, days, day_roles, set(ns_residents['name']), fixed_preassigned, min_gap=7)
    helper.assert_penalties("spacing_ns_soft_penalties", spacing_ns_soft_penalties)

    # WR minimum spacing
    spacing_wr_soft_penalties = general.add_minimum_spacing_soft_constraint(model, assign, days, day_roles, set(wr_residents['name']), fixed_preassigned, min_gap=9)
    helper.assert_penalties("spacing_wr_soft_penalties", spacing_wr_soft_penalties)

    # The rest minimum spacing
    non_nf_for_spacing = [r for r in non_nf_residents if r not in set(ns_residents['name']) and r not in set(wr_residents['name'])]
    spacing_nonnf_soft_penalties = general.add_minimum_spacing_soft_constraint(model, assign, days, day_roles, non_nf_for_spacing, fixed_preassigned, min_gap=5)
    helper.assert_penalties("spacing_nonnf_soft_penalties", spacing_nonnf_soft_penalties)

    role_pref_penalties = []
    nf_day_pref_penalties = []
    
    if resident_year == "seniors":
        # Add soft preferences for R3/R4 role tendencies in weekends
        role_pref_penalties = general.add_role_preferences_by_level(
            model,
            assign,
            day_roles,
            days,
            residents,
            resident_levels)
        
    helper.assert_penalties("role_pref_penalties", role_pref_penalties)

    # elif resident_year == "r1":
    #     # Add preference for NF day shifts (avoid Tue/Thu) # FIX: THIS WILL GIVE ERROR SINCE NF ARE ALWAYS NS
    #     nf_day_pref_penalties = nf.add_nf_day_preferences_juniors(
    #         model,
    #         assign,
    #         day_roles,
    #         days,
    #         nf_residents)

    # Role diversity
    diverse_role_penalties = general.balanced_rotation_penalty(model, assign, days, day_roles, residents, ns_residents=ns_residents)
    helper.assert_penalties("diverse_role_penalties", diverse_role_penalties)

    # Day diversity
    diverse_day_penalties = general.balanced_day_penalty(model, assign, days, day_roles, residents)
    helper.assert_penalties("diverse_day_penalties", diverse_day_penalties)

    # 2 WR => No weekends (Usually for r1)
    wr_penalties = wr.add_wr_soft_constraints(model, assign, days, day_roles, wr_residents, weekend_days)
    helper.assert_penalties("wr_penalties", wr_penalties)
    
    # Maximum hard days count
    hard_max_penalties = general.hard_days_max_penalty(model, assign, days, day_roles, residents, max_hard=2)
    helper.assert_penalties("hard_max_penalties", hard_max_penalties)

    # -----------------------------------------------------
    # 6. Objective
    # -----------------------------------------------------

    #st.write(f"🔁 Attempt Solving")  

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 50
    solver.parameters.random_seed = random.randint(1, 1_000_000)
    solver.parameters.num_search_workers = 8
    #solver.parameters.use_lns = True
    solver.parameters.optimize_with_core = True
    solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH


    # PHASE 1: SCORE BALANCE + DAYS DIVERSITY + HARD DAYS. HARD FAIRNESS (NO TRADE-OFF)
    objective.minimize_and_fix(
        model,
        solver,
        sum(non_nf_balance_penalties) + sum(nf_balance_penalties)+
        sum(hard_max_penalties) +
        sum(diverse_day_penalties) +
        sum(diverse_role_penalties))

    # PHASE 2: SPACING
    objective.minimize_and_fix(
        model,
        solver,
        sum(spacing_ns_soft_penalties) +
        sum(spacing_wr_soft_penalties))
    
    model.Minimize(
        # model,
        # solver,
        sum(spacing_nonnf_soft_penalties) +
        sum(spacing_nf_soft_penalties) )

    # PHASE 3: PREFERENCE
    # model.Minimize(
    #     sum(role_pref_penalties) + sum(nf_day_pref_penalty))

    status = solver.Solve(model)

    actual_violations = sum(solver.Value(p) for p in non_nf_balance_penalties)
    print(f"Actual non_nf_balance_penalties violations in solution: {actual_violations}")

    actual_violations = sum(solver.Value(p) for p in nf_balance_penalties)
    print(f"Actual nf_balance_penalties violations in solution: {actual_violations}")

    actual_violations = sum(solver.Value(p) for p in spacing_nonnf_soft_penalties)
    print(f"Actual spacing_nonnf_soft_penalties violations in solution: {actual_violations}")

    actual_violations = sum(solver.Value(p) for p in spacing_nf_soft_penalties)
    print(f"Actual spacing_nf_soft_penalties violations in solution: {actual_violations}")

    actual_violations = sum(solver.Value(p) for p in spacing_ns_soft_penalties)
    print(f"Actual spacing_ns_soft_penalties violations in solution: {actual_violations}")

    actual_violations = sum(solver.Value(p) for p in spacing_wr_soft_penalties)
    print(f"Actual spacing_wr_soft_penalties violations in solution: {actual_violations}")

    actual_violations = sum(solver.Value(p) for p in role_pref_penalties)
    print(f"Actual role_pref_penalties violations in solution: {actual_violations}")

    actual_violations = sum(solver.Value(p) for p in nf_day_pref_penalty)
    print(f"Actual nf_day_pref_penalty violations in solution: {actual_violations}")

    actual_violations = sum(solver.Value(p) for p in hard_max_penalties)
    print(f"Actual hard_max_penalties violations in solution: {actual_violations}")

    actual_violations = sum(solver.Value(p) for p in diverse_day_penalties)
    print(f"Actual diverse_day_penalties violations in solution: {actual_violations}")

    actual_violations = sum(solver.Value(p) for p in diverse_role_penalties)
    print(f"Actual diverse_role_penalties violations in solution: {actual_violations}")

    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        #st.success("🏆 Optimal solution found")
        print (("🏆 Optimal solution found"))
        last_feasible_solver = solver

    elif status == cp_model.FEASIBLE:
        #st.warning("⚠ Non-optimal solution found (all HARD constraints satisfied)")
        print(("⚠ Non-optimal solution found (all HARD constraints satisfied)"))
        last_feasible_solver = solver
        nf_day_viol = new.count_residents_violating(solver, nf_day_pref_penalty, assign, days, day_roles, nf_residents)
        #st.write(f"NF Day preference violated by {len(nf_day_viol)} residents")
        print(f"NF Day preference violated by {len(nf_day_viol)} residents")

        role_pref_viol = new.count_residents_violating(solver, role_pref_penalties, assign, days, day_roles, residents)
        #st.write(f"Role preferences violated by {len(role_pref_viol)} residents")
        print((f"Role preferences violated by {len(role_pref_viol)} residents"))

        hard_max_viol = new.count_residents_violating(solver, hard_max_penalties, assign, days, day_roles, residents)
        #st.write(f"Hard days rules violated by {len(hard_max_viol)} residents")
        print((f"Hard days rules violated by {len(hard_max_viol)} residents"))

    elif status == cp_model.INFEASIBLE:
        st.error("❌ No feasible solution exists that satisfies hard constraints")
        print(("❌ No feasible solution exists that satisfies hard constraints"))

    elif status == cp_model.MODEL_INVALID:
        st.error("💥 Model is invalid — check variables or constraints")
        print(("💥 Model is invalid — check variables or constraints"))

    else:
        st.warning("⏹ Solver stopped early (time limit or search aborted)")
        print(("⏹ Solver stopped early (time limit or search aborted)"))
    # -----------------------------------------------------
    # 8. Extract results
    # -----------------------------------------------------
    
    schedule_df, scores_df = objective.extract_schedule(
        last_feasible_solver,
        assign,
        days,
        day_roles,
        residents,
        wr_residents,
        ns_residents,
        night_counts,
        wr_counts,
        score_vars,
        max_shifts,
        max_points,
        nf_calendar_df,
        resident_levels,
        limited_shift_residents)

    print("THE FINAL RETURN IS HAPPENNDNING:")
    print(schedule_df.head())
    return schedule_df, scores_df
