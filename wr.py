import datetime
import pandas as pd
import helper
def add_weekend_rounds_constraint(
    model,
    assign,
    roles,
    weekend_rounds_df,
    resident_year,
    preassigned_wr_df=None,
    combined_blackout_dict=None,
    soft_grayouts=None
):
    """ 
    Enforce weekend round assignments with blackout/grayout overrides.
    
    - Preassigned WR residents are forced into their role/date and conflicts removed.
    - Seniors are assigned to 'EW Day' on their weekend round dates.
    - All assignments remove conflicting blackouts and grayouts.
    - Prevents other residents from taking the same date-role.
    """

    # Normalize dates
    weekend_rounds_df["date"] = pd.to_datetime(weekend_rounds_df["date"]).dt.normalize()

    # --- STEP 1: Handle preassigned WR residents ---
    if preassigned_wr_df is not None and not preassigned_wr_df.empty:
        preassigned_wr_df["date"] = pd.to_datetime(preassigned_wr_df["date"]).dt.normalize()

        for _, row in preassigned_wr_df.iterrows():
            resident = row["name"].strip()
            date = pd.Timestamp(row["date"]).normalize()
            role = str(row["role"]).strip().lower()

            # Remove blackout/grayout conflicts
            for coll, coll_name in [(combined_blackout_dict, "blackout"), (soft_grayouts, "soft grayout")]:
                if coll is None:
                    continue
                if isinstance(coll, pd.DataFrame):
                    mask = ~((coll["name"].str.strip() == resident) & (coll["date"] == date))
                    removed = len(coll) - mask.sum()
                    coll[:] = coll.loc[mask]
                    if removed > 0:
                        print(f"🟢 Removed {coll_name} for {resident} on {date.date()} (preassigned {role})")
                elif isinstance(coll, dict) and resident in coll:
                    before_len = len(coll[resident])
                    coll[resident] = [d for d in coll[resident] if pd.to_datetime(d).normalize() != date]
                    after_len = len(coll[resident])
                    if before_len != after_len:
                        print(f"🟢 Removed {coll_name} (dict) for {resident} on {date.date()} (preassigned {role})")

            # Defensive role check
            if role not in roles:
                print(f"⚠️ Skipping preassigned WR for {resident}: role '{role}' not in roles list.")
                continue

            # Force assignment
            model.Add(assign[(date, role, resident)] == 1)

            # Prevent anyone else from taking that same date-role
            for (d, r, other) in assign.keys():
                if d == date and r == role and other != resident:
                    model.Add(assign[(d, r, other)] == 0)

    # --- STEP 2: Assign seniors to EW Day ---
    if resident_year == "seniors":
        for _, row in weekend_rounds_df.iterrows():
            resident = row["name"].strip()
            date = pd.Timestamp(row["date"]).normalize()
            role = "ew day"

            if role not in roles:
                raise ValueError("role 'ew day' not found in roles list.")

            # Remove blackout/grayout conflicts for senior
            for coll, coll_name in [(combined_blackout_dict, "blackout"), (soft_grayouts, "soft grayout")]:
                if coll is None:
                    continue
                if isinstance(coll, pd.DataFrame):
                    mask = ~((coll["name"].str.strip() == resident) & (coll["date"] == date))
                    removed = len(coll) - mask.sum()
                    coll[:] = coll.loc[mask]
                    if removed > 0:
                        print(f"🟢 Removed {coll_name} for {resident} on {date.date()} (senior EW assignment)")
                elif isinstance(coll, dict) and resident in coll:
                    before_len = len(coll[resident])
                    coll[resident] = [d for d in coll[resident] if pd.to_datetime(d).normalize() != date]
                    after_len = len(coll[resident])
                    if before_len != after_len:
                        print(f"🟢 Removed {coll_name} (dict) for {resident} on {date.date()} (senior EW assignment)")

            # Force assignment
            model.Add(assign[(date, role, resident)] == 1)

            # Prevent anyone else from taking that same date-role
            for (d, r, other) in assign.keys():
                if d == date and r == role and other != resident:
                    model.Add(assign[(d, r, other)] == 0)
        
def build_weekend_round_assignments(residents_df, start_date, resident_year, r2_cover):
    """
    Build weekend round assignments (no blackout logic).

    Rules:
    - Each resident may have multiple weekend ranges (e.g., "28–29 Nov" and "12–13 Dec").
    - For each 2-day range, distribute residents across the two days:
        * If even number of residents: split evenly.
        * If odd number: extra goes to the first day.
    - Seniors: only get Friday WR (skip Saturday).
    Returns:
        list of (resident_name, wr_date)
    """
    assignments = []

    # --- If R2 is covering in R1 
    if r2_cover is not None:
        resident_name = r2_cover[0]
        date = r2_cover[1]
        assignments.append((resident_name, date))

    # --- Collect all valid date ranges per resident ---
    all_ranges = []  # list of (resident_name, [dates in one range])
    for _, row in residents_df.iterrows():
        resident_name = row["name"].strip()
        wr_field = str(row.get("weekend round", "")).strip().lower()

        if not wr_field or wr_field == "no":
            continue

        # Handle multi-line weekend ranges
        for rng in wr_field.split(" and "):
            dates = helper.expand_dates(rng, base_year=start_date.year, anchor_month=start_date.month)
            if not dates:
                continue

            dates = sorted(pd.Timestamp(d).normalize() for d in dates)
            all_ranges.append((resident_name, dates))

    # --- Group by each distinct range of dates ---
    range_pool = {}
    for resident_name, dates in all_ranges:
        key = tuple(dates)
        range_pool.setdefault(key, []).append(resident_name)

    # --- Assign residents per date range ---
    for dates, residents in range_pool.items():
        dates = sorted(dates)

        # 1-day weekend (rare)
        if len(dates) == 1:
            d = dates[0]
            if resident_year.lower() == "seniors" and d.day_name() != "Friday":
                continue
            for r in residents:
                assignments.append((r, d))
            continue

        # 2-day weekend (Fri/Sat typical)
        if len(dates) == 2:
            d1, d2 = dates

            if resident_year.lower() == "seniors":
                # Seniors only get Friday WR
                for r in residents:
                    if d1.day_name() == "Friday":
                        assignments.append((r, d1))
                continue

            # Distribute juniors across the two days
            n = len(residents)
            half = n // 2
            remainder = n % 2

            # First half (+remainder) -> d1, rest -> d2
            for r in residents[: half + remainder]:
                assignments.append((r, d2))
            for r in residents[half + remainder:]:
                assignments.append((r, d1))
            continue

        # Longer ranges (>2 days) — assign everyone to first day
        for r in residents:
            assignments.append((r, dates[0]))

    return assignments

def add_wr_soft_constraints(model, assign, days, roles, weekend_rounds_df, weekend_days):
    """
    Soft constraint: if a resident already has 2 WR dates in weekend_rounds_df,
    discourage assigning them to any additional weekend shifts (Fri, Sat),
    and also discourage Thursdays separately.
    """
    penalties = []

    # Count WR dates per resident
    wr_counts = weekend_rounds_df.groupby("name")["date"].nunique().to_dict()

    for resident, count in wr_counts.items():
        if count >= 2:
            for d in days:
                # Case 1: weekend days (Fri, Sat)
                if d in weekend_days:
                    for role in roles:
                        if (d, role, resident) in assign:
                            var = assign[(d, role, resident)]
                            penalty_var = model.NewIntVar(0, 1, f"wr_penalty_{resident}_{d}_{role}")
                            model.Add(penalty_var == var)
                            penalties.append(penalty_var)

    return penalties

def add_wr_hard_constraints(
    model,
    assign,
    days,
    roles,
    preassigned_wr_df,
    weekend_days
):
    """
    HARD CONSTRAINT:
    - Always forbid weekend assignments (Fri/Sat)
    - EXCEPT:
        If a (resident, date, role) is preassigned as WR,
        that exact assignment is allowed.
    """

    # Build allowed WR assignments (exceptions)
    allowed_wr = set()

    if preassigned_wr_df is not None and not preassigned_wr_df.empty:
        wr_df = preassigned_wr_df.copy()
        wr_df["date"] = pd.to_datetime(wr_df["date"]).dt.date

        # Expecting columns: name, date, role
        allowed_wr = {
            (row.date, row.role, row.name)
            for row in wr_df.itertuples(index=False)
        }

    # Enforce hard constraint
    for d in days:
        d_date = pd.to_datetime(d).date()

        if d_date in weekend_days:  # Fri / Sat
            for role in roles:
                for resident in {r for (_, _, r) in assign.keys()}:
                    key = (d, role, resident)
                    if key not in assign:
                        continue

                    # Allow only explicitly preassigned WR
                    if (d_date, role, resident) not in allowed_wr:
                        model.Add(assign[key] == 0)

