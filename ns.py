import pandas as pd
import random
import blackout
import helper

def fill_ns_cells(
    resident_year,
    non_nf_residents,
    nf_residents,
    nf_max_limit,
    nf_calendar_df,
    wr_residents,
    resident_level,
    blackout_df=None,
    nf_cols=None,
    nf_blackout_lookup=None,
    preassigned_ns_df=None
):
    """ 
    Fill empty NF cells in the calendar with unused residents, ensuring they are not on vacation or blackout on that date.
    """
    import pandas as pd
    import random
    import blackout

    if nf_cols is None:
        nf_cols = ["NF1", "NF2", "NF3"]

    # Step 1: Build resident pool
    available_residents = [r for r in non_nf_residents if r not in wr_residents]
    random.shuffle(available_residents)

    # Step 2: Build blackout lookup
    blackout_lookup = blackout.build_blackout_lookup(blackout_df) if blackout_df is not None else {}

    ns_records = []
    nf_assigned = set()

    # Step 3a: Fill preassigned slots first
    if preassigned_ns_df is not None and not preassigned_ns_df.empty:
        for _, row in preassigned_ns_df.iterrows():
            date = pd.to_datetime(row["date"])
            resident = row["name"].strip()
            role = row["role"].strip().lower()

            # ✅ skip if blackout on that date
            if date in blackout_lookup.get(resident, set()):
                print(f"⚠️ Skipping {resident} on {date.date()} — blackout conflict")
                continue

            # Fill preassigned
            mask = pd.to_datetime(nf_calendar_df["date"]) == date
            if mask.any() and role in nf_calendar_df.columns:
                idx = nf_calendar_df.index[mask][0]
                nf_calendar_df.at[idx, role] = resident
                ns_records.append({"date": date, "name": resident, "role": role})
                blackout.update_blackout(resident, [date], blackout_lookup)
                if resident in available_residents:
                    available_residents.remove(resident)

    # Step 3b: Fill remaining empty NF slots 
    for col in nf_cols:
        for idx, val in nf_calendar_df[col].items():
            if val == "" and available_residents:
                date = pd.to_datetime(nf_calendar_df.at[idx, "date"])

                # ✅ filter out residents with blackout on that date
                candidates = [r for r in available_residents if date not in blackout_lookup.get(r, set())]
                if not candidates:
                    print(f"⚠️ No available residents for {date.date()} in {col} (all blacked out)")
                    continue

                if resident_year == "r1":
                    assigned = assign_ns_juniors(
                        date, available_residents, blackout_lookup, nf_residents=nf_residents, nf_max_limit=nf_max_limit, 
                        nf_assigned=nf_assigned, nf_blackout_lookup=nf_blackout_lookup, wr_residents=wr_residents)
                elif resident_year == "seniors":
                    assigned = assign_ns_seniors(date, candidates, blackout_lookup, resident_level, col)
                else:
                    assigned = None

                if assigned:
                    resident = assigned
                    nf_calendar_df.at[idx, col] = resident
                    ns_records.append({"date": date, "name": resident, "role": col})
                    blackout.update_blackout(resident, [date], blackout_lookup)
                    if resident in available_residents:
                        available_residents.remove(resident)

    ns_residents = pd.DataFrame(ns_records)
    return nf_calendar_df, ns_residents, blackout_lookup

def assign_ns_seniors(date, available_residents, blackout_lookup, resident_level, role):
    """ 
    Assign one resident to NS slot on a given date, with weighted preference:
    - R4s more likely for ER1/ER2
    - R3s more likely for EW 
    """
    
    # Filter out blacked-out residents
    candidates = [r for r in available_residents if (r, date) not in blackout_lookup]
    if not candidates:
        return None
    
    # Build weights
    weights = []
    for r in candidates:
        level = resident_level.get(r, "").lower()
        # ER1/ER2 → bias toward R4
        if role in ["er-1 night", "er-2 night"] and level == "r4":
            weights.append(3)
        elif role in ["er-1 night", "er-2 night"] and level == "r3":
            weights.append(1)
        # EW → bias toward R3
        elif role == "ew night" and level == "r3":
            weights.append(3)
        elif role == "ew night" and level == "r4":
            weights.append(1)
        # fallback equal weight
        else:
            weights.append(1)
    
    # Weighted random choice
    chosen = random.choices(candidates, weights=weights, k=1)[0]

    # Remove the chosen resident from available_residents
    if chosen in available_residents:
        available_residents.remove(chosen)

    return chosen

def assign_ns_juniors(date, available_residents, blackout_lookup, nf_residents=None, 
                      nf_max_limit=None, nf_assigned=None, nf_blackout_lookup=None, wr_residents=None):
    """Assign one resident to NS, respecting NF blackout + previous/next NS days."""

    if nf_assigned is None:
        nf_assigned = set()

    print("WR RESIDENTS")
    for r in nf_residents:
        print(r)

    # Extend available_residents with NF residents if allowed
    if nf_residents and nf_max_limit and nf_max_limit[0] > 0:
        for r in nf_residents:
            if r not in available_residents and r not in wr_residents:
                available_residents.append(r)

    # --- Filter candidates ---
    candidates = []
    for r in available_residents:
        blocked = (r, date) in blackout_lookup
        nf_blocked = nf_blackout_lookup.get((r, date), False) if nf_blackout_lookup else False

        # Optional: check previous and next dates
        prev_day = date - pd.Timedelta(days=1)
        next_day = date + pd.Timedelta(days=1)
        adjacent_blocked = ((r, prev_day) in blackout_lookup or
                            (r, next_day) in blackout_lookup or
                            (r, prev_day) in nf_blackout_lookup or
                            (r, next_day) in nf_blackout_lookup)

        if not blocked and not nf_blocked and not adjacent_blocked:
            candidates.append(r)

    if not candidates:
        return None

    # --- Weighted random choice ---
    weights = []
    for r in candidates:
        base_weight = 3
        if nf_residents and r in nf_residents and r not in nf_assigned:
            base_weight *= 5
        weights.append(base_weight)

    chosen = random.choices(candidates, weights=weights, k=1)[0]

    # --- Bookkeeping ---
    available_residents.remove(chosen)
    if nf_residents and chosen in nf_residents:
        nf_assigned.add(chosen)

    # --- Update NS blackout to prevent adjacent assignments ---
    if nf_blackout_lookup is not None:
        prev_day = date - pd.Timedelta(days=1)
        next_day = date + pd.Timedelta(days=1)
        nf_blackout_lookup[(chosen, prev_day)] = True
        nf_blackout_lookup[(chosen, next_day)] = True

    return chosen


