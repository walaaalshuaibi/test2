import pandas as pd
import random
import blackout
import streamlit as st

def fill_ns_cells(
        resident_year,
        non_nf_residents, 
        nf_residents,
        nf_calendar_df, 
        wr_residents, 
        resident_level, 
        blackout_df, 
        soft_grayouts,
        nf_cols,
        preassigned_ns_df=None):
    """ 
    Fill empty NF cells in the calendar with unused residents, ensuring they are not on vacation or blackout on that date.
    """

    if nf_cols is None:
        nf_cols = ["NF1", "NF2", "NF3"]

    # Step 1: Build resident pool
    available_residents = [r for r in non_nf_residents if r not in set(wr_residents["name"])]
    random.shuffle(available_residents)

    ns_records = []
    nf_assigned = set()

    # Precompute sorted NF residents for juniors (done only once)
    sorted_juniors = None
    if resident_year == "r1" and nf_residents:
        # Build NF blocks per resident from nf_calendar_df filtered by nf_residents
        resident_blocks = {}
        for c in nf_cols:
            for _, row in nf_calendar_df.iterrows():
                name = row[c]
                if isinstance(name, str) and name.strip() and name in nf_residents:
                    name = name.strip()
                    resident_blocks.setdefault(name, []).append(pd.Timestamp(row["date"]))

        # Convert lists to (start, end) blocks
        resident_blocks = {name: (min(dates), max(dates)) for name, dates in resident_blocks.items()}

        # Sort residents by NF start date
        sorted_juniors = sorted(resident_blocks.items(), key=lambda x: x[1][0])

    # Step 3a: Fill preassigned slots first
    if preassigned_ns_df is not None and not preassigned_ns_df.empty:
        for _, row in preassigned_ns_df.iterrows():
            date = pd.to_datetime(row["date"])
            resident = row["name"].strip()
            role = row["role"].strip().lower()

            if date in blackout_df.get(resident, set()):
                st.warning(f"⚠️ Skipping {resident} on {date.date()} — blackout conflict")
                continue

            mask = pd.to_datetime(nf_calendar_df["date"]) == date
            if mask.any() and role in nf_calendar_df.columns:
                idx = nf_calendar_df.index[mask][0]
                nf_calendar_df.at[idx, role] = resident
                ns_records.append({"date": date, "name": resident, "role": role})
                blackout.update_blackout(resident, [date], blackout_df)
                if resident in available_residents:
                    available_residents.remove(resident)

    # Step 3b: Fill remaining empty NF slots
    for col in nf_cols:
        for idx, val in nf_calendar_df[col].items():
            if val == "" and available_residents:
                date = pd.to_datetime(nf_calendar_df.at[idx, "date"])

                candidates = [r for r in available_residents if date not in blackout_df.get(r, set())]
                if not candidates:
                    st.warning(f"⚠️ No available residents for {date.date()} in {col} (all blacked out)")
                    continue

                if resident_year == "r1":
                    print("THIS IS JUNIORS")
                    print(sorted_juniors)
                    assigned = assign_ns_juniors(
                        date, candidates, blackout_df, nf_assigned, sorted_juniors)                                
                    
                elif resident_year == "seniors":
                    assigned = assign_ns_seniors(date, candidates, blackout_df, soft_grayouts, resident_level, col)

                else:
                    assigned = None

                if assigned:
                    resident = assigned
                    nf_calendar_df.at[idx, col] = resident
                    ns_records.append({"date": date, "name": resident, "role": col})
                    blackout.update_blackout(resident, [date], blackout_df)
                    if resident in available_residents:
                        available_residents.remove(resident)

                    assigned_blackouts = blackout_df.get(assigned, set())
                    print(f"{assigned} blackout dates:", sorted(assigned_blackouts))

    ns_residents = pd.DataFrame(ns_records)
    return nf_calendar_df, ns_residents, blackout_df

def assign_ns_seniors(date, available_residents, blackout_lookup, soft_grayouts, resident_level, role):
    """ 
    Assign one resident to NS slot on a given date, with weighted preference:
    - R4s more likely for ER1/ER2
    - R3s more likely for EW 
    """
    
    # Filter out blacked-out residents
    candidates = [
        r for r in available_residents
        if date not in blackout_lookup.get(r, set())
        and date not in soft_grayouts.get(r, set())]
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

def assign_ns_juniors(date, candidates, blackout_lookup, nf_assigned, sorted_residents):
    """
    Assign NS (Thursday night) for juniors based on precomputed NF blocks
    """
    import pandas as pd
    date = pd.Timestamp(date)

    total_residents = len(sorted_residents)
    half_index = total_residents // 2

    for i, (name, (nf_start, nf_end)) in enumerate(sorted_residents):
        already_assigned = name in nf_assigned
        first_half = i < half_index

        if already_assigned:
            print(f"Resident: '{name}' already assigned, skipping")
            continue

        if first_half:
            ns_candidate = nf_end + pd.Timedelta(days=15)
            half_label = "first-half"
        else:
            ns_candidate = nf_start - pd.Timedelta(days=8)
            half_label = "second-half"

        matches_target = ns_candidate.date() == date.date()
        blackout_conflict = date in blackout_lookup.get(name, set())

        print(f"Resident: '{name}' | NF days: {(nf_end - nf_start).days + 1} | "
              f"already_assigned: {already_assigned}")
        print(f"  NF start={nf_start.date()}, NF end={nf_end.date()} | "
              f"Half: {half_label} -> ns_candidate={ns_candidate.date()} | "
              f"matches_target={matches_target} | blackout={blackout_conflict} | ")

        if matches_target and not blackout_conflict:
            nf_assigned.add(name)
            print(f"  ✅ Assigned NS to '{name}' on {date.date()}")
            return name

    for cand in candidates:
        if date not in blackout_lookup.get(cand, set()):
            nf_assigned.add(cand)
            print(f"  ⚠️ Fallback: assigning '{cand}' to {date.date()}")
            return cand
            
    print(f"  ⚠️ No NS assignment found for {date.date()}")
    return None

def night_thursday_hard_constraint(model, assign, days, roles, ns_residents):
    """
    Hard constraint: Night shifters can be assigned to at most 1 Thursday.
    """

    # Filter Thursdays
    thursdays = [d for d in days if d.strftime('%a') == 'Thu']

    for r in ns_residents:
        # All Thursday assignments for this resident
        flags = [
            assign[(d, role, r)]
            for d in thursdays
            for role in roles
            if (d, role, r) in assign
        ]

        if flags:
            # HARD constraint: at most 1 Thursday
            model.Add(sum(flags) <= 1)

