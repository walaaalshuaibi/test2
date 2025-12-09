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
                    print("started r1 ns.....")
                    assigned = assign_ns_juniors(date, candidates, blackout_lookup, nf_assigned, nf_blackout_lookup, nf_calendar_df)                                
                    
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

                    assigned_blackouts = blackout_lookup.get(assigned, set())
                    print(f"{assigned} blackout dates:", sorted(assigned_blackouts))

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


def assign_ns_juniors(date, candidates, blackout_lookup, nf_assigned, nf_blackout_lookup, nf_calendar_df):
    """
    Assign NS (Thursday night) for juniors based on NF blocks:
    - First half of NF residents (by NF start date) → NS = last NF date + 15
    - Second half → NS = first NF date - 8
    - Skips residents already assigned, blackout conflicts, and respects nf_blackout_lookup
    """
    import pandas as pd

    date = pd.Timestamp(date)

    # Collect all NF columns
    nf_cols_all = [c for c in nf_calendar_df.columns if "night" in c.lower()]

    # Build NF blocks per resident: {name: (nf_start, nf_end)}
    resident_blocks = {}
    for c in nf_cols_all:
        for _, row in nf_calendar_df.iterrows():
            name = row[c]
            if isinstance(name, str) and name.strip():
                name = name.strip()
                resident_blocks.setdefault(name, []).append(pd.Timestamp(row["date"]))

    # Convert lists to (start, end) blocks
    resident_blocks = {name: (min(dates), max(dates)) for name, dates in resident_blocks.items()}

    # Sort residents by NF start date
    sorted_residents = sorted(resident_blocks.items(), key=lambda x: x[1][0])
    total_residents = len(sorted_residents)
    half_index = total_residents // 2

    for i, (name, (nf_start, nf_end)) in enumerate(sorted_residents):
        already_assigned = name in nf_assigned
        first_half = i < half_index

        if already_assigned:
            print(f"Resident: '{name}' already assigned, skipping")
            continue

        # Determine candidate NS date
        if first_half:
            ns_candidate = nf_end + pd.Timedelta(days=15)
            half_label = "first-half"
        else:
            ns_candidate = nf_start - pd.Timedelta(days=8)
            half_label = "second-half"

        matches_target = ns_candidate.date() == date.date()
        blackout_conflict = date in blackout_lookup.get(name, set())
        nf_block_conflict = nf_blackout_lookup and (date in nf_blackout_lookup.get(name, set()))

        print(f"Resident: '{name}' | NF days: {(nf_end - nf_start).days + 1} | "
              f"already_assigned: {already_assigned}")
        print(f"  NF start={nf_start.date()}, NF end={nf_end.date()} | "
              f"Half: {half_label} -> ns_candidate={ns_candidate.date()} | "
              f"matches_target={matches_target} | blackout={blackout_conflict} | "
              f"nf_block_blackout={nf_block_conflict}")

        # Assign if matches and no conflicts
        if matches_target and not blackout_conflict and not nf_block_conflict:
            nf_assigned.add(name)
            print(f"  ✅ Assigned NS to '{name}' on {date.date()}")
            return name

    # ---------- ADD FALLBACK HERE ----------
    # If no NF resident was assigned, pick from candidates
    # 'candidates' should be a list of available residents (non-NF, not WR, not blacked out)
    for cand in candidates:
        if date not in blackout_lookup.get(cand, set()):
            nf_assigned.add(cand)
            print(f"  ⚠️ Fallback: assigning '{cand}' to {date.date()}")
            return cand
            
    print(f"  ⚠️ No NS assignment found for {date.date()}")
    return None
