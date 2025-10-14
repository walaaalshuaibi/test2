import pandas as pd
from datetime import datetime, timedelta
import re
import random

# ==========================================================
# ================== PREPROCESS DATA =======================
# ==========================================================

def rotate_list(lst, offset):
    return lst[offset % len(lst):] + lst[:offset % len(lst)] if lst else []

def extract_shift_columns():
    # Extract the columns needed
    nf_roles = ["ER-1 Night", "ER-2 Night", "EW Night"]
    day_roles = ["ER-1 Day", "ER-2 Day", "EW Day"]
    return nf_roles, day_roles

# Helper: expand date ranges from NF column
def expand_dates(date_range_str, base_year):
    if not date_range_str or date_range_str.strip().lower() == "no":
        return []
    
    # Normalize dashes and spacing
    clean_str = date_range_str.replace("–", "-").replace("—", "-").strip()
    clean_str = re.sub(r"\s*-\s*", "-", clean_str)
    
    # If single date
    if "-" not in clean_str:
        return [pd.Timestamp(datetime.strptime(clean_str + f" {base_year}", "%d %b %Y")).normalize()]
    
    # Split into start and end
    start_str, end_str = clean_str.split("-", 1)
    start_str = start_str.strip()
    end_str = end_str.strip()
    
    # Parse start
    try:
        start_date = datetime.strptime(start_str + f" {base_year}", "%d %b %Y")
    except ValueError:
        end_parts = end_str.split()
        if len(end_parts) == 2:
            start_date = datetime.strptime(start_str + " " + end_parts[1] + f" {base_year}", "%d %b %Y")
        else:
            raise
    
    # Infer end year
    try:
        end_date = datetime.strptime(end_str + f" {base_year}", "%d %b %Y")
        if end_date < start_date:
            end_date = datetime.strptime(end_str + f" {base_year + 1}", "%d %b %Y")
    except ValueError:
        start_parts = start_str.split()
        if len(start_parts) == 2:
            try:
                end_date = datetime.strptime(end_str + " " + start_parts[1] + f" {base_year}", "%d %b %Y")
                if end_date < start_date:
                    end_date = datetime.strptime(end_str + " " + start_parts[1] + f" {base_year + 1}", "%d %b %Y")
            except ValueError:
                raise
        else:
            raise
    
    # Expand full range
    return [pd.Timestamp(start_date + timedelta(days=i)).normalize() for i in range((end_date - start_date).days + 1)]

def build_nf_calendar(residents_df, start_date, nf_cols=None):
    """ 
    Build an NF calendar from residents_df and expand into NF1..NF{n_slots} columns.
    """
    
    if nf_cols is None:
        nf_cols = ["nf1", "nf2", "nf3"]
    
    # --- Step 1: Collect NF assignments per date ---
    calendar = {}
    for _, row in residents_df.iterrows():
        for d in expand_dates(row["nf"], base_year=start_date.year):
            # Ensure d is a pd.Timestamp
            d_ts = pd.Timestamp(d)
            calendar.setdefault(d_ts, []).append(row["name"])
    
    # --- Step 2: Determine full date range ---
    start_date = pd.Timestamp(start_date)
    last_date = max(calendar.keys()) if calendar else start_date
    last_date = pd.Timestamp(last_date)
    all_dates = [start_date + pd.Timedelta(days=i) for i in range((last_date - start_date).days + 1)]
    
    # --- Step 3: Build base NF calendar DataFrame ---
    base_calendar = pd.DataFrame({
        "day": [d.strftime("%a") for d in all_dates],
        "date": all_dates,  # keep as Timestamp
        "nf": [", ".join(calendar.get(d, [])) for d in all_dates]
    })
    
    # --- Step 4: Expand into NF1..NF{n_slots} with rotation ---
    calendar_data = []
    for day_idx, row in base_calendar.iterrows():
        names = row["nf"].split(", ") if row["nf"] else []
        rotated_names = rotate_list(names, day_idx)
        row_dict = {
            "day": row["day"],
            "date": row["date"]
        }
        for idx, col_name in enumerate(nf_cols):
            row_dict[col_name] = rotated_names[idx] if idx < len(rotated_names) else ""
        calendar_data.append(row_dict)
    
    return pd.DataFrame(calendar_data)

def build_blackout_lookup(blackout_df):
    blackout_dict = {}
    for _, row in blackout_df.iterrows():
        resident = row["resident"]
        date = pd.to_datetime(row["date"])
        blackout_dict.setdefault(resident, set()).add(date)
    return blackout_dict

def assign_ns_slot(date, available_residents, blackout_lookup, resident_level, role):
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

def fill_ns_cells(non_nf_residents, nf_calendar_df, wr_residents, resident_level, blackout_df=None, nf_cols=None):
    """ 
    Fill empty NF cells in the calendar with unused residents, ensuring they are not on vacation or blackout on that date.
    """
    
    if nf_cols is None:
        nf_cols = ["NF1", "NF2", "NF3"]
    
    # --- Step 1: Build resident pool ---
    available_residents = [r for r in non_nf_residents if r not in wr_residents]
    random.shuffle(available_residents)
    
    # --- Step 2: Build lookup sets for blackout ---
    blackout_lookup = build_blackout_lookup(blackout_df) if blackout_df is not None else {}
    
    # --- Step 3: Fill missing NF slots ---
    ns_records = []
    for col in nf_cols:
        for idx, val in nf_calendar_df[col].items():
            if val == "" and available_residents:
                date = pd.to_datetime(nf_calendar_df.at[idx, "date"])
                assigned = assign_ns_slot(date, available_residents, blackout_lookup, resident_level, col)
                if assigned:
                    resident = assigned
                    nf_calendar_df.at[idx, col] = resident
                    ns_records.append({"date": date, "resident": resident, "role": col})
    
    # --- Step 4: Build NS residents DataFrame ---
    ns_residents = pd.DataFrame(ns_records)
    return nf_calendar_df, ns_residents, blackout_lookup

def update_blackout(resident, dates, blackout_dict, buffer_days=0, exclude_dates=None, record_list=None):
    """ 
    Adds blackout dates for a resident with optional buffer and exclusions. 
    Args:
        resident (str): Resident name.
        dates (list): List of central dates.
        blackout_dict (dict): Dict to update.
        buffer_days (int): Days before/after each date to include.
        exclude_dates (set): Dates to exclude from blackout.
        record_list (list): Optional list to append blackout records.
    """
    
    for d in dates:
        if buffer_days > 0:
            blackout_range = pd.date_range(d - pd.Timedelta(days=buffer_days), d + pd.Timedelta(days=buffer_days))
        else:
            blackout_range = pd.DatetimeIndex([d])
        
        if exclude_dates:
            blackout_range = blackout_range.difference(pd.DatetimeIndex(exclude_dates))
        
        blackout_dict.setdefault(resident, set()).update(blackout_range)
        
        if record_list is not None:
            record_list.append({"resident": resident, "date": d})

def build_combined_blackout_df(*blackout_dicts):
    combined = {}
    for bd in blackout_dicts:
        for r, dates in bd.items():
            combined.setdefault(r, set()).update(dates)
    
    return pd.DataFrame(
        [{"resident": r, "date": d} for r, dates in combined.items() for d in dates]
    )