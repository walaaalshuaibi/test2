from datetime import datetime, timedelta
import re
import pandas as pd
import random

def resident_score_expr(assign, days, day_roles, resident, weekend_days, weekend_rounds_df=None, ns_residents_df=None):
    """ 
    Build the linear expression for a resident's score:
    - 1 point per weekday shift
    - 2 points per weekend shift
    - +2 bonus per weekend rounds 
    """
    
    # Base score: weighted by weekday/weekend
    expr = sum(
        assign[(d, role, resident)] * (2 if d in weekend_days else 1) 
        for d in days for role in day_roles
    )
    
    # Weekend round bonus: +2 per WR date
    if weekend_rounds_df is not None and not weekend_rounds_df.empty:
        wr_dates = weekend_rounds_df.loc[
            weekend_rounds_df["name"].str.strip() == resident, "date"
        ]
        expr += 2 * len(wr_dates)
    
    # NS bonus
    if ns_residents_df is not None and not ns_residents_df.empty:
        ns_residents = set(ns_residents_df["name"].str.strip())
        if resident in ns_residents:
            expr += 2
            
    return expr

# ==========================================================
# ================== PREPROCESS DATA =======================
# ==========================================================

def rotate_list(lst, offset):
    return lst[offset % len(lst):] + lst[:offset % len(lst)] if lst else []

# Helper: expand date ranges from NF column
def expand_dates(date_range_str, base_year):
    """
    Expand strings like:
      - "01 Dec"
      - "28-29 Nov"
      - "30 Nov - 01 Dec"
      - "28-29 Nov and 12-13 Dec"
      - "01- 02 Dec"
    into a list of normalized pd.Timestamp dates, inferring months across ranges.

    Rules:
    - Split only on 'and' (case-insensitive). Commas are not treated as separators here.
    - If the start lacks a month (e.g., '28-29 Nov'), borrow the month from the end part ('Nov').
    - If the end lacks a month (rare), borrow from the start part.
    - Handle year rollover when end < start (e.g., '30 Dec - 02 Jan').
    """
    if not date_range_str or str(date_range_str).strip().lower() == "no":
        return []

    s = str(date_range_str)

    # Normalize dash variants and spacing around dashes
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s*-\s*", "-", s).strip()

    # Split only on 'and' to support multiple ranges
    parts = re.split(r"\band\b", s, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]

    dates_out = []

    for part in parts:
        # Case 1: single date (no dash)
        if "-" not in part:
            dt = datetime.strptime(f"{part} {base_year}", "%d %b %Y")
            dates_out.append(pd.Timestamp(dt).normalize())
            continue

        # Case 2: range with dash
        start_str, end_str = part.split("-", 1)
        start_str = start_str.strip()
        end_str = end_str.strip()

        # Helper: does token contain both day and month?
        def has_month(token):
            return len(token.split()) >= 2

        # Parse start, possibly borrowing month from end
        try:
            if has_month(start_str):
                start_date = datetime.strptime(f"{start_str} {base_year}", "%d %b %Y")
            else:
                # Borrow month from end if present (e.g., '28-29 Nov')
                end_parts = end_str.split()
                if len(end_parts) >= 2:
                    start_date = datetime.strptime(f"{start_str} {end_parts[-1]} {base_year}", "%d %b %Y")
                else:
                    # Last resort: assume same month as end’s numeric day (unlikely)
                    start_date = datetime.strptime(f"{start_str} {base_year}", "%d %b %Y")
        except ValueError:
            # Try adding month from end more explicitly
            end_parts = end_str.split()
            if len(end_parts) >= 2:
                start_date = datetime.strptime(f"{start_str} {end_parts[-1]} {base_year}", "%d %b %Y")
            else:
                raise

        # Parse end, possibly borrowing month from start
        try:
            if has_month(end_str):
                end_date = datetime.strptime(f"{end_str} {base_year}", "%d %b %Y")
            else:
                start_parts = start_str.split()
                if len(start_parts) >= 2:
                    end_date = datetime.strptime(f"{end_str} {start_parts[-1]} {base_year}", "%d %b %Y")
                else:
                    end_date = datetime.strptime(f"{end_str} {base_year}", "%d %b %Y")
            # Year rollover if end < start (e.g., '30 Dec - 01 Jan')
            if end_date < start_date:
                # If end had explicit month Jan, this’s expected; add a year
                end_date = datetime.strptime(end_date.strftime("%d %b") + f" {base_year + 1}", "%d %b %Y")
        except ValueError:
            # Borrow month from start explicitly
            start_parts = start_str.split()
            if len(start_parts) >= 2:
                end_date = datetime.strptime(f"{end_str} {start_parts[-1]} {base_year}", "%d %b %Y")
                if end_date < start_date:
                    end_date = datetime.strptime(end_date.strftime("%d %b") + f" {base_year + 1}", "%d %b %Y")
            else:
                raise

        # Expand inclusive range
        span_days = (end_date - start_date).days
        for i in range(span_days + 1):
            dates_out.append(pd.Timestamp(start_date + timedelta(days=i)).normalize())

    return dates_out

def calculate_max_limits(residents, nf_residents,  resident_max_limit, nf_max_limit ):
    """
    Calculate maximum shifts, points, and weekend limits for each resident.
    """
    max_shifts, max_points, weekend_limits = {}, {}, {}

    for resident in residents:    
        if resident in nf_residents:        # night floaters
            shifts, points, weekend = (nf_max_limit)
        else: 
            shifts, points, weekend = (resident_max_limit)
            
        # Define Max Shifts, Max Points, Max Weekends
        max_shifts[resident] = shifts
        max_points[resident] = points
        weekend_limits[resident] = weekend

    return max_shifts, max_points, weekend_limits

def build_fixed_preassigned(nf_calendar_df,
                            ns_residents,
                            weekend_rounds_df,
                            preassigned_ns_df=None,
                            preassigned_wr_df=None):
    """
    Build a dictionary of ALL preassigned shifts for spacing penalties.
    Output:
        { resident_name : set([date1, date2, ...]) }
    """

    fixed = {}

    # ---- 1. NF Calendar (NF residents have fixed days) ----
    for _, row in nf_calendar_df.iterrows():
        for role_col in nf_calendar_df.columns:
            if role_col.lower() in ["date", "day", "weekday"]:
                continue
            r = row[role_col]
            if pd.isna(r):
                continue
            fixed.setdefault(r.strip(), set()).add(row["date"])

    # ---- 2. NS (night shifts that are pre-filled) ----
    for _, row in ns_residents.iterrows():
        r = row["name"].strip()
        d = row["date"]
        fixed.setdefault(r, set()).add(d)

    # ---- 3. WR Weekend Rounders ----
    for _, row in weekend_rounds_df.iterrows():
        r = row["name"].strip()
        d = row["date"]
        fixed.setdefault(r, set()).add(d)

    # ---- 4. Excel Preassigned NS ----
    if preassigned_ns_df is not None:
        for _, row in preassigned_ns_df.iterrows():
            r = row["name"].strip()
            d = row["date"]
            fixed.setdefault(r, set()).add(d)

    # ---- 5. Excel Preassigned WR ----
    if preassigned_wr_df is not None:
        for _, row in preassigned_wr_df.iterrows():
            r = row["name"].strip()
            d = row["date"]
            fixed.setdefault(r, set()).add(d)

    return fixed

def get_all_assigned_dates(
    r,
    solver,
    assign,
    days,
    roles,
    ns_df=None,
    wr_df=None,
    nf_calendar_df=None,
    extra_preassigned=None
):
    """
    Collect ALL assigned dates for a resident from:
        - OR-Tools solver assignments
        - NS preassignments (ns_df)
        - WR preassignments (wr_df)
        - NF calendar (nf_calendar_df)
        - extra_preassigned: list/dict of additional custom assignments

    Returns:
        A sorted list of unique datetime.date objects.
    """

    all_dates = set()

    # ----------------------------------------------------
    # 1) OR-TOOLS ASSIGNED SHIFTS
    # ----------------------------------------------------
    for d in days:
        date_val = d.date() if hasattr(d, "date") else d
        for role in roles:
            try:
                if solver.Value(assign[(d, role, r)]) == 1:
                    all_dates.add(date_val)
            except Exception:
                pass

    # ----------------------------------------------------
    # 2) NS PREASSIGNMENTS
    # ----------------------------------------------------
    if ns_df is not None and isinstance(ns_df, pd.DataFrame) and not ns_df.empty:
        if "name" in ns_df.columns and "date" in ns_df.columns:
            ns_dates = ns_df.loc[ns_df["name"] == r, "date"]
            for d in ns_dates:
                all_dates.add(pd.to_datetime(d).date())

    # ----------------------------------------------------
    # 3) WR PREASSIGNMENTS
    # ----------------------------------------------------
    if wr_df is not None and isinstance(wr_df, pd.DataFrame) and not wr_df.empty:
        if "name" in wr_df.columns and "date" in wr_df.columns:
            wr_dates = wr_df.loc[wr_df["name"] == r, "date"]
            for d in wr_dates:
                all_dates.add(pd.to_datetime(d).date())

    # ----------------------------------------------------
    # 4) NF CALENDAR (night float)
    # ----------------------------------------------------
    if nf_calendar_df is not None and isinstance(nf_calendar_df, pd.DataFrame) and not nf_calendar_df.empty:
        if "name" in nf_calendar_df.columns and "date" in nf_calendar_df.columns:
            nf_dates = nf_calendar_df.loc[nf_calendar_df["name"] == r, "date"]
            for d in nf_dates:
                all_dates.add(pd.to_datetime(d).date())

    # ----------------------------------------------------
    # 5) EXTRA CUSTOM PREASSIGNMENTS
    # ----------------------------------------------------
    if extra_preassigned:
        # expects either dict: {resident: [dates]}
        # OR list of (resident, date)
        if isinstance(extra_preassigned, dict):
            for d in extra_preassigned.get(r, []):
                all_dates.add(pd.to_datetime(d).date())

        elif isinstance(extra_preassigned, list):
            for item in extra_preassigned:
                if isinstance(item, tuple) and len(item) == 2:
                    name, d = item
                    if name == r:
                        all_dates.add(pd.to_datetime(d).date())

    # ----------------------------------------------------
    # Return sorted list
    # ----------------------------------------------------
    return sorted(all_dates)
