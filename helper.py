from datetime import datetime, timedelta
import re
import pandas as pd
import random

def resident_score_expr(assign, days, roles, resident, weekend_days, weekend_rounds_df=None, ns_residents_df=None):
    """ 
    Build the linear expression for a resident's score:
    - 1 point per weekday shift
    - 2 points per weekend shift
    - +2 bonus per weekend rounds 
    """
    
    # Base score: weighted by weekday/weekend
    expr = sum(
        assign[(d, role, resident)] * (2 if d in weekend_days else 1) 
        for d in days for role in roles
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

def calculate_max_limits(residents, nf_residents, wr_residents, night_counts, resident_max_limit, nf_max_limit ):
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
