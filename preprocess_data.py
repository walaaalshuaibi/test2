import pandas as pd
import helper

# -----------------------------------------------------
def prepare_data(residents_df, start_date, num_weeks, resident_year):
    """ 
    Prepare scheduling inputs from the residents DataFrame. 
    """
    
    # Residents list
    residents = residents_df['name'].tolist()
    
    # R4 vs R3 resident_levels
    resident_levels = dict(zip(residents_df["name"], residents_df["level"]))
    
    # Days in scheduling period
    start_date = pd.to_datetime(start_date)
    days = pd.date_range(start_date, start_date + pd.Timedelta(weeks=num_weeks) - pd.Timedelta(days=1))
    
    weekend_days = set(d for d in days if d.strftime('%a') in ['Fri', 'Sat'])
    
    # Roles
    nf_roles, day_roles = helper.extract_shift_columns()
    nf_roles = [r.strip().lower() for r in nf_roles]
    day_roles = [r.strip().lower() for r in day_roles]
    
    # NF Calendar
    nf_calendar_df = helper.build_nf_calendar(residents_df, start_date, nf_cols=nf_roles)
    
    # Initialize blackouts and counters
    nf_blackout = {resident_name: set() for resident_name in residents}
    wr_blackout = {resident_name: set() for resident_name in residents}
    vacation_blackout = {resident_name: set() for resident_name in residents}
    night_counts = {resident_name: 0 for resident_name in residents}
    
    vacation_records = []
    weekend_rounds_records = []
    
    for _, row in residents_df.iterrows():
        resident_name = row['name']
        
        # NF blackout (night shifts)
        if str(row["nf"]).strip().lower() != "no":
            for rng in row['nf'].split("\n"):
                for d in helper.expand_dates(rng, base_year=start_date.year):
                    night_counts[resident_name] += 1
                    helper.update_blackout(resident_name, [d], nf_blackout, buffer_days=5)
        
        # Vacation blackout
        if str(row["leave"]).strip().lower() != "no":
            for rng in row['leave'].split("\n"):
                dates = helper.expand_dates(rng, base_year=start_date.year)
                if not dates:
                    continue
                buffer = 2 if len(dates) == 5 else 0
                helper.update_blackout(resident_name, dates, vacation_blackout, buffer_days=buffer, record_list=vacation_records)
        
        # Weekend rounds blackout
        # Initialize a tracker outside the loop
        range_assignment_tracker = {}

        for _, row in residents_df.iterrows():
            resident_name = row['name']

            if str(row["weekend round"]).strip().lower() != "no":
                for rng in row["weekend round"].split("\n"):
                    dates = helper.expand_dates(rng, base_year=start_date.year)
                    if not dates:
                        continue
                    dates = [pd.Timestamp(d).normalize() for d in sorted(dates)]

                    if resident_year.lower() == "senior":
                        wr_date_ts = dates[0]  # seniors always take first
                    else:
                        # Initialize tracker for this range if needed
                        if rng not in range_assignment_tracker:
                            range_assignment_tracker[rng] = 0

                        # Assign next available date
                        idx = range_assignment_tracker[rng]
                        wr_date_ts = dates[idx % len(dates)]

                        # Update tracker
                        range_assignment_tracker[rng] += 1

                    # Add WR blackout
                    helper.update_blackout(
                        resident_name,
                        [wr_date_ts],
                        wr_blackout,
                        buffer_days=2,
                        exclude_dates={wr_date_ts},
                        record_list=weekend_rounds_records
                    )

                    # Remove assigned WR date from all blackout dicts
                    for b in [nf_blackout, vacation_blackout, wr_blackout]:
                        b[resident_name].discard(wr_date_ts)

    
    vacation_df = pd.DataFrame(vacation_records)
    weekend_rounds_df = pd.DataFrame(weekend_rounds_records)
    wr_residents = set(weekend_rounds_df["resident"]) if not weekend_rounds_df.empty else set()
    
    # Split Residents to NF or Not
    nf_residents = {resident_name for resident_name, c in night_counts.items() if c > 2}
    non_nf_residents = set(residents) - nf_residents
    
    # Build combined blackout dict
    combined_blackout_df = helper.build_combined_blackout_df(nf_blackout, wr_blackout, vacation_blackout)
    resident_level = dict(zip(residents_df["name"], residents_df["level"]))
    
    # Fill NF table with NS
    filled_nf_calendar_df, ns_residents, updated_blackout = helper.fill_ns_cells(
        non_nf_residents, 
        nf_calendar_df, 
        wr_residents=wr_residents, 
        resident_level=resident_level, 
        blackout_df=combined_blackout_df, 
        nf_cols=nf_roles
    )
    
    # NS blackout (2 day before and after)
    ns_blackout = {resident_name: set() for resident_name in residents}
    
    if ns_residents is not None and not ns_residents.empty:
        for _, row in ns_residents.iterrows():
            resident_name = row["resident"]
            ns_date = pd.to_datetime(row["date"])
            helper.update_blackout(resident_name, [ns_date], ns_blackout, buffer_days=2)

        # --- Extended NF blackout logic ---
    for _, row in ns_residents.iterrows():
        resident_name = row["resident"]
        ns_date = pd.to_datetime(row["date"])
        role = str(row.get("role", "")).strip().lower()

        # Only apply for Thursday NF and specific roles
        if ns_date.weekday() == 3 and role in ["ew night", "er-1 night", "er-2 night"]:
            # Block Friday + Saturday of the same week (buffer_days=1 covers both)
            same_week_friday = ns_date + pd.Timedelta(days=1)

            # Block next week's Friday + Saturday
            next_week_friday = ns_date + pd.Timedelta(days=8)

            # Apply blackouts
            helper.update_blackout(resident_name, [same_week_friday], ns_blackout, buffer_days=1)
            helper.update_blackout(resident_name, [next_week_friday], ns_blackout, buffer_days=1)

            # Optional: block all future Thursdays for this resident
            future_thursdays = nf_calendar_df.loc[
                (nf_calendar_df["date"] > ns_date) & (nf_calendar_df["date"].dt.weekday == 3),
                "date"
            ].tolist()

            if future_thursdays:
                helper.update_blackout(resident_name, future_thursdays, ns_blackout)

    
    return (
        residents, 
        resident_levels, 
        day_roles, 
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
        filled_nf_calendar_df
    )
