import pandas as pd
import helper
import general
import nf
import ns
import blackout
# -----------------------------------------------------
def prepare_data(residents_df, start_date, num_weeks, resident_year, nf_max_limit, optional_rules, resident_max_limit):
    """
    Prepare scheduling inputs from the residents DataFrame.
    """

    # Clean column names
    residents_df.columns = residents_df.columns.str.strip().str.lower()

    # Strip trailing/leading spaces from the 'name' values
    residents_df['name'] = residents_df['name'].str.strip()

    # Convert to list
    residents = residents_df['name'].tolist()

    # CALCULATE MAX WEEKENDS
    shifts_limit, points_limit = resident_max_limit[:2]
    resident_max_limit = resident_max_limit + (points_limit - shifts_limit,)

    shifts_limit, points_limit = nf_max_limit[:2]
    nf_max_limit = nf_max_limit + (points_limit - shifts_limit,)


    # Identify duplicates based on resident "name" column (or all columns if you prefer)
    duplicates = residents_df[residents_df.duplicated(subset=["name"], keep="first")]

    # Print who was removed
    if not duplicates.empty:
        print("🚨 Removed duplicates:")
        for name in duplicates["name"].unique():
            print(f"  - {name}")

    if optional_rules is None or not optional_rules:
        optional_rules = {  
                            "NS_next_weekend_blockout": True,
                            "NS_all_future_thursdays_blockout": True,
                            "WR_assigned_to_EW": True,
                            "WR_assigned_to_EW": True
                         }

    # Drop duplicates, keeping the first occurrence
    residents_df = residents_df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)

    if resident_year == "seniors":
        resident_levels = dict(zip(residents_df["name"], residents_df["level"]))
    elif resident_year == "r1":
        resident_levels = {name: "r1" for name in residents_df["name"]}

    start_date = pd.to_datetime(start_date)
    days = pd.date_range(start_date, start_date + pd.Timedelta(weeks=num_weeks) - pd.Timedelta(days=1))
    weekend_days = set(d for d in days if d.strftime('%a') in ['Fri', 'Sat'])

    nf_roles, day_roles = general.extract_shift_columns(resident_year=resident_year)
    nf_roles = [r.strip().lower() for r in nf_roles]
    day_roles = [r.strip().lower() for r in day_roles]

    return (
        residents_df, residents, resident_max_limit, nf_max_limit, optional_rules, 
        resident_levels, start_date, days, weekend_days, nf_roles, day_roles 

    )
