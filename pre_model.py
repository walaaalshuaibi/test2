import pandas as pd
import helper
import general
import nf
import ns
import blackout
# -----------------------------------------------------
def prepare_data(residents_df, start_date, num_weeks, resident_year, buffers, nf_max_limit, optional_rules,
    limited_shift_residents, off_days, on_days, r2_cover=None, preassigned_ns_df=None):
    """
    Prepare scheduling inputs from the residents DataFrame.
    """

    NF_buffer, Vacation_buffer, WR_buffer, NS_buffer = buffers
    residents = residents_df['name'].tolist()
    print(residents_df.head(2))
    if resident_year == "seniors":
        resident_levels = dict(zip(residents_df["name"], residents_df["level"]))
    elif resident_year == "r1":
        resident_levels = {name: "r1" for name in residents_df["name"]}

    start_date = pd.to_datetime(start_date)
    days = pd.date_range(start_date, start_date + pd.Timedelta(weeks=num_weeks) - pd.Timedelta(days=1))
    weekend_days = set(d for d in days if d.strftime('%a') in ['Fri', 'Sat'])

    nf_roles, day_roles = general.extract_shift_columns()
    nf_roles = [r.strip().lower() for r in nf_roles]
    day_roles = [r.strip().lower() for r in day_roles]

    nf_calendar_df = nf.build_nf_calendar(residents_df, start_date, nf_cols=nf_roles)


    print("THIS IS NF CALENDAR")
    print(nf_calendar_df["date"].min(), nf_calendar_df["date"].max())

    print("NF calendar columns:", nf_calendar_df.columns.tolist())



    nf_blackout, night_counts = blackout.nf_blackout_section(residents_df, start_date, NF_buffer) if NF_buffer is not None else ({}, {})
    vacation_blackout, vacation_records = blackout.vacation_blackout_section(residents_df, start_date, Vacation_buffer) if Vacation_buffer is not None else ({}, [])
    wr_blackout, weekend_rounds_records = blackout.wr_blackout_section(residents_df, start_date, WR_buffer, resident_year, r2_cover=r2_cover) if WR_buffer is not None else ({}, [])
    on_blackout, off_blackout = blackout.on_off_days_section(on_days, off_days) if (on_days is not None or off_days is not None) else ({}, {})



    vacation_df = pd.DataFrame(vacation_records)
    weekend_rounds_df = pd.DataFrame(weekend_rounds_records)
    wr_residents = set(weekend_rounds_df["name"]) if not weekend_rounds_df.empty else set()

    nf_residents = {r for r, c in night_counts.items() if c > 2}
    non_nf_residents = set(residents) - nf_residents

    combined_blackout_dict, combined_blackout_df = blackout.build_combined_blackout_df(
    nf_blackout, wr_blackout, vacation_blackout, wr_records=weekend_rounds_records, resident_year=resident_year)

    nf_blackout_lookup = {(r, d): True for r, dates in nf_blackout.items() for d in dates}

    filled_nf_calendar_df, ns_residents, updated_blackout = ns.fill_ns_cells(
        resident_year,
        non_nf_residents, 
        nf_residents,
        nf_max_limit,
        nf_calendar_df, 
        wr_residents=wr_residents, 
        resident_level=resident_levels, 
        blackout_df=combined_blackout_df, 
        nf_cols=nf_roles,
        nf_blackout_lookup=nf_blackout_lookup,
        preassigned_ns_df=preassigned_ns_df
    )

    ns_blackout = {r: set() for r in residents}
    if ns_residents is not None and not ns_residents.empty:
        for _, row in ns_residents.iterrows():
            resident_name = row["name"]
            ns_date = pd.to_datetime(row["date"])
            blackout.update_blackout(resident_name, [ns_date], ns_blackout, buffer_days=NS_buffer)
            if optional_rules.get("NS_next_weekend_blockout", False):
                next_week_friday = ns_date + pd.Timedelta(days=8)
                blackout.update_blackout(resident_name, [next_week_friday], ns_blackout, buffer_days=1)
            if optional_rules.get("NS_all_future_thursdays_blockout", False):
                future_thursdays = nf_calendar_df.loc[
                    (nf_calendar_df["date"] > ns_date) & (nf_calendar_df["date"].dt.weekday == 3), "date"
                ].tolist()
                if future_thursdays:
                    blackout.update_blackout(resident_name, future_thursdays, ns_blackout)

    # ✅ Merge NS blackout into the combined blackout dictionary
    for resident, dates in ns_blackout.items():
        combined_blackout_dict.setdefault(resident, set()).update(dates)

    # ✅ Rebuild the DataFrame after merging
    combined_blackout_df = pd.DataFrame(
        [{"name": r, "date": d} for r, dates in combined_blackout_dict.items() for d in dates]
    )

    return (
        residents, resident_levels, day_roles, days, weekend_days,
        nf_residents, non_nf_residents, wr_residents,
        combined_blackout_df, combined_blackout_dict,
        night_counts, vacation_df, weekend_rounds_df,
        ns_residents, filled_nf_calendar_df
    )
