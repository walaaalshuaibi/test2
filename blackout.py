import pandas as pd
import wr
import helper
import blackout

def update_blackout(resident, dates, blackout_dict, buffer_days=0, record_list=None):
    """ 
    Adds blackout dates for a resident with optional buffer and exclusions. 
    Args:
        resident (str): Resident name.
        dates (list): List of central dates.
        blackout_dict (dict): Dict to update.
        buffer_days (int): Days before/after each date to include.
        record_list (list): Optional list to append blackout records.
    """
    
    for d in dates:
        if buffer_days > 0:
            blackout_range = pd.date_range(d - pd.Timedelta(days=buffer_days), d + pd.Timedelta(days=buffer_days))
        else:
            blackout_range = pd.DatetimeIndex([d])
        
        blackout_dict.setdefault(resident, set()).update(blackout_range)
        
        if record_list is not None:
            record_list.append({"name": resident, "date": d})

def build_combined_blackout_df(*blackout_dicts, wr_records=None, resident_year):
    combined = {}
    for bd in blackout_dicts:
        for r, dates in bd.items():
            combined.setdefault(r, set()).update(dates)

    if resident_year == "seniors":
        # 🔑 Remove WR dates from all blackout sets if wr_records provided
        if wr_records:
            for rec in wr_records:
                resident = rec["name"]
                wr_date = pd.to_datetime(rec["date"])
                if resident in combined:
                    combined[resident].discard(wr_date)

    # Also build a DataFrame for optional analysis/export
    combined_df = pd.DataFrame(
        [{"name": r, "date": d} for r, dates in combined.items() for d in dates]
    )

    return combined, combined_df

def build_blackout_lookup(blackout_df):
    blackout_dict = {}
    for _, row in blackout_df.iterrows():
        resident = row["name"]
        date = pd.to_datetime(row["date"])
        blackout_dict.setdefault(resident, set()).add(date)
    return blackout_dict

def nf_blackout_section(residents_df, start_date, NF_buffer):
    nf_blackout = {r["name"]: set() for _, r in residents_df.iterrows()}
    night_counts = {r["name"]: 0 for _, r in residents_df.iterrows()}

    for _, row in residents_df.iterrows():
        resident_name = row["name"]

        if str(row.get("nf", "no")).strip().lower() != "no":
            for rng in row["nf"].split("\n"):
                for d in helper.expand_dates(rng, base_year=start_date.year):
                    night_counts[resident_name] += 1
                    update_blackout(resident_name, [d], nf_blackout, buffer_days=NF_buffer)

    return nf_blackout, night_counts


def vacation_blackout_section(residents_df, start_date, Vacation_buffer):
    vacation_blackout = {r["name"]: set() for _, r in residents_df.iterrows()}
    vacation_records = []

    for _, row in residents_df.iterrows():
        resident_name = row["name"]

        if str(row.get("leave", "no")).strip().lower() != "no":
            for rng in row["leave"].split("\n"):
                dates = helper.expand_dates(rng, base_year=start_date.year)
                if not dates:
                    continue

                buffer = Vacation_buffer if len(dates) == 5 else 0
                update_blackout(
                    resident_name,
                    dates,
                    vacation_blackout,
                    buffer_days=buffer,
                    record_list=vacation_records,
                )

    return vacation_blackout, vacation_records


def wr_blackout_section(residents_df, start_date, WR_buffer, resident_year,
                        nf_blackout=None, vacation_blackout=None, r2_cover=None):
    """
    Wrapper that builds WR assignments and applies blackout rules.
    Returns:
      wr_blackout, weekend_rounds_records
    """
    wr_blackout = {r["name"]: set() for _, r in residents_df.iterrows()}
    weekend_rounds_records = []

    # --- get assignments using your logic ---
    assignments = wr.build_weekend_round_assignments(residents_df, start_date, resident_year, r2_cover)

    for resident_name, wr_date_ts in assignments:
        blackout.update_blackout(
            resident_name,
            [wr_date_ts],
            wr_blackout,
            buffer_days=WR_buffer,
            record_list=weekend_rounds_records,
        )

    return wr_blackout, weekend_rounds_records

def on_off_days_section(on_days=None, off_days=None):
    """
    Builds blackout dicts for on_days and off_days.
    """
    on_blackout = {}
    off_blackout = {}

    if on_days is not None and not on_days.empty:
        on_days["date"] = pd.to_datetime(on_days["date"])
        for _, row in on_days.iterrows():
            resident, d = row["name"], row["date"]
            blackout_dates = pd.date_range(d - pd.Timedelta(days=1), d + pd.Timedelta(days=1)).difference([d])
            on_blackout.setdefault(resident, set()).update(blackout_dates)

    if off_days is not None and not off_days.empty:
        off_days["date"] = pd.to_datetime(off_days["date"])
        for _, row in off_days.iterrows():
            resident, d = row["name"], row["date"]
            off_blackout.setdefault(resident, set()).add(d)

    return on_blackout, off_blackout