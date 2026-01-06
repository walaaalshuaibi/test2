import pandas as pd
import wr
import helper
from datetime import timedelta

# ======================================================
# MAIN ENTRY
# ======================================================

def prepare_blackouts_and_grayouts(
    buffers,
    residents_df,
    start_date,
    num_weeks,
    resident_year,
    r2_cover,
    on_days,
    off_days,
    residents
):

    NF_buffer, Vacation_buffer, WR_buffer, _ = buffers

    # ---------------- NF ----------------
    nf_blackout, nf_grayout, night_counts = (
        nf_blackout_section(residents_df, start_date, num_weeks, NF_buffer)
        if NF_buffer is not None else ({}, {}, {})
    )

    # ---------------- Vacation ----------------
    vacation_blackout, vacation_grayout = (
        vacation_blackout_section(residents_df, start_date, Vacation_buffer)
        if Vacation_buffer is not None else ({}, {})
    )

    # ---------------- WR ----------------
    wr_blackout, wr_grayout, wr_records = (
        wr_blackout_section(
            residents_df, start_date, WR_buffer, resident_year, r2_cover
        ) if WR_buffer is not None else ({}, {}, [])
    )

    # ---------------- ON / OFF ----------------
    # FIX: ADD GRAYOUT FOR OFF
    on_blackout, on_grayout, off_blackout = on_off_blackout_section(on_days, off_days)

    # ================= COMBINE =================

    hard_blackouts = combine_dicts(
        nf_blackout,
        vacation_blackout,
        wr_blackout,
        off_blackout
    )

    soft_grayouts = combine_dicts(
        nf_grayout,
        vacation_grayout,
        wr_grayout,
        on_grayout
    )

    # 🔒 HARD ALWAYS WINS
    soft_grayouts = remove_soft_conflicts(soft_grayouts, hard_blackouts)

    # ================= OUTPUT =================

    wr_residents = pd.DataFrame(wr_records) # df
    nf_residents = {r for r, c in night_counts.items() if c > 2} # set 
    non_nf_residents = set(residents) - nf_residents # set

    return {
        "blackouts": hard_blackouts,
        "grayouts": soft_grayouts,
        "blackout_df": dict_to_df(hard_blackouts),
        "grayout_df": dict_to_df(soft_grayouts),
        "night_counts": night_counts,
        "wr_residents": wr_residents,
        "nf_residents": nf_residents,
        "non_nf_residents": non_nf_residents,
    }

# ======================================================
# NF
# ======================================================

def nf_blackout_section(residents_df, start_date, num_weeks, NF_buffer):

    end_date = start_date + timedelta(weeks=num_weeks) - timedelta(days=1)

    nf_blackout = {}
    nf_grayout = {}
    night_counts = {}

    for _, row in residents_df.iterrows():
        name = row["name"]
        nf_blackout[name] = set()
        nf_grayout[name] = set()
        night_counts[name] = 0

        nf_value = str(row.get("nf", "")).strip()
        if not nf_value or nf_value.lower() == "no":
            continue

        for rng in nf_value.split("\n"):
            dates = helper.expand_dates(
                rng,
                base_year=start_date.year,
                anchor_month=start_date.month,
            )

            valid = [d for d in dates if start_date <= d <= end_date]
            night_counts[name] += len(valid)

            # 🔴 HARD NF
            nf_blackout[name].update(valid)

            # ⚪ SOFT buffer
            for d in valid:
                for i in range(1, NF_buffer + 1):
                    for adj in (d - timedelta(days=i), d + timedelta(days=i)):
                        if start_date <= adj <= end_date:
                            nf_grayout[name].add(adj)

    return nf_blackout, nf_grayout, night_counts

# ======================================================
# VACATION
# ======================================================

def vacation_blackout_section(residents_df, start_date, Vacation_buffer):

    hard = {}
    soft = {}

    for _, row in residents_df.iterrows():
        name = row["name"]
        hard[name] = set()
        soft[name] = set()

        leave = str(row.get("leave", "")).strip()
        if not leave or leave.lower() == "no":
            continue

        for rng in leave.split("\n"):
            dates = helper.expand_dates(
                rng,
                base_year=start_date.year,
                anchor_month=start_date.month,
            )

            hard[name].update(dates)

            if len(dates) == 5 and Vacation_buffer:
                for d in dates:
                    for i in range(1, Vacation_buffer + 1):
                        soft[name].add(d - pd.Timedelta(days=i))
                        soft[name].add(d + pd.Timedelta(days=i))

    return hard, soft

# ======================================================
# WR
# ======================================================

def wr_blackout_section(residents_df, start_date, WR_buffer, resident_year, r2_cover):

    hard = {}
    soft = {}
    records = []

    for _, r in residents_df.iterrows():
        hard[r["name"]] = set()
        soft[r["name"]] = set()

    assignments = wr.build_weekend_round_assignments(
        residents_df, start_date, resident_year, r2_cover
    )

    for name, wr_date in assignments:
        hard[name].add(wr_date)
        records.append({"name": name, "date": wr_date})

        for i in range(1, WR_buffer + 1):
            soft[name].add(wr_date - pd.Timedelta(days=i))
            soft[name].add(wr_date + pd.Timedelta(days=i))

    return hard, soft, records

# ======================================================
# ON / OFF
# ======================================================

def on_off_blackout_section(on_days=None, off_days=None):

    on_grayout = {}
    on_blackout = {}
    off_blackout = {}

    if on_days is not None and not on_days.empty:
        on_days["date"] = pd.to_datetime(on_days["date"])
        for _, row in on_days.iterrows():
            name, d = row["name"], row["date"]
            on_grayout.setdefault(name, set()).update(
                pd.date_range(d - pd.Timedelta(days=1), d + pd.Timedelta(days=1)).difference([d])
            )

    if off_days is not None and not off_days.empty:
        off_days["date"] = pd.to_datetime(off_days["date"])
        for _, row in off_days.iterrows():
            off_blackout.setdefault(row["name"], set()).add(row["date"])

    return on_blackout, on_grayout, off_blackout

# ======================================================
# NS
# ======================================================

def ns_grayout_section(ns_df, rules, nf_calendar_df, buffers, combined_blackout_df, soft_grayouts):
    """
    Build NS grayouts (buffered days) for residents, 
    skipping dates already in the combined hard blackout.
    """
    _, _, _, NS_buffer = buffers

    for _, row in ns_df.iterrows():
        name = row["name"]
        d = pd.to_datetime(row["date"])

        # Initialize soft_grayouts for resident if missing
        if name not in soft_grayouts:
            soft_grayouts[name] = set()

        # Add buffered days around the NS date
        for i in range(1, NS_buffer + 1):
            for adj in (d - pd.Timedelta(days=i), d + pd.Timedelta(days=i)):
                if adj not in combined_blackout_df.get(name, set()):
                    soft_grayouts[name].add(adj)

        # Optional rules
        if rules.get("NS_next_weekend_blockout"):
            future_date = d + pd.Timedelta(days=8)
            if future_date not in combined_blackout_df.get(name, set()):
                soft_grayouts[name].add(future_date)

        if rules.get("NS_all_future_thursdays_blockout"):
            thursdays = nf_calendar_df.loc[
                (nf_calendar_df["date"] > d)
                & (nf_calendar_df["date"].dt.weekday == 3),
                "date",
            ]
            for t in thursdays:
                if t not in combined_blackout_df.get(name, set()):
                    soft_grayouts[name].add(t)

    return soft_grayouts

# ======================================================
# HELPERS
# ======================================================

def combine_dicts(*dicts):
    combined = {}
    for d in dicts:
        for r, dates in d.items():
            combined.setdefault(r, set()).update(dates)
    return combined

def remove_soft_conflicts(soft, hard):
    """Ensure grayouts never overlap blackouts."""
    for r in soft:
        if r in hard:
            soft[r] -= hard[r]
    return soft

def dict_to_df(d):
    return pd.DataFrame(
        [{"name": r, "date": dt} for r, dates in d.items() for dt in dates]
    )

def update_blackout(resident, dates, blackout_dict): 
    for d in dates: 
        blackout_range = pd.DatetimeIndex([d]) 
        blackout_dict.setdefault(resident, set()).update(blackout_range) 