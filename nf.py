import pandas as pd
import helper
from datetime import timedelta

def build_nf_calendar(residents_df, start_date, buffers=None, nf_cols=None):
    """
    Build NF calendar that includes all NF assignments, including cross-year,
    and ensures residents with NF dates in the next year are included.
    """

    if nf_cols is None:
        nf_cols = ["nf1", "nf2", "nf3"]

    start_date = pd.Timestamp(start_date).normalize()

    print(f"display start date: {start_date.year}")

    # --- Step 1: Collect all NF assignments per resident ---
    calendar = {}  # date -> list of resident names
    for _, row in residents_df.iterrows():
        nf_range = row.get("nf", "")
        if not nf_range or str(nf_range).strip().lower() == "no":
            continue
        
        nf_dates = helper.expand_dates(
            str(nf_range),
            base_year=start_date.year,
            anchor_month=start_date.month
        )

        for d in nf_dates:
            d_ts = pd.Timestamp(d).normalize()
            calendar.setdefault(d_ts, []).append(row["name"])


    if not calendar:
        return pd.DataFrame(columns=["day", "date"] + nf_cols)

    # --- Step 2: Determine full range of NF dates ---
    min_date = min(calendar.keys())
    max_date = max(calendar.keys())

    # Build full range from min_date to max_date (cross-year)
    all_dates = [min_date + pd.Timedelta(days=i) for i in range((max_date - min_date).days + 1)]

    # --- Step 3: Build base NF calendar DataFrame ---
    base_calendar = pd.DataFrame({
        "date": all_dates,
        "day": [d.strftime("%a") for d in all_dates],
        "nf": [", ".join(calendar.get(d, [])) for d in all_dates]
    })

    # --- Step 4: Expand into NF1..NF{n_slots} with rotation ---
    calendar_data = []
    for day_idx, row in base_calendar.iterrows():
        names = row["nf"].split(", ") if row["nf"] else []
        rotated_names = helper.rotate_list(names, day_idx)
        row_dict = {
            "day": row["day"],
            "date": row["date"]
        }
        for idx, col_name in enumerate(nf_cols):
            row_dict[col_name] = rotated_names[idx] if idx < len(rotated_names) else ""
        calendar_data.append(row_dict)

    calendar_df = pd.DataFrame(calendar_data)

    # --- Step 5: Slice calendar to display_start_date onwards for display ---
    calendar_df = calendar_df[calendar_df["date"] >= start_date].reset_index(drop=True)

    return calendar_df


def add_nf_day_preferences_seniors(
    model,
    assign,
    roles,
    days,
    nf_residents
):
    """
    HARD CONSTRAINTS:
    - NF residents cannot be assigned to ER-1 role.
    - NF residents can only be assigned to DAY shifts on Tue / Thu.
      (Day shifts on other weekdays are forbidden.)
    """
    preferred_days = {1, 3}  # Tuesday=1, Thursday=3

    for r in nf_residents:
        for d in days:
            weekday = pd.to_datetime(d).weekday()
            for role in roles:
                role_l = role.lower()

                # HARD: no ER-1 shifts for NF residents
                if "er-1" in role_l:
                    model.Add(assign[(d, role, r)] == 0)

                # HARD: no day shifts outside Tue/Thu
                if "day" in role_l and weekday not in preferred_days:
                    model.Add(assign[(d, role, r)] == 0)

def add_nf_day_preferences_juniors(model, assign, roles, days, nf_residents):
    """
    - Hard constraints, NF residents should avoid Tuesday and Thursday
    """
    
    penalties = []
    avoid_days = {1, 3}  # Tuesday=1, Thursday=3 (Python weekday: Monday=0)

    for r in nf_residents:
        for d in days:
            weekday = pd.to_datetime(d).weekday()
            for role in roles:
                # HARD CONSTRAINT: no Tue/Thu for NF
                if weekday in avoid_days and "day" in role.lower():
                    penalty_var = model.NewIntVar(0, 1, f"nf_day_penalty_{d}_{role}_{r}")
                    model.Add(assign[(d, role, r)] == 0)
                    penalties.append(penalty_var)

    return penalties