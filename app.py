# run streamlit code: streamlit run app.py

# app.py
import streamlit as st
import pandas as pd
from datetime import date
import general
import helper
from scheduler import schedule_with_ortools_full_modular  # your main scheduler function
from save_files import save_schedule_as_excel, save_score_as_excel
import traceback
import io
# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Resident Day Scheduler", layout="wide")
st.title("📅 Resident Day Scheduler (WIP)")

# -------------------------
# Initialize session state
# -------------------------
if "residents_df" not in st.session_state:
    st.session_state["residents_df"] = pd.DataFrame()

if "constraints" not in st.session_state:
    st.session_state["constraints"] = pd.DataFrame(columns=["resident", "type", "value"])

if "schedule_df" not in st.session_state:
    st.session_state["schedule_df"] = None

if "scores_df" not in st.session_state:
    st.session_state["scores_df"] = None

# =========================================================
# Helpers
# =========================================================
def build_constraints_from_session():
    df = st.session_state["constraints"]
    limited_shift_residents = {}
    off_rows, on_rows = [], []

    for _, row in df.iterrows():
        r, ctype, val = row["resident"], row["type"], row["value"]

        if ctype == "max shifts":
            limited_shift_residents[r] = int(val)

        elif ctype in ["off day", "on day"]:
            # Ensure val is a single Timestamp
            val_date = pd.to_datetime(val)
            val_date = val_date.normalize()  # zero the time component

            if ctype == "off day":
                off_rows.append({"resident": r, "date": val_date})
            else:
                on_rows.append({"resident": r, "date": val_date})

    off_days = pd.DataFrame(off_rows) if off_rows else None
    on_days = pd.DataFrame(on_rows) if on_rows else None

    return limited_shift_residents or None, off_days, on_days

# =========================================================
# 🏠 TAB 1: Residents & Constraints
# =========================================================
tab1, tab2, tab3 = st.tabs(["🏠 Residents & Constraints", "⚙️ Scheduler", "📊 Results"])

with tab1:
    st.header("Resident List")

    residents_file = st.file_uploader("Upload Residents Excel", type=["xlsx"])
    if residents_file:
        st.session_state["residents_df"] = pd.read_excel(residents_file)
        st.success("✅ Residents file uploaded successfully!")

    if not st.session_state["residents_df"].empty:
        st.dataframe(st.session_state["residents_df"], use_container_width=True)

        st.divider()
        st.subheader("➕ Add Constraints")

        resident_list = st.session_state["residents_df"]["Name"].dropna().tolist()
        selected_resident = st.selectbox("Select Resident", resident_list)
        constraint_type = st.selectbox("Constraint Type", ["off day", "on day", "max shifts"])

        if constraint_type in ["off day", "on day"]:
            constraint_value = st.date_input("Select Date", value=date.today())
        else:  # Max Shifts
            constraint_value = st.number_input("Enter Max Shifts", min_value=1, step=1)

        if st.button("Add Constraint"):
            new_constraint = pd.DataFrame(
                [[selected_resident, constraint_type, constraint_value]],
                columns=["resident", "type", "value"],
            )
            st.session_state["constraints"] = pd.concat(
                [st.session_state["constraints"], new_constraint], ignore_index=True
            )
            st.success(f"Constraint added for {selected_resident}!")

        # Show constraints table
        st.subheader("Current Constraints")

        if not st.session_state["constraints"].empty:
            for i, row in st.session_state["constraints"].iterrows():
                cols = st.columns([3, 2, 3, 1])  # layout for name | type | value | delete button
                with cols[0]:
                    st.write(f"👤 **{row['resident']}**")
                with cols[1]:
                    st.write(row["type"])
                with cols[2]:
                    st.write(str(row["value"]))
                with cols[3]:
                    if st.button("🗑️ Delete", key=f"delete_{i}"):
                        st.session_state["constraints"].drop(index=i, inplace=True)
                        st.session_state["constraints"].reset_index(drop=True, inplace=True)
                        st.rerun()  # instantly refresh UI
        else:
            st.info("No constraints added yet.")


# =========================================================
# ⚙️ TAB 2: Scheduler
# =========================================================
with tab2:
    st.header("Run Scheduler 🚀")

    if st.session_state["residents_df"].empty:
        st.warning("⚠️ Please upload residents file first in the previous tab.")
    else:
        # ---------------------------
        # Basic schedule settings
        # ---------------------------
        start_date = st.date_input("Select start date",value=None,key="start_date_input")
        num_weeks = st.number_input("Number of weeks to schedule", min_value=1, step=1, value=4)

        # Check if start_date is selected
        if start_date is None:
            st.warning("⚠️ Please select a start date before running the scheduler.")

        st.subheader("👤 Resident Year")
        resident_year = st.selectbox("Select Resident Year for this schedule", ["seniors", "r1"])

        # ---------------------------
        # Defaults
        # ---------------------------
        DEFAULTS = {
            "res_shift_max": 5,
            "res_points_max": 6,
            "nf_shift_max": 2,
            "nf_points_max": 2,
            "nf_buffer": 5,
            "ns_buffer": 2,
            "wr_buffer": 2,
            "vacation_buffer": 2
        }

        if ("resident_year_last" not in st.session_state 
            or st.session_state["resident_year_last"] != resident_year):

            if resident_year == "seniors":
                DEFAULTS = {
                    "res_shift_max": 5,
                    "res_points_max": 6,
                    "nf_shift_max": 2,
                    "nf_points_max": 2,
                    "nf_buffer": 5,
                    "ns_buffer": 2,
                    "wr_buffer": 2,
                    "vacation_buffer": 2
                }
            elif resident_year == "r1":
                DEFAULTS = {
                    "res_shift_max": 10,
                    "res_points_max": 12,
                    "nf_shift_max": 2,
                    "nf_points_max": 2,
                    "nf_buffer": 5,
                    "ns_buffer": 2,
                    "wr_buffer": 2,
                    "vacation_buffer": 2
                }

            # Reset session_state inputs
            st.session_state["res_shift_max_input"] = DEFAULTS["res_shift_max"]
            st.session_state["res_points_max_input"] = DEFAULTS["res_points_max"]
            st.session_state["nf_shift_max_input"] = DEFAULTS["nf_shift_max"]
            st.session_state["nf_points_max_input"] = DEFAULTS["nf_points_max"]
            st.session_state["nf_buffer_input"] = DEFAULTS["nf_buffer"]
            st.session_state["ns_buffer_input"] = DEFAULTS["ns_buffer"]
            st.session_state["wr_buffer_input"] = DEFAULTS["wr_buffer"]
            st.session_state["vac_buffer_input"] = DEFAULTS["vacation_buffer"]

            # Remember current resident year
            st.session_state["resident_year_last"] = resident_year

        # ---------------------------
        # Restore callbacks
        # ---------------------------
        def restore_regular_defaults():
            st.session_state["res_shift_max_input"] = int(DEFAULTS["res_shift_max"])
            st.session_state["res_points_max_input"] = int(DEFAULTS["res_points_max"])

        def restore_nf_defaults():
            st.session_state["nf_shift_max_input"] = int(DEFAULTS["nf_shift_max"])
            st.session_state["nf_points_max_input"] = int(DEFAULTS["nf_points_max"])

        def restore_buffers_defaults():
            st.session_state["nf_buffer_input"] = int(DEFAULTS["nf_buffer"])
            st.session_state["ns_buffer_input"] = int(DEFAULTS["ns_buffer"])
            st.session_state["wr_buffer_input"] = int(DEFAULTS["wr_buffer"])
            st.session_state["vac_buffer_input"] = int(DEFAULTS["vacation_buffer"])

        # ---------------------------
        # Regular Resident Limits
        # ---------------------------
        st.subheader("👤 Regular Resident Limits")
        col1, col2, col3 = st.columns([3,3,1])
        with col1:
            res_shift_max = st.number_input(
                "Max Shifts",
                min_value=0,
                step=1,
                value=st.session_state.get("res_shift_max_input", DEFAULTS["res_shift_max"]),
                key="res_shift_max_input"
            )
        with col2:
            res_points_max = st.number_input(
                "Max Points",
                min_value=0,
                step=1,
                value=st.session_state.get("res_points_max_input", DEFAULTS["res_points_max"]),
                key="res_points_max_input"
            )
        with col3:
            st.button("🔄 Restore Defaults", on_click=restore_regular_defaults, key="restore_regular_limits")
        resident_max_limit = (int(res_shift_max), int(res_points_max))

        # ---------------------------
        # Night Float (NF) Limits
        # ---------------------------
        st.subheader("🧠 Night Float (NF) Limits")
        col1, col2, col3 = st.columns([3,3,1])
        with col1:
            nf_shift_max = st.number_input(
                "Max NF Shifts",
                min_value=0,
                step=1,
                value=st.session_state.get("nf_shift_max_input", DEFAULTS["nf_shift_max"]),
                key="nf_shift_max_input"
            )
        with col2:
            nf_points_max = st.number_input(
                "Max NF Points",
                min_value=0,
                step=1,
                value=st.session_state.get("nf_points_max_input", DEFAULTS["nf_points_max"]),
                key="nf_points_max_input"
            )
        with col3:
            st.button("🔄 Restore Defaults", on_click=restore_nf_defaults, key="restore_nf_limits")

        # ---------------------------
        # Buffers (2x2 layout)
        # ---------------------------
        st.subheader("🛡️ Buffers (in days)")
        col0, col5 = st.columns([8,1])
        with col5:
            st.button("🔄 Restore Defaults", on_click=restore_buffers_defaults, key="restore_buffers")

        col1, col2 = st.columns(2)
        with col1:
            nf_buffer = st.number_input(
                "Blocked out days: before/after Night Float (NF) week",
                min_value=0,
                step=1,
                value=st.session_state.get("nf_buffer_input", DEFAULTS["nf_buffer"]),
                key="nf_buffer_input"
            )
        with col2:
            ns_buffer = st.number_input(
                "Blocked out days: before/after NS shifts",
                min_value=0,
                step=1,
                value=st.session_state.get("ns_buffer_input", DEFAULTS["ns_buffer"]),
                key="ns_buffer_input"
            )

        col3, col4 = st.columns(2)
        with col3:
            wr_buffer = st.number_input(
                "Blocked out days: before/after Weekend Rounds (WR)",
                min_value=0,
                step=1,
                value=st.session_state.get("wr_buffer_input", DEFAULTS["wr_buffer"]),
                key="wr_buffer_input"
            )
        with col4:
            vac_buffer = st.number_input(
                "Blocked out days: before/after Vacation",
                min_value=0,
                step=1,
                value=st.session_state.get("vac_buffer_input", DEFAULTS["vacation_buffer"]),
                key="vac_buffer_input"
            )

        buffers = (
                int(nf_buffer),       # NF_buffer
                int(vac_buffer),      # Vacation_buffer
                int(wr_buffer),       # WR_buffer
                int(ns_buffer)        # NS_buffer
            )

        # ---------------------------
        # R2 Coverage
        # ---------------------------
        r2_cover=None
        if resident_year == "r1":
            st.subheader("👤 R2 Coverage")
            r2_name = st.text_input(
                "Enter resident name for R2 coverage (any name, not in resident list)",
                value=""
            )
            r2_date = st.date_input("Select date for R2 coverage")
            r2_cover = (r2_name.strip(), r2_date) if r2_name.strip() else None

        # ---------------------------
        # Preassigned NS Shifts
        # ---------------------------
        st.subheader("📥 Preassigned NS Shifts (Optional)")
        if "preassigned_ns" not in st.session_state:
            st.session_state["preassigned_ns"] = pd.DataFrame(columns=["name", "date", "role"])

        with st.expander("Add Preassigned NS Shift"):
            resident_list = st.session_state["residents_df"]["Name"].dropna().tolist()
            ns_name = st.selectbox("Select Resident", resident_list, key="ns_name_input")
            ns_date = st.date_input("Select Date", value=date.today(), key="ns_date_input")
            ns_role = st.selectbox("Select Role", general.extract_shift_columns()[0], key="ns_role_input")
            
            if st.button("Add NS Preassignment", key="add_ns_preassign") and ns_name:
                new_row = pd.DataFrame([[ns_name, ns_date, ns_role]], columns=["name", "date", "role"])
                st.session_state["preassigned_ns"] = pd.concat(
                    [st.session_state["preassigned_ns"], new_row], ignore_index=True
                )

        # Display NS table with delete buttons
        if not st.session_state["preassigned_ns"].empty:
            st.subheader("Current NS Preassignments")
            for i, row in st.session_state["preassigned_ns"].iterrows():
                cols = st.columns([3, 2, 2, 1])
                with cols[0]:
                    st.write(row["name"])
                with cols[1]:
                    st.write(str(row["date"]))
                with cols[2]:
                    st.write(row["role"])
                with cols[3]:
                    if st.button("🗑️ Delete", key=f"delete_ns_{i}"):
                        st.session_state["preassigned_ns"].drop(index=i, inplace=True)
                        st.session_state["preassigned_ns"].reset_index(drop=True, inplace=True)
                        st.rerun()

        preassigned_ns_df = st.session_state["preassigned_ns"] if not st.session_state["preassigned_ns"].empty else None


        # ---------------------------
        # Preassigned WR Shifts (Optional)
        # ---------------------------
        if resident_year == "r1":
            st.subheader("📥 Preassigned WR Shifts (Optional)")
            if "preassigned_wr" not in st.session_state:
                st.session_state["preassigned_wr"] = pd.DataFrame(columns=["name", "date", "role"])

            from datetime import timedelta
            end_date = start_date + timedelta(weeks=num_weeks)
            all_dates = pd.date_range(start=start_date, end=end_date)
            wr_dates = [d.date() for d in all_dates if d.weekday() in [4, 5]]  # Friday=4, Saturday=5

            with st.expander("Add Preassigned WR Shift"):
                selected_date = st.selectbox("Select WR Date", wr_dates, key="wr_date_input")

                available_residents = []
                for _, row in st.session_state["residents_df"].iterrows():
                    wr_col_value = row.get("Weekend round")
                    if wr_col_value:
                        expanded_dates = [d.date() if isinstance(d, pd.Timestamp) else d
                                        for d in helper.expand_dates(wr_col_value, base_year=2025)]
                        if selected_date in expanded_dates:
                            available_residents.append(row["Name"])

                if available_residents:
                    wr_name = st.selectbox("Select Resident", available_residents, key="wr_name_input")
                else:
                    st.warning("⚠️ No residents have this date in their Weekend Round.")
                    wr_name = None

                if wr_name:
                    wr_role = st.selectbox("Select Role", general.extract_shift_columns()[1], key="wr_role_input")
                    if st.button("Add WR Preassignment", key="add_wr_preassign"):
                        new_row = pd.DataFrame([[wr_name, selected_date, wr_role]], columns=["name", "date", "role"])
                        st.session_state["preassigned_wr"] = pd.concat(
                            [st.session_state["preassigned_wr"], new_row], ignore_index=True
                        )

        # Display WR table with delete buttons
        if resident_year == "r1" and not st.session_state["preassigned_wr"].empty:
            st.subheader("Current WR Preassignments")
            for i, row in st.session_state["preassigned_wr"].iterrows():
                cols = st.columns([3, 2, 2, 1])
                with cols[0]:
                    st.write(row["name"])
                with cols[1]:
                    st.write(str(row["date"]))
                with cols[2]:
                    st.write(row["role"])
                with cols[3]:
                    if st.button("🗑️ Delete", key=f"delete_wr_{i}"):
                        st.session_state["preassigned_wr"].drop(index=i, inplace=True)
                        st.session_state["preassigned_wr"].reset_index(drop=True, inplace=True)
                        st.rerun()

        preassigned_wr_df = st.session_state["preassigned_wr"] if resident_year == "r1" and not st.session_state["preassigned_wr"].empty else None

        # ---------------------------
        # Run Scheduler
        # ---------------------------
        if st.button("Run Scheduler 🚀", key="run_scheduler"):
            if start_date is None:
                st.warning("⚠️ You must select a start date before running the scheduler.")
            else:
                with st.spinner("Scheduling in progress..."):
                    try:
                        limited_shift_residents, off_days, on_days = build_constraints_from_session()

                        schedule_df, scores_df = schedule_with_ortools_full_modular(
                            st.session_state["residents_df"],
                            start_date,
                            num_weeks,
                            limited_shift_residents=limited_shift_residents,
                            off_days=off_days,
                            on_days=on_days,
                            resident_max_limit=resident_max_limit,
                            nf_max_limit=nf_max_limit,
                            resident_year=resident_year,
                            buffers=buffers,
                            r2_cover=r2_cover,
                            preassigned_ns_df=preassigned_ns_df,
                            preassigned_wr_df=preassigned_wr_df
                        )

                        st.session_state["schedule_df"] = schedule_df
                        st.session_state["scores_df"] = scores_df
                        st.success("✅ Scheduling completed successfully!")

                    except Exception as e:
                        if "'level'" in str(e):
                            st.error("❌ Scheduling error: 'level' (make sure to change from senior to r1)")
                        else:
                            st.error(f"❌ Scheduling error: {e}")
                        st.text(traceback.format_exc())

# =========================================================
# 📊 TAB 3: Results
# =========================================================
with tab3:
    st.header("Results")

    if st.session_state["schedule_df"] is not None:
        st.subheader("📅 Schedule Preview")
        st.dataframe(st.session_state["schedule_df"], use_container_width=True)

        # Save to in‑memory Excel
        schedule_buffer = io.BytesIO()
        save_schedule_as_excel(st.session_state["schedule_df"],resident_year=resident_year ,output_path=schedule_buffer)
        st.download_button(
            label="⬇️ Download Schedule (Excel)",
            data=schedule_buffer.getvalue(),
            file_name="Schedule.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if st.session_state["scores_df"] is not None:
        st.subheader("📊 Scores Preview")
        st.dataframe(st.session_state["scores_df"], use_container_width=True)

        # Save to in‑memory Excel
        scores_buffer = io.BytesIO()
        save_score_as_excel(st.session_state["scores_df"], output_path=scores_buffer)
        st.download_button(
            label="⬇️ Download Scores (Excel)",
            data=scores_buffer.getvalue(),
            file_name="Score.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
