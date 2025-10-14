# run streamlit code: streamlit run app.py

# app.py
import streamlit as st
import pandas as pd
from datetime import date
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
    st.session_state["constraints"] = pd.DataFrame(columns=["Resident", "Type", "Value"])

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
        r, ctype, val = row["Resident"], row["Type"], row["Value"]

        if ctype == "Max Shifts":
            limited_shift_residents[r] = int(val)

        elif ctype in ["Off Day", "On Day"]:
            # Ensure val is a single Timestamp
            val_date = pd.to_datetime(val)
            val_date = val_date.normalize()  # zero the time component

            if ctype == "Off Day":
                off_rows.append({"Resident": r, "Date": val_date})
            else:
                on_rows.append({"Resident": r, "Date": val_date})

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
        constraint_type = st.selectbox("Constraint Type", ["Off Day", "On Day", "Max Shifts"])

        if constraint_type in ["Off Day", "On Day"]:
            constraint_value = st.date_input("Select Date", value=date.today())
        else:  # Max Shifts
            constraint_value = st.number_input("Enter Max Shifts", min_value=1, step=1)

        if st.button("Add Constraint"):
            new_constraint = pd.DataFrame(
                [[selected_resident, constraint_type, constraint_value]],
                columns=["Resident", "Type", "Value"],
            )
            st.session_state["constraints"] = pd.concat(
                [st.session_state["constraints"], new_constraint], ignore_index=True
            )
            st.success(f"Constraint added for {selected_resident}!")

        # Show constraints table
        st.subheader("Current Constraints")
        if not st.session_state["constraints"].empty:
            st.dataframe(st.session_state["constraints"], use_container_width=True)

            # Delete option
            delete_index = st.number_input(
                "Enter row index to delete", min_value=0, max_value=len(st.session_state["constraints"]) - 1, step=1
            )
            if st.button("Delete Constraint"):
                st.session_state["constraints"].drop(index=delete_index, inplace=True)
                st.session_state["constraints"].reset_index(drop=True, inplace=True)
                st.success("Constraint deleted!")


# =========================================================
# ⚙️ TAB 2: Scheduler
# =========================================================
with tab2:
    st.header("Run Scheduler")

    if st.session_state["residents_df"].empty:
        st.warning("⚠️ Please upload residents file first in the previous tab.")
    else:
        start_date = st.date_input("Select start date", value=date.today())
        num_weeks = st.number_input("Number of weeks to schedule", min_value=1, step=1, value=4)

        st.divider()
        st.subheader("🧠 Night Float (NF) Limits")
        nf_shift_max = st.number_input("Max NF Shifts", min_value=0, step=1, value=1)
        nf_points_max = st.number_input("Max NF Points", min_value=0, step=1, value=1)
        nf_weekend_max = st.number_input("Max NF Weekends", min_value=0, step=1, value=0)

        nf_max_limit = (nf_shift_max, nf_points_max, nf_weekend_max)

        st.divider()
        st.subheader("👤 Regular Resident Limits")
        res_shift_max = st.number_input("Max Shifts", min_value=0, step=1, value=5)
        res_points_max = st.number_input("Max Points", min_value=0, step=1, value=6)
        res_weekend_max = st.number_input("Max Weekend Shifts", min_value=0, step=1, value=2)

        resident_max_limit = (res_shift_max, res_points_max, res_weekend_max)
    

        st.divider()

        if st.button("Run Scheduler 🚀"):
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
                        nf_max_limit=nf_max_limit,
                        resident_max_limit=resident_max_limit
                    )

                    st.session_state["schedule_df"] = schedule_df
                    st.session_state["scores_df"] = scores_df
                    st.success("✅ Scheduling completed successfully!")

                except Exception as e:
                    st.error(f"❌ Scheduling error: {e}")
                    st.text("Traceback (most recent call last):")
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
        save_schedule_as_excel(st.session_state["schedule_df"], output_path=schedule_buffer)
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
