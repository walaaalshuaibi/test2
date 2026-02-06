import pandas as pd
import helper
from save_files import save_schedule_as_excel, save_score_as_excel
from scheduler import schedule_with_ortools_full_modular


residents_file = "/Users/walaaalshuaibi/Downloads/Scheduler V12/Testing R1 copy.xlsx"
residents_df = helper.read_excel_as_displayed(residents_file)
#residents_df = pd.read_excel(residents_file)

buffers = (
    6,       # NF_buffer
    2,      # Vacation_buffer
    2,       # WR_buffer
    2        # NS_buffer
)

resident_max_limit = (
    7, # Shifts
    8, # Points
)

nf_max_limit = (
    2,
    2,
)

schedule_df, scores_df = schedule_with_ortools_full_modular(
                            residents_df,
                            "21/12/2025",
                            4,
                            limited_shift_residents=None,
                            off_days=None,
                            on_days=None,
                            resident_max_limit=resident_max_limit,
                            nf_max_limit=nf_max_limit,
                            resident_year="r1",
                            buffers=buffers,
                            r2_cover=None,
                            preassigned_ns_df=None,
                            preassigned_wr_df=None)

print("SCHEDULE DF:")
print(schedule_df)

print("SCORES DF")
print(scores_df)

save_schedule_as_excel(schedule_df)
save_score_as_excel(scores_df)