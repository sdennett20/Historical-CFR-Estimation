from cfr_exact import running_cfr, adapt_drc_total_to_counts
import pandas as pd
df = pd.read_csv("Data/DRC2018_humdata_MOH-Total.csv")
counts = adapt_drc_total_to_counts(df)

results = running_cfr(
    counts,
    dataset_kind="count_table",   # or leave as "auto"
    methods=["naive", "resolved", "delay_adjusted"],
)

print(results.head())

from cfr_exact import estimate_delay_distributions_from_individual_data, standardize_line_list

# ll = standardize_line_list(df)
# delays = estimate_delay_distributions_from_individual_data(
#     ll,
#     onset_col="start_date",
#     outcome_date_col="outcome_date",
#     outcome_col="event",
# )

# death_delay = delays["death"]
# recovery_delay = delays.get("recovery")