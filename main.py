import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


df = pd.read_csv("Bass_Fishing_Data_Cleaned.csv")
water_temp_df = df.groupby("Water Temp")
fish_caught_df = df.groupby("Fish Caught")
lake_name_df = df.groupby("Lake")

Newman_Lake_df = lake_name_df.get_group("Newman Lake")
newman_wt_ser = Newman_Lake_df["Water Temp"]
newman_fc_ser = Newman_Lake_df["Fish Caught"]
# plt.plot(newman_wt_ser, newman_fc_ser)
# plt.savefig("newman_lake_fish_caught_vs_water_temp.pdf")

Eloika_Lake_df = lake_name_df.get_group("Eloika Lake")
eloika_wt_ser = Eloika_Lake_df["Water Temp"]
eloika_fc_ser = Eloika_Lake_df["Fish Caught"]
# plt.plot(eloika_wt_ser, eloika_fc_ser)
# plt.savefig("eloika_lake_fish_caught_vs_water_temp.pdf")

Long_Lake_df = lake_name_df.get_group("Long Lake")
long_wt_ser = Long_Lake_df["Water Temp"]
long_fc_ser = Long_Lake_df["Fish Caught"]
# plt.plot(long_wt_ser, long_fc_ser)
# plt.savefig("long_lake_fish_caught_vs_water_temp.pdf")

Liberty_Lake_df = lake_name_df.get_group("Liberty Lake")
liberty_wt_ser = Liberty_Lake_df["Water Temp"]
liberty_fc_ser = Liberty_Lake_df["Fish Caught"]
# plt.plot(liberty_wt_ser, liberty_fc_ser)
# plt.savefig("liberty_lake_fish_caught_vs_water_temp.pdf")

Silver_Lake_df = lake_name_df.get_group("Silver Lake")
silver_wt_ser = Silver_Lake_df["Water Temp"]
silver_fc_ser = Silver_Lake_df["Fish Caught"]
# plt.plot(silver_wt_ser, silver_fc_ser)
# plt.savefig("silver_lake_fish_caught_vs_water_temp.pdf")

Bonnie_Lake_df = lake_name_df.get_group("Bonnie Lake")
bonnie_wt_ser = Bonnie_Lake_df["Water Temp"]
bonnie_fc_ser = Bonnie_Lake_df["Fish Caught"]
# plt.plot(bonnie_wt_ser, bonnie_fc_ser)
# plt.savefig("bonnie_lake_fish_caught_vs_water_temp.pdf")

Loon_Lake_df = lake_name_df.get_group("Loon Lake")
loon_wt_ser = Loon_Lake_df["Water Temp"]
loon_fc_ser = Loon_Lake_df["Fish Caught"]
# plt.plot(loon_wt_ser, loon_fc_ser)
# plt.savefig("loon_lake_fish_caught_vs_water_temp.pdf")

Sacheen_Lake_df = lake_name_df.get_group("Sacheen Lake")
sacheen_wt_ser = Sacheen_Lake_df["Water Temp"]
sacheen_fc_ser = Sacheen_Lake_df["Fish Caught"]
# plt.plot(sacheen_wt_ser, sacheen_fc_ser)
# plt.savefig("sacheen_lake_fish_caught_vs_water_temp.pdf")

# Pend_Orielle_River_df = lake_name_df.get_group("Pend Orielle River")
# Snake_River_df = lake_name_df.get_group("Snake River")


# print(water_temp.mean())
# print(fish_caught.mean())
# plt.plot(water_temp, fish_caught)
# plt.savefig("fish_caught_vs_water_temp.pdf")

#DID I CATCH MORE BASS THIS YEAR FROM LONG LAKE THAN SILVER LAKE ON AVERAGE?
alpha = 0.05
t_computed, p_value = stats.ttest_ind(long_fc_ser, silver_fc_ser)
print("t-computed:", t_computed, "p-value:", p_value)
if p_value < alpha: 
    print("Reject H0, p-value:", p_value)
else:
    print("Fail to reject H0, p-value:", p_value)










