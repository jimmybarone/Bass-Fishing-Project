import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

df = pd.read_csv("Bass_Fishing_Data_Cleaned.csv")

# df2 = pd.read_csv("new_daily_weather.csv")
# # print(df2.isnull().sum())
# #used ^ to find columns with more than 182(50%) missing values 
# #columns with more than 50%: snow, wpgt, tsun

# df2.drop("snow", axis=1, inplace=True)
# df2.drop("wpgt", axis=1, inplace=True)
# df2.drop("tsun", axis=1, inplace=True)

# # print(df2.isnull().sum())
# df2.replace("", np.nan, inplace=True)
# df2.interpolate(method='linear', axis=0, inplace=True)
# df2.to_csv("SpokaneWA_daily_weather_cleaned.csv")

df3 = pd.read_csv("Spokane_daily_weather_cleaned.csv")
merged_df = df.merge(df3, on="Date")
merged_df.to_csv("merged_df.csv")

merged_df.replace(to_replace="", value=np.nan, inplace=True)
merged_df.drop("Unnamed: 0", axis=1, inplace=True)
# print(merged_df.head())
merged_df["Fish over 3.5lbs"].replace(to_replace=2.0, value=1, inplace=True)
merged_df["Fish over 3.5lbs"].replace(to_replace=1.0, value=1, inplace=True)
merged_df["Fish over 3.5lbs"].replace(to_replace=0.0, value=0, inplace=True)
# merged_df.to_csv("merged_df_cleaned.csv")
# merged_df["Fish over 3.5lbs"].replace(to_replace=np.NaN, value=1, inplace=True)
# print(merged_df["Fish over 3.5lbs"])
merged_df = pd.read_csv("merged_df_cleaned.csv", index_col=0)


water_temp_df = merged_df.groupby("Water Temp")
fish_caught_df = merged_df.groupby("Fish Caught")
lake_name_df = merged_df.groupby("Lake")
water_temp_ser = merged_df["Water Temp"]
fish_caught_ser = merged_df["Fish Caught"]
lake_name_ser = merged_df["Lake"]
Lake_Name = input("Input lake name:")

def WT_vs_FC_Plot(lake_name_df, Lake_Name):
    Lake_Name_df = lake_name_df.get_group("Lake_Name")
    Lake_Name_wt_ser = Lake_Name_df["Water Temp"]
    Lake_Name_fc_ser = Lake_Name_df["Fish Caught"]
    plt.figure()
    plt.scatter(Lake_Name_wt_ser, Lake_Name_fc_ser)
    plt.xlabel("Water Temperature")
    plt.ylabel("Fish Caught by Trip")
    plt.savefig("newman_lake_fish_caught_vs_water_temp.pdf")
    # plt.show

# def Eloika_WT_vs_FC_Plot():
#     Eloika_Lake_df = lake_name_df.get_group("Eloika Lake")
#     eloika_wt_ser = Eloika_Lake_df["Water Temp"]
#     eloika_fc_ser = Eloika_Lake_df["Fish Caught"]
#     plt.figure()
#     plt.scatter(eloika_wt_ser, eloika_fc_ser)
#     plt.xlabel("Water Temperature")
#     plt.ylabel("Fish Caught by Trip")
#     plt.savefig("eloika_lake_fish_caught_vs_water_temp.pdf")
#     # plt.show

# def Long_WT_vs_FC_Plot():
#     Long_Lake_df = lake_name_df.get_group("Long Lake")
#     long_wt_ser = Long_Lake_df["Water Temp"]
#     long_fc_ser = Long_Lake_df["Fish Caught"]
#     long_5fb_ser = Long_Lake_df["5 Fish Bag"]
#     plt.figure()
#     plt.scatter(long_wt_ser, long_fc_ser)
#     plt.xlabel("Water Temperature")
#     plt.ylabel("Fish Caught by Trip")
#     plt.savefig("long_lake_fish_caught_vs_water_temp.pdf")
#     # plt.show

# def Liberty_WT_vs_FC_Plot():
#     Liberty_Lake_df = lake_name_df.get_group("Liberty Lake")
#     liberty_wt_ser = Liberty_Lake_df["Water Temp"]
#     liberty_fc_ser = Liberty_Lake_df["Fish Caught"]
#     plt.figure()
#     plt.scatter(liberty_wt_ser, liberty_fc_ser)
#     plt.xlabel("Water Temperature")
#     plt.ylabel("Fish Caught by Trip")
#     plt.savefig("liberty_lake_fish_caught_vs_water_temp.pdf")
#     # plt.show

# def Silver_WT_vs_FC_Plot():
#     Silver_Lake_df = lake_name_df.get_group("Silver Lake")
#     silver_wt_ser = Silver_Lake_df["Water Temp"]
#     silver_fc_ser = Silver_Lake_df["Fish Caught"]
#     silver_f5b_ser = Silver_Lake_df["5 Fish Bag"]
#     plt.figure()
#     plt.scatter(silver_wt_ser, silver_fc_ser)
#     plt.xlabel("Water Temperature")
#     plt.ylabel("Fish Caught by Trip")
#     plt.savefig("silver_lake_fish_caught_vs_water_temp.pdf")
#     # plt.show

# def Bonnie_WT_vs_FC_Plot():
#     Bonnie_Lake_df = lake_name_df.get_group("Bonnie Lake")
#     bonnie_wt_ser = Bonnie_Lake_df["Water Temp"]
#     bonnie_fc_ser = Bonnie_Lake_df["Fish Caught"]
#     plt.figure()
#     plt.scatter(bonnie_wt_ser, bonnie_fc_ser)
#     plt.xlabel("Water Temperature")
#     plt.ylabel("Fish Caught by Trip")
#     plt.savefig("bonnie_lake_fish_caught_vs_water_temp.pdf")
#     # plt.show

# def Loon_WT_vs_FC_Plot():
#     Loon_Lake_df = lake_name_df.get_group("Loon Lake")
#     loon_wt_ser = Loon_Lake_df["Water Temp"]
#     loon_fc_ser = Loon_Lake_df["Fish Caught"]
#     plt.figure()
#     plt.scatter(loon_wt_ser, loon_fc_ser)
#     plt.xlabel("Water Temperature")
#     plt.ylabel("Fish Caught by Trip")
#     plt.savefig("loon_lake_fish_caught_vs_water_temp.pdf")
#     # plt.show

# def Sacheen_WT_vs_FC_Plot():
#     Sacheen_Lake_df = lake_name_df.get_group("Sacheen Lake")
#     sacheen_wt_ser = Sacheen_Lake_df["Water Temp"]
#     sacheen_fc_ser = Sacheen_Lake_df["Fish Caught"]
#     plt.figure()
#     plt.scatter(sacheen_wt_ser, sacheen_fc_ser)
#     plt.xlabel("Water Temperature")
#     plt.ylabel("Fish Caught by Trip")
#     plt.savefig("sacheen_lake_fish_caught_vs_water_temp.pdf")
#     # plt.show

# # Pend_Orielle_River_df = lake_name_df.get_group("Pend Orielle River")
# # Snake_River_df = lake_name_df.get_group("Snake River")

# def All_Lakes_WT_vs_FC_Plot():
#     plt.figure()
#     plt.scatter(newman_wt_ser, newman_fc_ser, label="Newman")
#     plt.scatter(eloika_wt_ser, eloika_fc_ser, label="Eloika")
#     plt.scatter(long_wt_ser, long_fc_ser, label="Long")
#     plt.scatter(liberty_wt_ser, liberty_fc_ser, label="Liberty")
#     plt.scatter(silver_wt_ser, silver_fc_ser, label="Silver")
#     plt.scatter(bonnie_wt_ser, bonnie_fc_ser, label="Bonnie")
#     plt.scatter(loon_wt_ser, loon_fc_ser, label="Loon")
#     plt.scatter(sacheen_wt_ser, sacheen_fc_ser, label="Sacheen")
#     plt.xlabel("Water Temperature")
#     plt.ylabel("Fish Caught by Trip")
#     plt.legend()
#     plt.savefig("watertempvsfishcaught.pdf")

# def Long_vs_Silver_Test():
#     #DID I CATCH MORE BASS THIS YEAR FROM LONG LAKE THAN SILVER LAKE ON AVERAGE?
#     alpha = 0.05
#     t_computed, p_value = stats.ttest_ind(long_fc_ser, silver_fc_ser)
#     print("t-computed:", t_computed, "p-value:", p_value)
#     if p_value < alpha: 
#         print("Reject H0, p-value:", p_value)
#     else:
#         print("Fail to reject H0, p-value:", p_value)

# def Bigs_vs_Pressure_Test():
#     #IS THERE A SIG DIFFERENCE IN PRESSURE ON DAYS WHEN i CATCH A BIG FISH AND DAYS WHEN i DONT?
#     fish_over_3_5_df = merged_df.groupby("Fish over 3.5lbs")
#     no_big_fish_df = fish_over_3_5_df.get_group(0)
#     caught_big_fish_df = fish_over_3_5_df.get_group(1)
#     pres_big_ser= caught_big_fish_df["pres"]
#     pres_no_big_ser = no_big_fish_df["pres"]
#     alpha = 0.05
#     t_computed, p_value = stats.ttest_ind(pres_big_ser, pres_no_big_ser)
#     print("t-computed:", t_computed, "p-value:", p_value)
#     if p_value < alpha: 
#         print("Reject H0, p-value:", p_value)
#     else:
#         print("Fail to reject H0, p-value:", p_value)

# def Bag_vs_WT_Test():
#     #IS THERE A SIG DIFFERENCE IN WATER TEMP ON DAYS WHEN i CATCH A BAG OF FISH OVER 10LBS AND DAYS WHEN i DONT?
#     # bag_df = merged_df.groupby("5 Fish Bag")
#     bag_over_10_df = merged_df[merged_df["5 Fish Bag"] > 10]
#     bag_under_10_df = merged_df[merged_df["5 Fish Bag"] <= 10]
#     wt_over_10_ser = bag_over_10_df["Water Temp"]
#     wt_under_10_ser = bag_under_10_df["Water Temp"]
#     alpha = 0.05
#     t_computed, p_value = stats.ttest_ind(wt_over_10_ser, wt_under_10_ser)
#     print("t-computed:", t_computed, "p-value:", p_value)
#     if p_value < alpha: 
#         print("Reject H0, p-value:", p_value)
#     else:
#         print("Fail to reject H0, p-value:", p_value)

# def Long_Vs_Silver_Bags_Test():
#     #IS THERE A SIG DIFFERENCE IN THE AVERAGE SIZE BAG I CAUGHT FROM SILVER LAKE AND LONG LAKE?
#     alpha = 0.05
#     t_computed, p_value = stats.ttest_ind(silver_f5b_ser, long_5fb_ser)
#     print("t-computed:", t_computed, "p-value:", p_value)
#     if p_value < alpha: 
#         print("Reject H0, p-value:", p_value)
#     else:
#         print("Fail to reject H0, p-value:", p_value)

# def AM_vs_PM_Test():
#     #IS THERE A SIG DIFFERENCE IN THE AMOUNT OF FISH I CAUGHT IN THE AM AND THE PM THIS YEAR?
#     am_fish_ser = merged_df["Fish Caught AM"]
#     pm_fish_ser = merged_df["Fish Caught PM"]
#     alpha = 0.05
#     t_computed, p_value = stats.ttest_ind(am_fish_ser, pm_fish_ser)
#     print("t-computed:", t_computed, "p-value:", p_value)
#     if p_value < alpha: 
#         print("Reject H0, p-value:", p_value)
#     else:
#         print("Fail to reject H0, p-value:", p_value)

# def Big_Fish_Dec_Tree():
#     le = preprocessing.LabelEncoder()
#     merged_df["Lake"] = le.fit_transform(merged_df["Lake"])
#     merged_df["Lake Type"] = le.fit_transform(merged_df["Lake Type"])

#     y = merged_df["Fish over 3.5lbs"]
#     X = merged_df.drop(["Fish over 3.5lbs", "Date", "5 Fish Bag", "Fish Caught", "Fish Caught AM", "Fish Caught PM"], axis=1)
#     tree_clf = DecisionTreeClassifier(random_state=0, max_depth=3)
#     tree_clf.fit(X, y)
#     plt.figure(figsize = (15, 10))
#     plot_tree(tree_clf, feature_names=X.columns, class_names={1: "Big Fish Will Be Caught!", 0: "No Big Fish Today :("}, filled=True)
#     plt.savefig("Big_Fish_Decision_Tree.pdf")

#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y, test_size=0.33)
#     # print(X_test)
#     # print(y_test)
#     scaler = MinMaxScaler()
#     scaler.fit(X_train)
#     X_train_normalized = scaler.transform(X_train)
#     # print(X_train_normalized)
#     X_test_normalized = scaler.transform(X_test)
#     # print(X_test_normalized)



#     dec_tree_clf = DecisionTreeClassifier(random_state=0, max_depth=3)
#     dec_tree_clf.fit(X_test_normalized, y_test)
#     y_predicted = dec_tree_clf.predict(X_test_normalized)
#     dec_tree_acc = dec_tree_clf.score(X_test_normalized, y_test)
#     print("accuracy:", dec_tree_acc)

# def Big_Fish_kNN():
#     knn_clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
#     knn_clf.fit(X_train_normalized,  y_train)
#     y_predicted = knn_clf.predict(X_test_normalized)
#     print("y predicted:", y_predicted)
#     print("nearest neighbors:", knn_clf.kneighbors(X_test_normalized))

#     acc = knn_clf.score(X_test_normalized, y_test)
#     print("accuracy:", acc)





