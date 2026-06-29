import pandas as pd

rosello2015 = pd.read_csv("rosello2015_supplementary1.csv")
# print(rosello2015.info())
print(rosello2015["Outbreak"].value_counts())
print(rosello2015["Year_outbreak"].value_counts())
