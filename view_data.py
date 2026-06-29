import pandas as pd

rosello2015 = pd.read_csv("Data/rosello2015_supplementary1.csv")
# print(rosello2015.info())
print(rosello2015["Outbreak"].value_counts())
print(rosello2015["Year_outbreak"].value_counts())

sierraleone2014 = pd.read_csv("Data/ebola_sierraleone_2014_confirmed.csv")
print(sierraleone2014.info())

uganda2022 = pd.read_csv("Data/Uganda2022globaldothealth.csv")
print(uganda2022.info())

DRC2018 = pd.read_csv("Data/DRC2018_humdata_MOH-Total.csv")
print(DRC2018.info())