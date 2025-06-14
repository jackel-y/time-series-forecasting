import numpy as np
import pandas as pd
from typing import Tuple

from scipy.stats import shapiro
import scipy.stats as stats

def data_preprocessing(
    df: pd.DataFrame,
    test: pd.DataFrame,
    transactions: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs the SQL queries provided and obtains the necessary data for
    creating the model.
    Returns:
        data (pd.DataFrame): Pandas dataframe with the results of the SQL
        queries required for modelling.
        demo_dedupe (pd.DataFrame): Pandas dataframe with membership_txt along with their demogrpahics data, deduplicated
    Raises:
        ValueError : data has no rows populated
    """

    # Datetime
    df["date"] = pd.to_datetime(df.date)
    test["date"] = pd.to_datetime(test.date)
    transactions["date"] = pd.to_datetime(transactions.date)

    # Data types
    df.onpromotion = df.onpromotion.astype("float16")
    df.sales = df.sales.astype("float32")
    stores.cluster = stores.cluster.astype("int8")

    print(f"Original Data: {df.shape}")
    # Remove data for stores prior to their opening 
    df = df[~((df.store_nbr == 52) & (df.date < "2017-04-20"))]
    df = df[~((df.store_nbr == 22) & (df.date < "2015-10-09"))]
    df = df[~((df.store_nbr == 42) & (df.date < "2015-08-21"))]
    df = df[~((df.store_nbr == 21) & (df.date < "2015-07-24"))]
    df = df[~((df.store_nbr == 29) & (df.date < "2015-03-20"))]
    df = df[~((df.store_nbr == 20) & (df.date < "2015-02-13"))]
    df = df[~((df.store_nbr == 53) & (df.date < "2014-05-29"))]
    df = df[~((df.store_nbr == 36) & (df.date < "2013-05-09"))]
    
    print(f"Removed old store data: {df.shape}")
    
    # Remove product categories not sold in specific stores
    c = df.groupby(["store_nbr", "family"]).sales.sum().reset_index().sort_values(["family","store_nbr"])
    c = c[c.sales == 0]

    outer_join = df.merge(c[c.sales == 0].drop("sales",axis = 1), how = 'outer', indicator = True)
    df = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis = 1)

    print(f"Removed product categories not sold: {df.shape}")

    # Create a zero data prediction for product categories not sold in specific spects
    
    zero_prediction = []
    for i in range(0,len(c)):
        zero_prediction.append(
            pd.DataFrame({
                "date":pd.date_range("2017-08-16", "2017-08-31").tolist(),
                "store_nbr":c.store_nbr.iloc[i],
                "family":c.family.iloc[i],
                "sales":0
            })
        )
    zero_prediction = pd.concat(zero_prediction)

    oil["date"] = pd.to_datetime(oil.date)
    # Resample
    oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
    # Interpolate
    oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
    oil["dcoilwtico_interpolated"] =oil.dcoilwtico.interpolate() ##by default a linear interpolation before and after 

    return df, zero_prediction, test, transactions, stores, oil

def feature_engineering(
    df: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame
) -> pd.DataFrame:
    """Runs the SQL queries provided and obtains the necessary data for
    creating the model.
    Returns:
        data (pd.DataFrame): Pandas dataframe with the results of the SQL
        queries required for modelling.
        demo_dedupe (pd.DataFrame): Pandas dataframe with membership_txt along with their demogrpahics data, deduplicated
    Raises:
        ValueError : data has no rows populated
    """

    df["date"] = pd.to_datetime(df.date)




    # Transferred Holidays
    tr1 = df[(df.type == "Holiday") & (df.transferred == True)].drop("transferred", axis = 1).reset_index(drop = True)
    tr2 = df[(df.type == "Transfer")].drop("transferred", axis = 1).reset_index(drop = True)
    tr = pd.concat([tr1,tr2], axis = 1)
    tr = tr.iloc[:, [5,1,2,3,4]]

    df = df[(df.transferred == False) & (df.type != "Transfer")].drop("transferred", axis = 1)
    df = pd.concat([df,tr]).reset_index(drop = True)


    # Additional Holidays
    df["description"] = df["description"].str.replace("-", "").str.replace("+", "").str.replace('\d+', '')
    df["type"] = np.where(df["type"] == "Additional", "Holiday", df["type"])

    # Bridge Holidays
    df["description"] = df["description"].str.replace("Puente ", "")
    df["type"] = np.where(df["type"] == "Bridge", "Holiday", df["type"])

    
    # Work Day Holidays, that is meant to payback the Bridge.
    work_day = df[df.type == "Work Day"]  
    df = df[df.type != "Work Day"]  


    # Split

    # Events are national
    events = df[df.type == "Event"].drop(["type", "locale", "locale_name"], axis = 1).rename({"description":"events"}, axis = 1)

    df = df[df.type != "Event"].drop("type", axis = 1)
    regional = df[df.locale == "Regional"].rename({"locale_name":"state", "description":"holiday_regional"}, axis = 1).drop("locale", axis = 1).drop_duplicates()
    national = df[df.locale == "National"].rename({"description":"holiday_national"}, axis = 1).drop(["locale", "locale_name"], axis = 1).drop_duplicates()
    local = df[df.locale == "Local"].rename({"description":"holiday_local", "locale_name":"city"}, axis = 1).drop("locale", axis = 1).drop_duplicates()



    d = pd.merge(pd.concat([train,test]), stores)
    d["store_nbr"] = d["store_nbr"].astype("int8")


    # National Holidays & Events
    #d = pd.merge(d, events, how = "left")
    d = pd.merge(d, national, how = "left")
    # Regional
    d = pd.merge(d, regional, how = "left", on = ["date", "state"])
    # Local
    d = pd.merge(d, local, how = "left", on = ["date", "city"])

    # Work Day: It will be removed when real work day colum created
    d = pd.merge(d,  work_day[["date", "type"]].rename({"type":"IsWorkDay"}, axis = 1),how = "left")

    # EVENTS
    events["events"] =np.where(events.events.str.contains("futbol"), "Futbol", events.events)

    def one_hot_encoder(df, nan_as_category=True):
        original_columns = list(df.columns)
        categorical_columns = df.select_dtypes(["category", "object"]).columns.tolist()
        # categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
        new_columns = [c for c in df.columns if c not in original_columns]
        df.columns = df.columns.str.replace(" ", "_")
        return df, df.columns.tolist()

    events, events_cat = one_hot_encoder(events, nan_as_category=False)
    events["events_Dia_de_la_Madre"] = np.where(events.date == "2016-05-08", 1,events["events_Dia_de_la_Madre"])
    events = events.drop(239)

    d = pd.merge(d, events, how = "left")
    d[events_cat] = d[events_cat].fillna(0)

    # New features
    d["holiday_national_binary"] = np.where(d.holiday_national.notnull(), 1, 0)
    d["holiday_local_binary"] = np.where(d.holiday_local.notnull(), 1, 0)
    d["holiday_regional_binary"] = np.where(d.holiday_regional.notnull(), 1, 0)

    # 
    d["national_independence"] = np.where(d.holiday_national.isin(['Batalla de Pichincha',  'Independencia de Cuenca', 'Independencia de Guayaquil', 'Independencia de Guayaquil', 'Primer Grito de Independencia']), 1, 0)
    d["local_cantonizacio"] = np.where(d.holiday_local.str.contains("Cantonizacio"), 1, 0)
    d["local_fundacion"] = np.where(d.holiday_local.str.contains("Fundacion"), 1, 0)
    d["local_independencia"] = np.where(d.holiday_local.str.contains("Independencia"), 1, 0)


    holidays, holidays_cat = one_hot_encoder(d[["holiday_national","holiday_regional","holiday_local"]], nan_as_category=False)
    d = pd.concat([d.drop(["holiday_national","holiday_regional","holiday_local"], axis = 1),holidays], axis = 1)

    he_cols = d.columns[d.columns.str.startswith("events")].tolist() + d.columns[d.columns.str.startswith("holiday")].tolist() + d.columns[d.columns.str.startswith("national")].tolist()+ d.columns[d.columns.str.startswith("local")].tolist()
    d[he_cols] = d[he_cols].astype("int8")

    d[["family", "city", "state", "type"]] = d[["family", "city", "state", "type"]].astype("category")

    del holidays, holidays_cat, work_day, local, regional, national, events, events_cat, tr, tr1, tr2, he_cols


    def AB_Test(dataframe, group, target):
        
        # Split A/B
        groupA = dataframe[dataframe[group] == 1][target]
        groupB = dataframe[dataframe[group] == 0][target]
        
        # Assumption: Normality
        ntA = shapiro(groupA)[1] < 0.05
        ntB = shapiro(groupB)[1] < 0.05
        # H0: Distribution is Normal! - False
        # H1: Distribution is not Normal! - True
        
        if (ntA == False) & (ntB == False): # "H0: Normal Distribution"
            # Parametric Test
            # Assumption: Homogeneity of variances
            leveneTest = stats.levene(groupA, groupB)[1] < 0.05
            # H0: Homogeneity: False
            # H1: Heterogeneous: True
            
            if leveneTest == False:
                # Homogeneity
                ttest = stats.ttest_ind(groupA, groupB, equal_var=True)[1]
                # H0: M1 == M2 - False
                # H1: M1 != M2 - True
            else:
                # Heterogeneous
                ttest = stats.ttest_ind(groupA, groupB, equal_var=False)[1]
                # H0: M1 == M2 - False
                # H1: M1 != M2 - True
        else:
            # Non-Parametric Test
            ttest = stats.mannwhitneyu(groupA, groupB)[1] 
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
            
        # Result
        temp = pd.DataFrame({
            "AB Hypothesis":[ttest < 0.05], 
            "p-value":[ttest]
        })
        temp["Test Type"] = np.where((ntA == False) & (ntB == False), "Parametric", "Non-Parametric")
        temp["AB Hypothesis"] = np.where(temp["AB Hypothesis"] == False, "Fail to Reject H0", "Reject H0")
        temp["Comment"] = np.where(temp["AB Hypothesis"] == "Fail to Reject H0", "A/B groups are similar!", "A/B groups are not similar!")
        temp["Feature"] = group
        temp["GroupA_mean"] = groupA.mean()
        temp["GroupB_mean"] = groupB.mean()
        temp["GroupA_median"] = groupA.median()
        temp["GroupB_median"] = groupB.median()
        
        # Columns
        if (ntA == False) & (ntB == False):
            temp["Homogeneity"] = np.where(leveneTest == False, "Yes", "No")
            temp = temp[["Feature","Test Type", "Homogeneity","AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]
        else:
            temp = temp[["Feature","Test Type","AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]
        
        # Print Hypothesis
        # print("# A/B Testing Hypothesis")
        # print("H0: A == B")
        # print("H1: A != B", "\n")
        
        return temp
        
    # Apply A/B Testing
    he_cols = d.columns[d.columns.str.startswith("events")].tolist() + d.columns[d.columns.str.startswith("holiday")].tolist() + d.columns[d.columns.str.startswith("national")].tolist()+ d.columns[d.columns.str.startswith("local")].tolist()
    ab = []
    for i in he_cols:
        ab.append(AB_Test(dataframe=d[d.sales.notnull()], group = i, target = "sales"))
    ab = pd.concat(ab)

    l = ab[ab['AB Hypothesis']=='Fail to Reject H0'].Feature.tolist()

    # Identify columns to drop that exist in the DataFrame
    columns_to_drop = [col for col in l if col in d.columns]

    print(f"Data shape with all holiday_dates features: {d.shape}")

    # Remove the columns
    d = d.drop(columns=columns_to_drop)

    print(f"Data shape with some holiday_dates features removed: {d.shape}")
    print() 

    print(f"The following columns has been removed:")

    for col in l:
        print(col)


    # Time Related Features
    def create_date_features(df):
        df['month'] = df.date.dt.month.astype("int8")
        df['day_of_month'] = df.date.dt.day.astype("int8")
        df['day_of_year'] = df.date.dt.dayofyear.astype("int16")
        df['week_of_month'] = (df.date.apply(lambda d: (d.day-1) // 7 + 1)).astype("int8")
        df['week_of_year'] = (df.date.dt.isocalendar().week).astype("int8")
        df['day_of_week'] = (df.date.dt.dayofweek + 1).astype("int8")
        df['year'] = df.date.dt.year.astype("int32")
        df["is_wknd"] = (df.date.dt.weekday // 4).astype("int8")
        df["quarter"] = df.date.dt.quarter.astype("int8")
        df['is_month_start'] = df.date.dt.is_month_start.astype("int8")
        df['is_month_end'] = df.date.dt.is_month_end.astype("int8")
        df['is_quarter_start'] = df.date.dt.is_quarter_start.astype("int8")
        df['is_quarter_end'] = df.date.dt.is_quarter_end.astype("int8")
        df['is_year_start'] = df.date.dt.is_year_start.astype("int8")
        df['is_year_end'] = df.date.dt.is_year_end.astype("int8")
        # 0: Winter - 1: Spring - 2: Summer - 3: Fall
        df["season"] = np.where(df.month.isin([12,1,2]), 0, 1)
        df["season"] = np.where(df.month.isin([6,7,8]), 2, df["season"])
        df["season"] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df["season"])).astype("int8")
        return df
    d = create_date_features(d)

    # Workday column
    d["workday"] = np.where((d.holiday_national_binary == 1) | (d.holiday_local_binary==1) | (d.holiday_regional_binary==1) | (d['day_of_week'].isin([6,7])), 0, 1)
    d["workday"] = pd.Series(np.where(d.IsWorkDay.notnull(), 1, d["workday"])).astype("int8")
    d.drop("IsWorkDay", axis = 1, inplace = True)

    # Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. 
    # Supermarket sales could be affected by this.
    d["wageday"] = pd.Series(np.where((d['is_month_end'] == 1) | (d["day_of_month"] == 15), 1, 0)).astype("int8")

    print() 
    print(f"Data shape with holiday_dates and date features added: {d.shape}")


    a = d.sort_values(["store_nbr", "family", "date"])
    for i in [20, 30, 45, 60, 90, 120, 365, 730]:
        a["SMA"+str(i)+"_sales_lag16"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(16).values
        a["SMA"+str(i)+"_sales_lag30"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(30).values
        a["SMA"+str(i)+"_sales_lag60"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(60).values
    a[["sales"]+a.columns[a.columns.str.startswith("SMA")].tolist()].corr()

    print() 
    print(f"Data shape with holiday, dates, and SMA features added: {a.shape}")

    oil['above_70_flag'] = oil['dcoilwtico_interpolated'] > 70
    oil['above_70_flag'] = oil['above_70_flag'].astype(int)
    a = pd.merge(a, oil[['date','above_70_flag']], on='date', how='left')

    print() 
    print(f"Data shape with holiday, dates, SMA, and oil_70_flag features added: {a.shape}")

    return a