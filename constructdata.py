import os
import pandas as pd


def constuct_time_factors(time_data):
    time_data = time_data.copy()
    date = time_data["COLLISION_DATE"]
    time = time_data["COLLISION_TIME"]

    month_list = []
    season_list = []
    for d in date:
        date_string_list = d.split("-")
        month = int(date_string_list[1])
        month_list.append(month)
        if 2 < month < 6:
            season = 1
        elif 5 < month < 9:
            season = 2
        elif 8 < month < 12:
            season = 3
        else:
            season = 4
        season_list.append(season)
    time_data.loc[:, "MONTH"] = month_list
    time_data.loc[:, "SEASON"] = season_list

    hour_list = []
    for i in time:
        hour = i // 100
        hour_list.append(hour)
    time_data.loc[:, "24HOURS"] = hour_list

    return time_data


if __name__ == "__main__":
    read_path = "D:\\factors_analysis\\test_data"
    save_path = "D:\\factors_analysis\\constructed_test_data"
    selected_factors = ["ACCIDENT_YEAR", "COLLISION_DATE", "COLLISION_TIME", "DAY_OF_WEEK", "INTERSECTION",
                        "WEATHER_1", "STATE_HWY_IND", "ROAD_SURFACE", "ROAD_COND_1", "LIGHTING",
                        "STWD_VEHTYPE_AT_FAULT", "CHP_VEHTYPE_AT_FAULT", "DIRECTION", "PCF_VIOL_CATEGORY",
                        "TYPE_OF_COLLISION", "PEDESTRIAN_ACCIDENT", "BICYCLE_ACCIDENT", "MOTORCYCLE_ACCIDENT",
                        "TRUCK_ACCIDENT", "MVIW", "TOW_AWAY", "HIT_AND_RUN", "CONTROL_DEVICE", "NUMBER_INJURED",
                        "COUNT_PED_INJURED", "COUNT_BICYCLIST_INJURED", "COUNT_MC_INJURED", "PCF_VIOLATION",
                        "PARTY_COUNT", "POINT_Y", "POINT_X", "COLLISION_SEVERITY"]
    files_name = os.listdir(read_path)
    for name in files_name:
        file_path = os.path.join(read_path, name)
        df = pd.read_csv(file_path)
        data = df[selected_factors]
        time_data = data[["ACCIDENT_YEAR", "COLLISION_DATE", "COLLISION_TIME"]]
        time_factors = constuct_time_factors(time_data)
        other_factors = data.iloc[:, 3:32].copy()
        constructed_data = pd.concat([time_factors, other_factors], axis=1)
        print(constructed_data)
        # output_path = os.path.join(save_path, name)
        # constructed_data.to_csv(output_path, index=False)
