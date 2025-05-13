def load_file(path):
    """
    Loads a file and extracts radar records.

    Args:
        path (str): Path to the file to load.

    Returns:
        list: A list of radar records extracted from the file.
    """
    # Load the file and get the start and end times
    reader = pydarnio.SDarnRead(path)
    records = reader.read_fitacf()
    return records


def process_fitacf_to_filelists(folder):
    """
    Processes FITACF files in a folder to create a list of file sets based on time ranges and radar coverage.

    Args:
        folder (str): Path to the folder containing FITACF files.

    Returns:
        list: A list of file sets, each containing time ranges and associated radar files.
    """
    dataframe = pd.DataFrame(columns=["start_time", "end_time", "radar", "filename"])

    # Step 2: Go through the folder and create entries in the dataframe
    for file in os.listdir(folder):
        records = load_file(file)
        radar_name = file.split("/")[-1].split(".")[-2]
        times = [
            (
                datetime.datetime(record["start.year"], record["start.month"], record["start.day"], record["start.hour"], record["start.minute"]),
                datetime.datetime(record["end.year"], record["end.month"], record["end.day"], record["end.hour"], record["end.minute"])
            )
            for record in records if "start.year" in record and "end.year" in record
        ]
        startTimes, endTimes = zip(*times)
        minStartTime = min(startTimes)
        maxEndTime = max(endTimes)
        entry = {"start_time": minStartTime, "end_time": maxEndTime, "radar": radar_name, "filename": file}
        dataframe = dataframe.append(entry, ignore_index=True)

    # Step 3: Create a list of time ranges
    radar_names = set(dataframe["radar"].unique().tolist())
    df_t_min = dataframe["start_time"].min()
    df_t_max = dataframe["end_time"].max()
    time_points = pd.date_range(df_t_min, df_t_max, freq="10min")

    # Step 4: Filter time ranges to only those that have coverage from all radars
    file_sets = []
    for time_point in time_points:
        entries = dataframe[(dataframe["start_time"] <= time_point) & (dataframe["end_time"] >= time_point)]
        radar_names_in_entries = set(entries["radar"].unique().tolist())
        if radar_names_in_entries == radar_names:
            datapoint = {"time": time_point, "files": tuple(sorted(entries["filename"].tolist()))}
            file_sets.append(datapoint)

    # Step 5: Sort the file sets by time
    file_sets.sort(key=lambda x: x["time"])

    return file_sets


def find_conv_maps_from_filelists(folder, file_sets):
    """
    Finds corresponding CONVMAP files for a list of FITACF file sets.

    Args:
        folder (str): Path to the folder containing CONVMAP files.
        file_sets (list): A list of file sets containing time ranges and associated radar files.

    Returns:
        tuple: A tuple containing:
            - files_to_process (list): A list of file sets with associated CONVMAP files.
            - file_sets (list): The original file sets with updated CONVMAP file information.
    """
    dataframe = pd.DataFrame(columns=["start_time", "end_time", "filename"])

    # Step 2: Go through the folder and create entries in the dataframe
    for file in os.listdir(folder):
        records = load_file(file)
        times = [
            (
                datetime.datetime(record["start.year"], record["start.month"], record["start.day"], record["start.hour"], record["start.minute"]),
                datetime.datetime(record["end.year"], record["end.month"], record["end.day"], record["end.hour"], record["end.minute"])
            )
            for record in records if "start.year" in record and "end.year" in record
        ]
        startTimes, endTimes = zip(*times)
        minStartTime = min(startTimes)
        maxEndTime = max(endTimes)
        entry = {"start_time": minStartTime, "end_time": maxEndTime, "filename": file}
        dataframe = dataframe.append(entry, ignore_index=True)

    # Step 3: For each file set, update with the CONVMAP file name
    files_to_process = []
    for file_set in file_sets:
        entries = dataframe[(dataframe["start_time"] <= file_set["time"]) & (dataframe["end_time"] >= file_set["time"])]
        if len(entries) > 0:
            file_set["convmap_file"] = entries.iloc[0]["filename"]
            files_to_process.append(file_set)
        else:
            print(f"No convmap file found for time {file_set['time']}")

    return files_to_process, file_sets