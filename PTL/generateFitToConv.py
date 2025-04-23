
def load_file(path):
        #load the file and get the start and end times
        reader = pydarnio.SDarnRead(path)
        records = reader.read_fitacf()
        #get the radar name from the filename
        #get the filename
        #add to the dataframe
        return records

def process_fitacf_to_filelists(folder):
    #iterate through self.generator, and make a note of files by time range,
    # and radar, make note only of all the files that span a particular time range 
    #where there's coverage from all radars.
    #this will be used to create a list of files to process
    #step 1: create a dataframe with the start and end times of each file and the radar
    dataframe= pd.DataFrame(columns=["start_time", "end_time", "radar", "filename"])
    
      
    
    #step 2: go through the generator and create entries in the dataframe
    for file in os.listdir(folder):
        #get the start and end times of the file
        records=load_file(file)
        radar_name = file.split("/")[-1].split(".")[-2] #?
        times=[(datetime.datetime(record["start.year"], record["start.month"], record["start.day"], record["start.hour"], record["start.minute"]),
                    datetime.datetime(record["end.year"], record["end.month"], record["end.day"], record["end.hour"], record["end.minute"]))
                    for record in records if "start.year" in record and "end.year" in record]
        startTimes, endTimes = zip(*times)
        minStartTime = min(startTimes)
        maxEndTime = max(endTimes)
        #add entry to the dataframe
        entry={"start_time": minStartTime, "end_time": maxEndTime, "radar": radar_name, "filename": obj.object_name}
        dataframe=dataframe.append(entry, ignore_index=True)
    #step 3: create a list of time ranges 
    radar_names = set(dataframe["radar"].unique().tolist())
    df_t_min = dataframe["start_time"].min()
    df_t_max = dataframe["end_time"].max()
    time_points= pd.date_range(df_t_min, df_t_max, freq="10min")
    #step 4: filter time ranges to only those that have coverage from all radars
    #for each time point, check if all radars are present
    file_sets = []
    for time_point in time_points:
        #get the entries that are within the time range
        entries = dataframe[(dataframe["start_time"] <= time_point) & (dataframe["end_time"] >= time_point)]
        #check if all radars are present
        radar_names_in_entries = set(entries["radar"].unique().tolist())
        if radar_names_in_entries == radar_names:
            #add the time range to the list
            datapoint={"time":time_point, "files": tuple(sort(entries["filename"].tolist()))}
            file_sets.append(datapoint)

    #step 5: create a list of files to process based on the time ranges

    #step 5.1: sort the file sets by time
    file_sets.sort(key=lambda x: x["time"])
    #step 5.2: set the object to read the file_sets instead of the generator
    
    
    #see moreat https://arxiv.org/abs/2301.00250

    return file_sets

def find_conv_maps_from_filelists(folder, file_sets):
    #open folder of convmap files,
    #create a dataframe with the start and end times of each file and the file name, 

    #for each file_set, update with the file name of the convmap file that includes the time range
    #step 1: create a dataframe with the start and end times of each file and the file name
    dataframe= pd.DataFrame(columns=["start_time", "end_time", "filename"])
    #step 2: go through the generator and create entries in the dataframe
    for file in os.listdir(folder):
        #get the start and end times of the file
        records=load_file(file)
        times=[(datetime.datetime(record["start.year"], record["start.month"], record["start.day"], record["start.hour"], record["start.minute"]),
                    datetime.datetime(record["end.year"], record["end.month"], record["end.day"], record["end.hour"], record["end.minute"]))
                    for record in records if "start.year" in record and "end.year" in record]
        startTimes, endTimes = zip(*times)
        minStartTime = min(startTimes)
        maxEndTime = max(endTimes)
        #add entry to the dataframe
        entry={"start_time": minStartTime, "end_time": maxEndTime, "filename": file}
        dataframe=dataframe.append(entry, ignore_index=True)
    #step 3: for each file_set, update with the file name of the convmap file that includes the time range
    files_to_process = []
    for file_set in file_sets:
        #get the entries that are within the time range
        entries = dataframe[(dataframe["start_time"] <= file_set["time"]) & (dataframe["end_time"] >= file_set["time"])]
        #check if there are any entries
        if len(entries) > 0:
            #add the file name to the file_set
            file_set["convmap_file"] = entries.iloc[0]["filename"]
            files_to_process.append(file_set)
        else:
            print(f"No convmap file found for time {file_set['time']}")
    
    return files_to_process, file_sets


#create a datamodule from 2 folders, that returns file sets for a ML model that takes the fitacf files and predicts the convmap files