from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch.utils.data import IterableDataset, Dataset
import pydarnio
import numpy as np
import datetime
from minio import Minio
import os
import random
from tqdm import tqdm
import hashlib
import time

def gaussian(x, y, x0, y0, sigma_x, sigma_y):
    return np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)))



class SuperDARNDatasetFolder(Dataset):
    def __init__(self, *args,**kwargs):
        super().__init__()
        self.dataset= SuperDARNDataset.from_disk(*args,**kwargs)
        self.data=[file for file in self.dataset.generator()]
        self.data_dir = args[0]
        self.batch_size = kwargs.get("batch_size", 1)
    def __len__(self):
        #Return the number of batches
        return len(self.data)
    def __getitem__(self, index):
        #return the data at the index
        filename= self.data[index]
        try:
            output= self.dataset.process_file(open(os.path.join(self.data_dir, filename), 'rb'))
        except Exception as e:
            print("Error: ", e)
            output=None
        if output is None or output[0] is None or output[1] is None:
            # print("No data found for the given time range")
            output=self.__getitem__(random.randint(0, len(self)-1))
        return output
    

class SuperDARNDataset(IterableDataset):

    @classmethod
    def from_disk(cls, data_dir, batch_size, method='flat', window_size=10, **kwargs):
        #load the data from the disk
        self = cls.__new__(cls, {},"", batch_size, method, window_size, silent=True, **kwargs)
        
        def generator():
            import os
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(".bin") or file.endswith(".npy") or file.endswith(".txt"):
                        #ignore the files that are already processed
                        continue
                    yield file

        self.generator = generator
        def __iter__():
            for file in self.generator():
                output= self.process_file(open(os.path.join(data_dir, file), 'rb'))
                if output[0] is None or output[1] is None:
                    continue
                yield output
        self.__iter__ = __iter__
        return self
    
    def __new__(cls, miniodict, minioBucket, batch_size, method='flat', window_size=10, **kwargs):
        self = super().__new__(cls)
        self.__init__(miniodict, minioBucket, batch_size, method, window_size, **kwargs)
        return self


    def __init__(self,miniodict, minioBucket, batch_size, method='flat', window_size=10, **kwargs):
        super().__init__()
        self.minioBucket = minioBucket
        print(minioBucket)
        self.batch_size = batch_size
        self.grid_size = kwargs.get("grid_size", 300)
        self.time_step=kwargs.get("time_step", 1)
        try:
            self.minioClient = Minio(miniodict.get("host", "localhost") + ":" 
                                                + str(miniodict.get("port", 9000)),
                        access_key=miniodict.get("access_key", "minioadmin"),
                        secret_key=miniodict.get("secret_key", "minioadmin"),
                        secure=False)
            self.miniodict = miniodict    
            print("Buckets List:")
            buckets = self.minioClient.list_buckets()
            for bucket in buckets:
                print(bucket.name, bucket.creation_date, "<----- " if bucket.name == self.minioBucket else "")
        except Exception as e:
            
            if not kwargs.get("silent", False):
                raise e
        #list the buckets   

        self.window_size = window_size
        self.method = method
        if self.method == 'flat':
            self.process_conv_to_tensor = self.process_conv_to_flat
            self.process_fitacf_to_tensor = self.process_fitacf_to_flat
        elif self.method == 'grid':
            self.process_conv_to_tensor = self.process_conv_to_grid
            self.process_fitacf_to_tensor = self.process_fitacf_to_grid
        else:
            raise ValueError("method must be either 'flat' or 'grid'")
        self.location={"max_vector":88008, "max_mlat":0,
                      "max_mlon":0, 
                      "min_mlat":360,
                      "min_mlon":360}
        self.first_epoch=True


    def generator(self):   
        #return an iterator over the objects in the bucket
        for obj in self.minioClient.list_objects(self.minioBucket):
            yield obj

    def check_self_location(self,data):
        #check if the record is within the location bounds
        if not self.first_epoch:
            return
        vector_indexes=[record["vector.index"] for record in data if "vector.index" in record]
        #stack as one big array
        if len(vector_indexes) == 0:
            return
        vector_indexes=np.concatenate(vector_indexes)
        self.location["max_vector"] = np.max(vector_indexes) if len(vector_indexes) > 0 else self.location["max_vector"]
        mlat=[record["vector.mlat"] for record in data if "vector.mlat" in record]
        mlat=np.concatenate(mlat)
        self.location["max_mlat"] = np.max(mlat) if len(mlat) > 0 else self.location["max_mlat"]
        mlon=[record["vector.mlon"] for record in data if "vector.mlon" in record]
        mlon=np.concatenate(mlon)   
        self.location["max_mlon"] = np.max(mlon) if len(mlon) > 0 else self.location["max_mlon"]
        self.location["min_mlat"] = np.min(mlat) if len(mlat) > 0 else self.location["min_mlat"]
        self.location["min_mlon"] = np.min(mlon) if len(mlon) > 0 else self.location["min_mlon"]

            
    def process_fitacf_to_grid(self, data):
        pass
    def process_fitacf_to_flat(self, data):
        pass
    def process_conv_to_flat(self, data):
        #use the index location as index_select into tensor of size (num_points, x)
        data_tensor = torch.zeros(5, self.location["max_vector"]+1)
        for record in data:
            # if "vector.index" not in record:
            #     print("Index not found in record")
            #     continue
            # if np.max(record["vector.index"]) > self.location["max_vector"]:
            #     print("Index out of bounds: ", record["vector.index"])
            
            data_tensor[0, record["vector.index"]] = torch.tensor(record["vector.vel.median"])
            data_tensor[1, record["vector.index"]] = torch.tensor(record["vector.vel.sd"])
            data_tensor[2, record["vector.index"]] = torch.tensor(record["vector.kvect"])
            data_tensor[3, record["vector.index"]] = torch.tensor(record["vector.stid"]).to(torch.float)
            data_tensor[4, record["vector.index"]] = torch.tensor(record["vector.channel"]).to(torch.float)

        return data_tensor
        
    def process_conv_to_grid(self, data):
        #use the mlat, mlon to gaussian splat onto a x,y grid and stack
        # data_tensor = torch.zeros(5, self.grid_size, self.grid_size)
        #we're going to sample the data onto a 300x300 grid
        #we're going to use a gaussian kernel to splat the data onto the grid
        
        #step1: create a meshgrid
        # x = np.linspace(self.location["min_mlon"], self.location["max_mlon"], self.grid_size)
        # y = np.linspace(self.location["min_mlat"], self.location["max_mlat"], self.grid_size)
        # X, Y = np.meshgrid(x, y)
        #step 2 : create a gaussian kernel

        #step 3: splat the data onto the grid
        #make a tensor of mlat, mlon
        start_time=time.time()
        Coords=[np.stack([np.array(record["vector.mlon"]),
                 np.array(record["vector.mlat"]),
                 np.array(record["vector.vel.median"]),
                 np.array(record["vector.vel.sd"]),
                 np.array(record["vector.kvect"]),
                 np.array(record["vector.stid"]),
                 np.array(record["vector.channel"])], axis=0)
                 for record in data if "vector.mlat" in record]
        #convert each to numpy array then convert to tensor
        if len(Coords) == 0:
            return None
        Coords=np.concatenate(Coords, axis=1)
        #convert to tensor
        Coords=torch.tensor(Coords, dtype=torch.float32)
        now=time.time()
        time_dif=now-start_time
        # print("time to process coords: ", time_dif)
        start_time=time.time()
        x= Coords[0]
        y= Coords[1]
        Data=Coords[2:7].unsqueeze(1).unsqueeze(1)
        x_tensor=torch.zeros(self.grid_size,Coords.shape[1])
        x_tensor=x_tensor+torch.linspace(self.location["min_mlon"], self.location["max_mlon"], self.grid_size).reshape(-1,1)
        x=x.reshape(1,-1)

        # print("X Tensor Shape: ", x_tensor.shape)
        #both shapes are (grid_size, Coords.shape[0])
        x_diff=(x_tensor-x).pow(2) 

        y_tensor=torch.zeros(self.grid_size,Coords.shape[1])
        y_tensor=y_tensor+torch.linspace(self.location["min_mlat"], self.location["max_mlat"], self.grid_size).reshape(-1,1)
        y=y.reshape(1,-1)
        y_diff=(y_tensor-y).pow(2)
        x_diff=x_diff.reshape(self.grid_size,1,Coords.shape[1])
        y_diff=y_diff.reshape(1,self.grid_size,Coords.shape[1])
        dif=x_diff+y_diff

        dif=torch.exp(-dif/(2*1**2)).unsqueeze(0)
        #torch.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)))

        #shape is 1, G,G,B, need to combine with the data of shape 5,1,1,B to get 5,G,G
        data_tensor=Data*dif 
        data_tensor=torch.sum(data_tensor, dim=-1)

        # for record in data:
        #     if "vector.mlat" not in record:
        #         # print("vector.mlat not found in record")
        #         continue
        #     else:
        #         gj=np.array([gaussian(X, Y, record["vector.mlon"][j], record["vector.mlat"][j], 1, 1) for j in range(0, len(record["vector.mlat"]))])
        #         data_tensor[0] += (np.array(record["vector.vel.median"]).reshape(-1,1,1)*gj).sum(axis=0).reshape(self.grid_size, self.grid_size)
        #         data_tensor[1] += (np.array(record["vector.vel.sd"]).reshape(-1,1,1)*gj).sum(axis=0).reshape(self.grid_size, self.grid_size)
        #         data_tensor[2] += (np.array(record["vector.kvect"]).reshape(-1,1,1)*gj).sum(axis=0).reshape(self.grid_size, self.grid_size)
        #         data_tensor[3] += (np.array(record["vector.stid"]).reshape(-1,1,1)*gj).sum(axis=0).reshape(self.grid_size, self.grid_size)
        #         data_tensor[4] += (np.array(record["vector.channel"]).reshape(-1,1,1)*gj).sum(axis=0).reshape(self.grid_size, self.grid_size)

        #         # for j in range(0, len(record["vector.mlat"])):
        #         #     g=gaussian(X, Y, record["vector.mlon"][j], record["vector.mlat"][j], 1, 1)
        #         #     data_tensor[0] += record["vector.vel.median"][j]*g
        #         #     data_tensor[1] += record["vector.vel.sd"][j]*g
        #         #     data_tensor[2] += record["vector.kvect"][j]*g
        #         #     data_tensor[3] += record["vector.stid"][j]*g
        #         #     data_tensor[4] += record["vector.channel"][j]*g
        #         #check its the same as if we had used the batch_gaussian 
        # print("Data Tensor Shape: ", data_tensor.shape)
        # print("Time to process data: ", time.time()-start_time)
        return data_tensor

    def process_data_fitacf(self, data1):
        #step 1: Load the date range from a north.grid file
        #data1 = pydarnio.read_north_grd(data1)
        mindate=datetime.datetime(2025, 1, 3, 1, 20, 0)
        maxdate=datetime.datetime(1998,1,1,1,20,0)
        self.check_self_location(data1)

        # Use vectorized operations to find the minimum start date and maximum end date
        start_times = np.array([datetime.datetime(record["start.year"], record["start.month"], record["start.day"], record["start.hour"], record["start.minute"]) for record in data1], dtype='datetime64')
        end_times = np.array([datetime.datetime(record["end.year"], record["end.month"], record["end.day"], record["end.hour"], record["end.minute"]) for record in data1], dtype='datetime64')

        if len(start_times) > 0:
            mindate = min(mindate, start_times.min())
        if len(end_times) > 0:
            maxdate = max(maxdate, end_times.max())

        #step 2: pick a random span of time = 2*window_size minutes from the date range
        range = maxdate - mindate

        total_time_span=self.window_size*self.time_step
        
        #furthest forward in time we can start is maxdate - total_time_span
        start_max= maxdate - datetime.timedelta(minutes=total_time_span)

        start = mindate + datetime.timedelta(minutes=np.random.randint(0, range.total_seconds()/(60*10))*10) 
        #the factor of 10 is to ensure that the start time is a multiple of 10 minutes
        #TO DO : this can be removed and replaced with some of the logic for self.time_step
        t_end= start + datetime.timedelta(minutes=self.window_size)

        t_plus_time_step_start = start + datetime.timedelta(minutes=self.time_step*self.window_size)
        t_plus_time_step_end = start + datetime.timedelta(minutes=(self.time_step+1)*self.window_size)
        #step 3: find the data entries that are within the time span
        data_at_time_t = []
        data_at_time_t_plus_step = []
        # Convert start and end times of records to numpy datetime64 arrays for vectorized operations
        record_start_times = np.array([datetime.datetime(record["start.year"], record["start.month"], record["start.day"], record["start.hour"], record["start.minute"]) for record in data1], dtype='datetime64')
        record_end_times = np.array([datetime.datetime(record["end.year"], record["end.month"], record["end.day"], record["end.hour"], record["end.minute"]) for record in data1], dtype='datetime64')

        # Create masks for the two time windows
        mask_t = (record_start_times >= np.datetime64(start)) & (record_end_times <= np.datetime64(t_end))
        mask_t_plus_step = (record_start_times >= np.datetime64(t_plus_time_step_start)) & (record_end_times <= np.datetime64(t_plus_time_step_end))

        # Use the masks to filter records
        data_at_time_t = [data1[i] for i in range(len(data1)) if mask_t[i]]
        data_at_time_t_plus_step = [data1[i] for i in range(len(data1)) if mask_t_plus_step[i]]

        ##
        #TO DO: add retrieval for the corresponding time steps from GridGVec for pot_drop and latmin and latmax
        ##
        if len(data_at_time_t) == 0 or len(data_at_time_t_plus_step) == 0:
            # print("No data found for the given time range")
            return None
        return self.process_to_tensor(data_at_time_t), self.process_to_tensor(data_at_time_t_plus_step)
    def process_data_conv(self, data1):

        #step 1: Load the date range from a north.grid file
        #data1 = pydarnio.read_north_grd(data1)

        #start timer
        start_time = time.time()
        mindate=datetime.datetime(2025, 1, 3, 1, 20, 0)
        maxdate=datetime.datetime(1998,1,1,1,20,0)

        self.check_self_location(data1)

        # Use vectorized operations to find the minimum start date and maximum end date
        start_times = np.array([datetime.datetime(record["start.year"], record["start.month"], record["start.day"], record["start.hour"], record["start.minute"]) for record in data1], dtype='datetime64')
        end_times = np.array([datetime.datetime(record["end.year"], record["end.month"], record["end.day"], record["end.hour"], record["end.minute"]) for record in data1], dtype='datetime64')

        if len(start_times) > 0:
            mindate = min(mindate, start_times.min())
        if len(end_times) > 0:
            maxdate = max(maxdate, end_times.max())
        mindate = mindate.astype(datetime.datetime)
        maxdate = maxdate.astype(datetime.datetime)
      
        trange = maxdate - mindate
        range_seconds = trange.total_seconds()
        start = mindate + datetime.timedelta(minutes=np.random.randint(0, int(range_seconds // (60 * 10))) * 10)
        #the factor of 10 is to ensure that the start time is a multiple of 10 minutes
        #TO DO : this can be removed and replaced with some of the logic for self.time_step
        t_end= start + datetime.timedelta(minutes=self.window_size)

        t_plus_time_step_start = start + datetime.timedelta(minutes=self.time_step*self.window_size)
        t_plus_time_step_end = start + datetime.timedelta(minutes=(self.time_step+1)*self.window_size)
        data_at_time_t = []
        data_at_time_t_plus_step = []
        # Convert start and end times of records to numpy datetime64 arrays for vectorized operations
        record_start_times = np.array([datetime.datetime(record["start.year"], record["start.month"], record["start.day"], record["start.hour"], record["start.minute"]) for record in data1], dtype='datetime64')
        record_end_times = np.array([datetime.datetime(record["end.year"], record["end.month"], record["end.day"], record["end.hour"], record["end.minute"]) for record in data1], dtype='datetime64')

        # Create masks for the two time windows
        mask_t = (record_start_times >= np.datetime64(start)) & (record_end_times <= np.datetime64(t_end))
        mask_t_plus_step = (record_start_times >= np.datetime64(t_plus_time_step_start)) & (record_end_times <= np.datetime64(t_plus_time_step_end))

        # Use the masks to filter records
        data_at_time_t = [data1[i] for i in range(len(data1)) if mask_t[i]]
        data_at_time_t_plus_step = [data1[i] for i in range(len(data1)) if mask_t_plus_step[i]]
        if len(data_at_time_t) == 0 or len(data_at_time_t_plus_step) == 0:
            # print("No data found for the given time range")
            return None

        # end timer
        # print("Time taken to process entries: ", time.time()-start_time)
        return self.process_conv_to_tensor(data_at_time_t), self.process_conv_to_tensor(data_at_time_t_plus_step)


    def __len__(self):
        #Return the number of batches
        return len(list(self.generator()))

    def process_file(self, data1):
        file_stream = data1.read()
        reader = pydarnio.SDarnRead(file_stream,True)
        #do check for filetype  if its convmap then
        # if reader.filetype == "convmap":
        #     return self.process_data(reader.read_map())
        # elif reader.filetype == "fitacf":
        #     return self.process_data_fitacf(reader.read_fit())
        output= self.process_data_conv(reader.read_map())
        if output is None:
            # print("No data found for the given time range")
            return None
        if output[0] is None or output[1] is None:
            # print("No data found for the given time range")
            return None
        else:
            return output
    #The data loading step will find sequential timestamps in the data, and load them into memory
    def __iter__(self):
        #returns an iterator over the objects in the bucket 
        while True:
            try:
                obj = next(self.generator())
                try:
                    minioClient= Minio(self.miniodict.get("host", "localhost") + ":" + str(self.miniodict.get("port", 9000)),
                        access_key=self.miniodict.get("access_key", "minioadmin"),
                        secret_key=self.miniodict.get("secret_key", "minioadmin"),
                        secure=False) 
                    data = minioClient.get_object(self.minioBucket, obj.object_name)

                    data1=self.process_file(data)
                    #process the data
                    if data1 is None:
                        # print("No data found for the given time range")
                        continue
                    yield data1
                except Exception as e:
                    print("Error: ", e)
                    self.minioClient.remove_object(self.minioBucket, obj.object_name)
            except Exception as e:
                
                pass

    def __getitem__(self, index):
        #return the data at the index
        return next(self.__iter__())

class DatasetFromMinioBucket(LightningDataModule):
    def __init__(self, minioClient, bucket_name, data_dir, batch_size, method='flat', windowMinutes=10, **kwargs):
        super().__init__()
        self.minioClient = minioClient
        self.bucket_name = bucket_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.method = method
        self.window_size = windowMinutes
        self.preProcess = kwargs.get("preProcess", False)
        print("Preprocess: ", self.preProcess)
        self.cache_to_disk = kwargs.get("cache_first", False)
        self.HPC = kwargs.get("HPC", False)
        if self.cache_to_disk=="True":
            self.cache_to_disk = True
        if self.HPC=="True":
            self.HPC = True
        self.kwargs = kwargs
        print("using bucket: ", self.bucket_name)
        unique_dset_config={
            "minioClient": minioClient,
            "bucket_name": bucket_name,
            "data_dir": data_dir,
            "time_step": kwargs.get("time_step", 1),
            "grid_size": kwargs.get("grid_size", 300),
            "method": method,
            "window_size": windowMinutes,
        }
        # Exclude non-hashable objects like minioClient from unique_dset_config
        hashable_config = str(unique_dset_config).encode('utf-8')
        # Create a hash of the configuration
        self.dataset_hash = hashlib.md5(hashable_config).hexdigest()[:8]
        print("Dataset Hash: ", self.dataset_hash)
    def prepare_data(self):
        # Download data from Minio bucket

        print("in prepare_data", self.cache_to_disk, self.HPC)

        if self.cache_to_disk==True and self.HPC==False: 
            #if we are on HPC we dont want to pull from MINIO
            #download the data from the minio bucket to the path specified in data_dir
            print("Downloading data from Minio bucket to disk", self.data_dir)
            os.makedirs(self.data_dir, exist_ok=True)
            file_count=len([files for root, dirs, files in os.walk(self.data_dir)])
            print("File Count: ")
            MC= Minio(self.minioClient.get("host", "localhost") + ":" 
                                                + str(self.minioClient.get("port", 9000)),
                        access_key=self.minioClient.get("access_key", "minioadmin"),
                        secret_key=self.minioClient.get("secret_key", "minioadmin"),
                        secure=False)
            print("Minio Client: ", MC)
            print(len(list(MC.list_objects(self.bucket_name, recursive=True))))
            # if  file_count <= 0.9 * len(list(MC.list_objects(self.bucket_name, recursive=True))) :
            # This is a down and dirty way to check the folders roughly the right size without interrogating fs structure
            from MinioToDisk import download_minio_bucket_to_folder
            download_minio_bucket_to_folder(self.minioClient, self.bucket_name, self.data_dir)
            #check if list the directory is the same length as the minio buckets file list 
            #if not, download the missing files using the method in MinioToDisk

        if self.preProcess:
            #preprocess the data
            print("Preprocessing data")
            self.cache_to_disk = True
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

            if not os.path.exists(os.path.join(self.data_dir, "data",str(self.dataset_hash))):
                os.makedirs(os.path.join(self.data_dir, "data",str(self.dataset_hash)))            
                dataset = SuperDARNDatasetFolder(self.data_dir, self.batch_size, self.method, self.window_size, **self.kwargs)
                cpu_count = os.cpu_count() if os.getenv("HPC", "False") == "False" else min(os.cpu_count(), 8)# cap cpus at 8 on HPC to avoid OOM errors
                Data=DataLoader(dataset, batch_size=16, shuffle=False, num_workers=cpu_count, pin_memory=False)
                save_dataset_to_disk(Data, os.path.join(self.data_dir, "data",str(self.dataset_hash)))


    def setup(self, stage=None):
        if not self.cache_to_disk:
            self.dataset=SuperDARNDataset(self.minioClient, self.bucket_name, self.batch_size, self.method, self.window_size, **self.kwargs)
        else: 
            #load the data from the disk
            if self.preProcess:
                print("Loading prepared data from disk")
                self.dataset = DatasetFromPresaved(*load_dataset_from_disk( os.path.join(self.data_dir, "data",str(self.dataset_hash))))
            else:
                print("Loading data from disk")
                self.dataset = SuperDARNDatasetFolder(self.data_dir, self.batch_size, self.method, self.window_size, **self.kwargs)
            
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [int(len(self.dataset)*0.8), len(self.dataset)-int(len(self.dataset)*0.8)])

    def train_dataloader(self):
        #we CAN shuffle the dataset here, because each item includes timestep t and timestep t+1
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=12, pin_memory=True,prefetch_factor=3)
    def validation_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    


def save_dataset_to_disk(DataLoader, path,):
    #save the dataset to disk
    #create a folder with the name of the dataset
    if not os.path.exists(path):
        os.makedirs(path)
    #save the data to disk with size info in minibatches of 200
    tensorA=torch.tensor([])
    tensorB=torch.tensor([])
    tensorshape=None
    for idx,i in enumerate(tqdm(DataLoader)):
        dataA,dataB = i
        #add to the tensor
        if dataA is None or dataB is None:
            continue
        if tensorA.shape[0] == 0:
            tensorA = dataA
            tensorB = dataB
        else:
            tensorA = torch.cat((tensorA, dataA), dim=0)
            tensorB = torch.cat((tensorB, dataB), dim=0)
        if tensorA.shape[0] >= 200:
            #save the data to disk
            # tensorA[:200].numpy().tofile(os.path.join(path, "dataA_{}.bin".format(idx)))
            np.save(os.path.join(path, "dataA_{}.npy".format(idx)), tensorA[:200].numpy())
            # tensorB[:200].numpy().tofile(os.path.join(path, "dataB_{}.bin".format(idx)))
            np.save(os.path.join(path, "dataB_{}.npy".format(idx)), tensorB[:200].numpy())
            tensorA = tensorA[200:]
            tensorB = tensorB[200:]
            tensorshape=tensorA.shape

    #save the remaining data to disk
    if tensorA.shape[0] > 0:
        # tensorA.numpy().tofile(os.path.join(path, "dataA_{}.bin".format(idx)))
        np.save(os.path.join(path, "dataA_{}.npy".format(idx)), tensorA.numpy())
        # tensorB.numpy().tofile(os.path.join(path, "dataB_{}.bin".format(idx)))
        np.save(os.path.join(path, "dataB_{}.npy".format(idx)), tensorB.numpy())
        tensorshape=list([tensorA.shape[i] for i in range(len(tensorA.shape))]) #because we are going to modify it
    #save the shape of the data
    tensorshape[0] = -1
    with open(os.path.join(path, "shape.txt"), 'w') as f:
        f.write(str(tensorshape))


def load_dataset_from_disk(path):
    #load the dataset from disk
    dataA = []
    dataB = []
    with open(os.path.join(path, "shape.txt"), 'r') as f:
        shape = eval(f.read())
        shape[0] = -1

    for root, dirs, files in tqdm(os.walk(path)):
        for file in files:
            if "dataA" in file:
                dataA.append(os.path.join(root, file))
            elif "dataB" in file:
                dataB.append(os.path.join(root, file))
    #stack the data
    return dataA, dataB,shape


class DatasetFromPresaved(Dataset):
    def __init__(self, dataA, dataB,shape):
        #dataA and dataB are file lists 
        self.dataA = dataA
        self.dataB = dataB #ea
        self.shape = shape[-3:]
        self.len= len(dataA)*200
        #Each file is 344MB, so lets avoid loading the whole thing into memory


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        #reconstruct to the original shape
        file_index= index//200
        file_offset= index%200
        #to calculate the offset, each file holds a shape size is 200,5,G,G
        #we want to find the offset in the file to find the correct offset for the file
        dA=np.load(self.dataA[file_index], mmap_mode='r')
        #load the data from the file
        dB=np.load(self.dataB[file_index], mmap_mode='r')
        #get the corresponding data
        if dA.shape[0] < file_offset or dB.shape[0] < file_offset:
            self.len=self.len-200+dA.shape[0]
            return self.__getitem__(random.randint(0, self.len-1))
        dataA,dataB= dA[file_offset,:,:,:], dB[file_offset,:,:,:]
        #dataA and dataB are of shape 5,G,G
        #reshape the data to the original shape
        dataA = dataA.reshape(self.shape)
        dataB = dataB.reshape(self.shape)
        #convert to tensor
        dataA = torch.tensor(dataA, dtype=torch.float32)
        dataB = torch.tensor(dataB, dtype=torch.float32)
        return dataA, dataB

if __name__ == "__main__":
    #test the dataloader
    from minio import Minio
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--MINIOHost", type=str, default="10.45.15.149")
    parser.add_argument("--MINIOPort", type=int, default=9000)
    parser.add_argument("--MINIOAccesskey", type=str, default="minioadmin")
    parser.add_argument("--MINIOSecret", type=str, default="minioadmin")
    parser.add_argument("--bucket_name", type=str, default="convmap")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--method", type=str, default="grid")
    parser.add_argument("--WindowsMinutes", type=int, default=120)
    args = parser.parse_args()

    minioClient = {"host": args.MINIOHost, "port": args.MINIOPort, "access_key": args.MINIOAccesskey, "secret_key": args.MINIOSecret} 
    
    dataModule = DatasetFromMinioBucket(minioClient, args.bucket_name,args.data_dir, args.batch_size, args.method, args.WindowsMinutes)
    dataModule.prepare_data()
    dataModule.setup()
    for idx,batch in enumerate(dataModule.train_dataloader()):
        #use matplotlib to plot the data
        #create 2 subplots
        if idx % 10 == 0:
            fig, axs = plt.subplots(2)
            #plot the 2 arrays of size 20 x 88008, but resize them to 20 x 300 for visualization
            axs[0].imshow(batch[0].flatten(0,-2).numpy()[:, :300])
            axs[1].imshow(batch[1].flatten(0,-2).numpy()[:, :300])

            plt.savefig("test{}.png".format(idx))
            plt.close()
    print("Test Passed")
    print(dataModule.dataset.location)
