from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch.utils.data import IterableDataset
import pydarnio
import numpy as np
import datetime
from minio import Minio
import os


def gaussian(x, y, x0, y0, sigma_x, sigma_y):
            return np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)))


class SuperDARNDataset(IterableDataset):

    @classmethod
    def from_disk(cls, data_dir, batch_size, method='flat', window_size=10, **kwargs):
        #load the data from the disk
        self = cls.__new__(cls, {},"", batch_size, method, window_size, silent=True, **kwargs)
        
        def generator():
            import os
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    yield open(os.path.join(root, file), "rb")
        self.generator = generator
        def __iter__():
            for file in self.generator():
                yield self.process_file(file)
        
        self.__iter__ = __iter__
        return self
    
    def __new__(cls, miniodict, minioBucket, batch_size, method='flat', window_size=10, **kwargs):
        self = super().__new__(cls)
        self.__init__(miniodict, minioBucket, batch_size, method, window_size, **kwargs)
        return self


    def __init__(self,miniodict, minioBucket, batch_size, method='flat', window_size=10, **kwargs):
        super().__init__()
        self.minioBucket = minioBucket
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
                print(bucket.name, bucket.creation_date)
        except Exception as e:
            
            if not kwargs.get("silent", False):
                raise e
        print("Connected to Minio")
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
    def generator(self):   
        #return an iterator over the objects in the bucket
        for obj in self.minioClient.list_objects(self.minioBucket):
            yield obj


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
        data_tensor = torch.zeros(5, self.grid_size, self.grid_size)
        #we're going to sample the data onto a 300x300 grid
        #we're going to use a gaussian kernel to splat the data onto the grid
        #step1: create a meshgrid
        x = np.linspace(self.location["min_mlon"], self.location["max_mlon"], self.grid_size)
        y = np.linspace(self.location["min_mlat"], self.location["max_mlat"], self.grid_size)
        X, Y = np.meshgrid(x, y)
        #step 2 : create a gaussian kernel

        #step 3: splat the data onto the grid
        for record in data:
            if "vector.mlat" not in record:
                # print("vector.mlat not found in record")
                continue
            else:
                for j in range(0, len(record["vector.mlat"])):
                    data_tensor[0] += record["vector.vel.median"][j]*gaussian(X, Y, record["vector.mlon"][j], record["vector.mlat"][j], 1, 1)
                    data_tensor[1] += record["vector.vel.sd"][j]*gaussian(X, Y, record["vector.mlon"][j], record["vector.mlat"][j], 1, 1)
                    data_tensor[2] += record["vector.kvect"][j]*gaussian(X, Y, record["vector.mlon"][j], record["vector.mlat"][j], 1, 1)
                    data_tensor[3] += record["vector.stid"][j]*gaussian(X, Y, record["vector.mlon"][j], record["vector.mlat"][j], 1, 1)
                    data_tensor[4] += record["vector.channel"][j]*gaussian(X, Y, record["vector.mlon"][j], record["vector.mlat"][j], 1, 1)
        return data_tensor

    def process_data_fitacf(self, data1):
                #step 1: Load the date range from a north.grid file
        #data1 = pydarnio.read_north_grd(data1)
        mindate=datetime.datetime(2025, 1, 3, 1, 20, 0)
        maxdate=datetime.datetime(1998,1,1,1,20,0)

        for record in data1:
          
            start = datetime.datetime(record["start.year"], record["start.month"], record["start.day"], record["start.hour"], record["start.minute"])
            # print("Start Time: ", start)
            end = datetime.datetime(record["end.year"], record["end.month"], record["end.day"], record["end.hour"], record["end.minute"])
            # print("End Time: ", end)
            if start < mindate:
                mindate = start
                #rint("Min Date now: ", mindate)
            if end > maxdate:
                maxdate = end

            if "vector.index" not in record:
                # print("Index not found in record")
                continue
            # else :
            #     #rint("Record found: ", record)
            if np.max(record["vector.index"]) > self.location["max_vector"]:
                self.location.update({"max_vector":np.max(record["vector.index"])})
                #rint("Max Vector Size now: ",self.location["max_vector"])
            if  np.max(record["vector.mlat"]) > self.location["max_mlat"]:
                self.location.update({"max_mlat":np.max(record["vector.mlat"])})
                #rint("Max Latitude now: ",self.location["max_mlat"])
            if  np.max(record["vector.mlon"]) > self.location["max_mlon"]:
                self.location.update({"max_mlon":np.max(record["vector.mlon"])})
                #rint("Max Longitude now: ",self.location["max_mlon"])
            if  np.min(record["vector.mlat"]) < self.location["min_mlat"]:
                self.location.update({"min_mlat":np.min(record["vector.mlat"])})
                #rint("Min Latitude now: ",self.location["min_mlat"])
            if np.min(record["vector.mlon"]) < self.location["min_mlon"]:
                self.location.update({"min_mlon":np.min(record["vector.mlon"])})
                #rint("Min Longitude now: ",self.location["min_mlon"])

        #step 2: pick a random span of time = 2*window_size minutes from the date range
        range = maxdate - mindate
        start = mindate + datetime.timedelta(minutes=np.random.randint(0, range.total_seconds()/(60*10))*10) 
        #the factor of 10 is to ensure that the start time is a multiple of 10 minutes
        #TO DO : this can be removed and replaced with some of the logic for self.time_step
        end = start + datetime.timedelta(minutes=2*self.window_size)
        mid_point= start + datetime.timedelta(minutes=self.window_size)
        #step 3: find the data entries that are within the time span
        data_at_time_t = []
        data_at_time_t_plus_1 = []
        for record in data1:
            d_start = datetime.datetime(record["start.year"], record["start.month"], record["start.day"], record["start.hour"], record["start.minute"])
            d_end = datetime.datetime(record["end.year"], record["end.month"], record["end.day"], record["end.hour"], record["end.minute"])
            if d_start >= start and d_end <= mid_point:
                data_at_time_t.append(record)
            if d_start >= mid_point and d_end <= end:
                data_at_time_t_plus_1.append(record)

        ##
        #TO DO: add retrieval for the corresponding time steps from GridGVec for pot_drop and latmin and latmax
        ##

        return self.process_to_tensor(data_at_time_t), self.process_to_tensor(data_at_time_t_plus_1)
    def process_data_conv(self, data1):

        #step 1: Load the date range from a north.grid file
        #data1 = pydarnio.read_north_grd(data1)
        mindate=datetime.datetime(2025, 1, 3, 1, 20, 0)
        maxdate=datetime.datetime(1998,1,1,1,20,0)

        for record in data1:
          
            start = datetime.datetime(record["start.year"], record["start.month"], record["start.day"], record["start.hour"], record["start.minute"])
            # print("Start Time: ", start)
            end = datetime.datetime(record["end.year"], record["end.month"], record["end.day"], record["end.hour"], record["end.minute"])
            # print("End Time: ", end)
            if start < mindate:
                mindate = start
                #rint("Min Date now: ", mindate)
            if end > maxdate:
                maxdate = end
    
            if "vector.index" not in record:
                # print("Index not found in record")
                continue
            # else :
            #     #rint("Record found: ", record)
            if np.max(record["vector.index"]) > self.location["max_vector"]:
                self.location.update({"max_vector":np.max(record["vector.index"])})
                #rint("Max Vector Size now: ",self.location["max_vector"])
            if  np.max(record["vector.mlat"]) > self.location["max_mlat"]:
                self.location.update({"max_mlat":np.max(record["vector.mlat"])})
                #rint("Max Latitude now: ",self.location["max_mlat"])
            if  np.max(record["vector.mlon"]) > self.location["max_mlon"]:
                self.location.update({"max_mlon":np.max(record["vector.mlon"])})
                #rint("Max Longitude now: ",self.location["max_mlon"])
            if  np.min(record["vector.mlat"]) < self.location["min_mlat"]:
                self.location.update({"min_mlat":np.min(record["vector.mlat"])})
                #rint("Min Latitude now: ",self.location["min_mlat"])
            if np.min(record["vector.mlon"]) < self.location["min_mlon"]:
                self.location.update({"min_mlon":np.min(record["vector.mlon"])})
                #rint("Min Longitude now: ",self.location["min_mlon"])

        #step 2: pick a random span of time = 2*window_size minutes from the date range
        range = maxdate - mindate
        start = mindate + datetime.timedelta(minutes=np.random.randint(0, range.total_seconds()/(60*10))*10) #the factor of 10 is to ensure that the start time is a multiple of 10 minutes
        end = start + datetime.timedelta(minutes=2*self.window_size)
        mid_point= start + datetime.timedelta(minutes=self.window_size)
        #step 3: find the data entries that are within the time span
        data_at_time_t = []
        data_at_time_t_plus_1 = []
        for record in data1:
            d_start = datetime.datetime(record["start.year"], record["start.month"], record["start.day"], record["start.hour"], record["start.minute"])
            d_end = datetime.datetime(record["end.year"], record["end.month"], record["end.day"], record["end.hour"], record["end.minute"])
            if d_start >= start and d_end <= mid_point:
                data_at_time_t.append(record)
            if d_start >= mid_point and d_end <= end:
                data_at_time_t_plus_1.append(record)

        ##
        #TO DO: add retrieval for the corresponding time steps from GridGVec for pot_drop and latmin and latmax
        ##

        return self.process_conv_to_tensor(data_at_time_t), self.process_conv_to_tensor(data_at_time_t_plus_1)


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
        return self.process_data_conv(reader.read_map())

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
                    yield self.process_data(data1)
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
        self.cache_to_disk = kwargs.get("cache_first", False)
        self.HPC = kwargs.get("HPC", False)
        self.kwargs = kwargs
    def prepare_data(self):
        # Download data from Minio bucket
        if self.cache_to_disk==True and self.HPC==False: 
            #if we are on HPC we dont want to pull from MINIO
            #download the data from the minio bucket to the path specified in data_dir
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            file_count=len([files for root, dirs, files in os.walk(self.data_dir)])
            MC= Minio(self.minioClient.get("host", "localhost") + ":" 
                                                + str(self.minioClient.get("port", 9000)),
                        access_key=self.minioClient.get("access_key", "minioadmin"),
                        secret_key=self.minioClient.get("secret_key", "minioadmin"),
                        secure=False)
            if len(list(MC.list_objects(self.bucket_name, recursive=True))) <= 0.9 * file_count:
                # This is a down and dirty way to check the folders roughly the right size without interrogating fs structure
                from MinioToDisk import download_minio_bucket_to_folder
                download_minio_bucket_to_folder(self.minioClient, self.bucket_name, self.data_dir)
            #check if list the directory is the same length as the minio buckets file list 
            #if not, download the missing files using the method in MinioToDisk
        pass
    
    def setup(self, stage=None):
        if not self.cache_to_disk:
            self.dataset=SuperDARNDataset(self.minioClient, self.bucket_name, self.batch_size, self.method, self.window_size, **self.kwargs)
        else: 
            self.dataset = SuperDARNDataset.from_disk(self.data_dir, self.batch_size, self.method, self.window_size, **self.kwargs)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [int(len(self.dataset)*0.8), len(self.dataset)-int(len(self.dataset)*0.8)])

    def train_dataloader(self):
        #we CAN shuffle the dataset here, because each item includes timestep t and timestep t+1
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True,prefetch_factor=3)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    


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

    minioClient = {"host": args.MINIOHost, "port": args.MINIOPort, "access_key": args.MINIOAccesskey
                    , "secret_key": args.MINIOSecret} 
    
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
