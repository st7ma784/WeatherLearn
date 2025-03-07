from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch.utils.data import IterableDataset
import pydarnio
import numpy as np
import datetime

class SuperDARNDataset(IterableDataset):
    def __init__(self,minioClient, minioBucket, batch_size, method='flat', window_size=10):
        super().__init__()
        self.minioBucket = minioBucket
        self.batch_size = batch_size
        self.minioClient = minioClient
        self.window_size = window_size
        self.method = method
        if self.method == 'flat':
            self.process_to_tensor = self.process_to_flat
        elif self.method == 'grid':
            self.process_to_tensor = self.process_to_grid
        else:
            raise ValueError("method must be either 'flat' or 'grid'")
        self.data={"max_vector":0,
                    "max_mlat":0,
                      "max_mlon":0, 
                      "min_mlat":90,
                      "min_mlon":180}
    def process_to_flat(self, data):
        #use the index location as index_select into tensor of size (num_points, x)
        data_tensor = torch.zeros(5, self.data["max_vector"]+1)

        data_tensor[0, data[0].vector.index] = data[0].vector.vel.median
        data_tensor[1, data[0].vector.index] = data[0].vector.vel.sd
        data_tensor[2, data[0].vector.index] = data[0].vector.kvect
        data_tensor[3, data[0].vector.index] = data[0].vector.stid
        data_tensor[4, data[0].vector.index] = data[0].vector.channel
        return data_tensor


        
    def process_to_grid(self, data):
        #use the mlat, mlon to gaussian splat onto a x,y grid and stack
        data_tensor = torch.zeros(5, 300, 300)
        #we're going to sample the data onto a 300x300 grid
        #we're going to use a gaussian kernel to splat the data onto the grid
        #step1: create a meshgrid
        x = np.linspace(self.data["min_mlon"], self.data["max_mlon"], 300)
        y = np.linspace(self.data["min_mlat"], self.data["max_mlat"], 300)
        X, Y = np.meshgrid(x, y)
        #step 2 : create a gaussian kernel
        def gaussian(x, y, x0, y0, sigma_x, sigma_y):
            return np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)))
        #step 3: splat the data onto the grid
        for i in range(0, len(data)):
            for j in range(0, len(data[i].vector.mlat)):
                data_tensor[0] += data[i].vector.vel.median[j]*gaussian(X, Y, data[i].vector.mlon[j], data[i].vector.mlat[j], 1, 1)
                data_tensor[1] += data[i].vector.vel.sd[j]*gaussian(X, Y, data[i].vector.mlon[j], data[i].vector.mlat[j], 1, 1)
                data_tensor[2] += data[i].vector.kvect[j]*gaussian(X, Y, data[i].vector.mlon[j], data[i].vector.mlat[j], 1, 1)
                data_tensor[3] += data[i].vector.stid[j]*gaussian(X, Y, data[i].vector.mlon[j], data[i].vector.mlat[j], 1, 1)
                data_tensor[4] += data[i].vector.channel[j]*gaussian(X, Y, data[i].vector.mlon[j], data[i].vector.mlat[j], 1, 1)
        return data_tensor


    def process_data(self, data1):

        #step 1: Load the date range from a north.grid file
        #data1 = pydarnio.read_north_grd(data1)
        mindate=datetime.datetime(2025, 1, 3, 1, 20, 0)
        maxdate=datetime.datetime(1998,1,1,1,20,0)
        for i in range(0, len(data1)):
            if data1[i].start < mindate:
                mindate = data1[i].start
            if data1[i].end > maxdate:
                maxdate = data1[i].end
            if max(data1[i].vector.index) > self.data["max_vector"]:
                self.data["max_vector"] = max(data1[i].vector.index)
            if max(data1[i].vector.mlat) > self.data["max_mlat"]:
                self.data["max_mlat"] = max(data1[i].vector.mlat)
            if max(data1[i].vector.mlon) > self.data["max_mlon"]:
                self.data["max_mlon"] = max(data1[i].vector.mlon)
            if min(data1[i].vector.mlat) < self.data["min_mlat"]:
                self.data["min_mlat"] = min(data1[i].vector.mlat)
            if min(data1[i].vector.mlon) < self.data["min_mlon"]:
                self.data["min_mlon"] = min(data1[i].vector.mlon)
        print(mindate)
        print(maxdate)

        #step 2: pick a random span of time = 2*window_size minutes from the date range
        range = maxdate - mindate
        start = mindate + datetime.timedelta(minutes=np.random.randint(0, range.total_seconds()/(60*10))*10) #the factor of 10 is to ensure that the start time is a multiple of 10 minutes
        end = start + datetime.timedelta(minutes=2*self.window_size)
        mid_point= start + datetime.timedelta(minutes=self.window_size)
        #step 3: find the data entries that are within the time span
        data_at_time_t = []
        data_at_time_t_plus_1 = []
        for i in range(0, len(data1)):
            if data1[i].start >= start and data1[i].end <= mid_point:
                data_at_time_t.append(data1[i])
            if data1[i].start >= mid_point and data1[i].end <= end:
                data_at_time_t_plus_1.append(data1[i])

        return self.process_to_tensor(data_at_time_t), self.process_to_tensor(data_at_time_t_plus_1)


    def __len__(self):
        #Return the number of batches
        return len(self.minioClient.list_objects(self.minioBucket)) // self.batch_size


    #The data loading step will find sequential timestamps in the data, and load them into memory
    def __iter__(self):
        #Get the list of objects in the bucket
        #Iterate over the objects
        #sort the objects by name and take 2 adjacent objects at a time
        #yield the 2 objects as a tuple
        for obj in self.minioClient.list_objects(self.minioBucket):
            #Get the object names
            #Get the data from the objects
            data1=self.minioClient.get_object(self.minioBucket, obj)
            #yield the data
            # 
            yield self.process_data(data1)



class DatasetFromMinioBucket(LightningDataModule):
    def __init__(self, minioClient, bucket_name, data_dir, batch_size):
        super().__init__()
        self.minioClient = minioClient
        self.bucket_name = bucket_name
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # Download data from Minio bucket
        pass
    
    def setup(self, stage=None):
        dataset=SuperDARNDataset(self.minioClient, self.bucket_name, self.batch_size)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])

    def train_dataloader(self):
        #we CAN shuffle the dataset here, because each item includes timestep t and timestep t+1
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
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
    parser.add_argument("--MINIOHost", type=str, default="localhost")
    parser.add_argument("--MINIOPort", type=int, default=9000)
    parser.add_argument("--MINIOAccesskey", type=str, default="minioadmin")
    parser.add_argument("--MINIOSecret", type=str, default="minioadmin")
    parser.add_argument("--bucket_name", type=str, default="pydarn")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--method", type=str, default="flat")
    parser.add_argument("--window_size", type=int, default=10)
    args = parser.parse_args()

    minioClient = Minio(args.MINIOHost,
                        access_key=args.MINIOAccesskey,
                        secret_key=args.MINIOSecret,
                        secure=False)
    dataModule = DatasetFromMinioBucket(minioClient, args.bucket_name, args.data_dir, args.batch_size, args.method, args.window_size)
    dataModule.prepare_data()
    dataModule.setup()
    for batch in dataModule.train_dataloader():
        #use matplotlib to plot the data
        #create 2 subplots
        fig, axs = plt.subplots(2)
        #plot the 2 arrays 
        axs[0].imshow(batch[0][0])
        axs[1].imshow(batch[1][0])
        plt.show()
    