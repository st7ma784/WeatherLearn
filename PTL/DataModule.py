from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch.utils.data import IterableDataset
import pydarnio
import numpy as np
import datetime



def gaussian(x, y, x0, y0, sigma_x, sigma_y):
            return np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)))


class SuperDARNDataset(IterableDataset):
    def __init__(self,miniodict, minioBucket, batch_size, method='flat', window_size=10):
        super().__init__()
        self.minioBucket = minioBucket
        self.batch_size = batch_size

        self.minioClient = Minio(miniodict.get("host", "localhost") + ":" 
                                                + str(miniodict.get("port", 9000)),
                        access_key=miniodict.get("access_key", "minioadmin"),
                        secret_key=miniodict.get("secret_key", "minioadmin"),
                        secure=False)
        print("Connected to Minio")
        #list the buckets   

        print("Buckets List:")
        buckets = self.minioClient.list_buckets()
        for bucket in buckets:
            print(bucket.name, bucket.creation_date)
        self.generator= self.minioClient.list_objects(self.minioBucket)
        self.window_size = window_size
        self.method = method
        if self.method == 'flat':
            self.process_to_tensor = self.process_to_flat
        elif self.method == 'grid':
            self.process_to_tensor = self.process_to_grid
        else:
            raise ValueError("method must be either 'flat' or 'grid'")
        self.location={"max_vector":88008, "max_mlat":0,
                      "max_mlon":0, 
                      "min_mlat":360,
                      "min_mlon":360}
    def process_to_flat(self, data):
        #use the index location as index_select into tensor of size (num_points, x)
        data_tensor = torch.zeros(5, self.location["max_vector"]+1)
        for record in data:
            if "vector.index" not in record:
                print("Index not found in record")
                continue
            if np.max(record["vector.index"]) > self.location["max_vector"]:
                print("Index out of bounds: ", record["vector.index"])
            
            data_tensor[0, record["vector.index"]] = torch.tensor(record["vector.vel.median"])
            data_tensor[1, record["vector.index"]] = torch.tensor(record["vector.vel.sd"])
            data_tensor[2, record["vector.index"]] = torch.tensor(record["vector.kvect"])
            data_tensor[3, record["vector.index"]] = torch.tensor(record["vector.stid"]).to(torch.float)
            data_tensor[4, record["vector.index"]] = torch.tensor(record["vector.channel"]).to(torch.float)

        return data_tensor


        
    def process_to_grid(self, data):
        #use the mlat, mlon to gaussian splat onto a x,y grid and stack
        data_tensor = torch.zeros(5, 300, 300)
        #we're going to sample the data onto a 300x300 grid
        #we're going to use a gaussian kernel to splat the data onto the grid
        #step1: create a meshgrid
        x = np.linspace(self.location["min_mlon"], self.location["max_mlon"], 300)
        y = np.linspace(self.location["min_mlat"], self.location["max_mlat"], 300)
        X, Y = np.meshgrid(x, y)
        #step 2 : create a gaussian kernel
        
        #step 3: splat the data onto the grid
        for record in data:
            for j in range(0, len(record["vector.mlat"])):
                data_tensor[0] += record["vector.vel.median"][j]*gaussian(X, Y, record["vector.mlon"][j], record["vector.mlat"][j], 1, 1)
                data_tensor[1] += record["vector.vel.sd"][j]*gaussian(X, Y, record["vector.mlon"][j], record["vector.mlat"][j], 1, 1)
                data_tensor[2] += record["vector.kvect"][j]*gaussian(X, Y, record["vector.mlon"][j], record["vector.mlat"][j], 1, 1)
                data_tensor[3] += record["vector.stid"][j]*gaussian(X, Y, record["vector.mlon"][j], record["vector.mlat"][j], 1, 1)
                data_tensor[4] += record["vector.channel"][j]*gaussian(X, Y, record["vector.mlon"][j], record["vector.mlat"][j], 1, 1)
        return data_tensor


    def process_data(self, data1):

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
                print("Min Date now: ", mindate)
            if end > maxdate:
                maxdate = end
                print("Max Date now: ", maxdate)
            # print("boundary lat : ", type(record["boundary.mlat"]))
            # print("boundary lon:", type(record["boundary.mlon"]))
            # print("model lat: ",type(record["model.mlat"]))
            # print("model lon :" ,type(record["model.mlon"]))
            # print("vector mlat ",type(record["vector.mlat"]))
            # print("vector mlon",type(record["vector.mlon"]))
            if "vector.index" not in record:
                print("Index not found in record", record)
                continue
            else :
                print("Record found: ", record)
            if np.max(record["vector.index"]) > self.location["max_vector"]:
                self.location.update({"max_vector":np.max(record["vector.index"])})
                print("Max Vector Size now: ",self.location["max_vector"])
            if  np.max(record["vector.mlat"]) > self.location["max_mlat"]:
                self.location.update({"max_mlat":np.max(record["vector.mlat"])})
                print("Max Latitude now: ",self.location["max_mlat"])
            if  np.max(record["vector.mlon"]) > self.location["max_mlon"]:
                self.location.update({"max_mlon":np.max(record["vector.mlon"])})
                print("Max Longitude now: ",self.location["max_mlon"])
            if  np.min(record["vector.mlat"]) < self.location["min_mlat"]:
                self.location.update({"min_mlat":np.min(record["vector.mlat"])})
                print("Min Latitude now: ",self.location["min_mlat"])
            if np.min(record["vector.mlon"]) < self.location["min_mlon"]:
                self.location.update({"min_mlon":np.min(record["vector.mlon"])})
                print("Min Longitude now: ",self.location["min_mlon"])

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

        return self.process_to_tensor(data_at_time_t), self.process_to_tensor(data_at_time_t_plus_1)


    def __len__(self):
        #Return the number of batches
        return len(list(self.minioClient.list_objects(self.minioBucket))) // self.batch_size


    #The data loading step will find sequential timestamps in the data, and load them into memory
    def __iter__(self):
        #returns an iterator over the objects in the bucket 
        while True:
        
            obj = next(self.generator)
            #load the data from the object
            #get file from minio into a byte stream
            data = self.minioClient.get_object(self.minioBucket, obj.object_name)
            #convert the byte stream into a list of pydarnio objects
            file_stream = data.read()
            reader = pydarnio.SDarnRead(file_stream,True)
            data1=reader.read_map()
            #process the data
            yield self.process_data(data1)
            

    def __getitem__(self, index):
        #return the data at the index
        return next(self.__iter__())

class DatasetFromMinioBucket(LightningDataModule):
    def __init__(self, minioClient, bucket_name, data_dir, batch_size, method='flat', window_size=10):
        super().__init__()
        self.minioClient = minioClient
        self.bucket_name = bucket_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.method = method
        self.window_size = window_size

    def prepare_data(self):
        # Download data from Minio bucket
        pass
    
    def setup(self, stage=None):
        self.dataset=SuperDARNDataset(self.minioClient, self.bucket_name, self.batch_size, self.method, self.window_size)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [int(len(self.dataset)*0.8), int(len(self.dataset)*0.2)])

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
    parser.add_argument("--MINIOHost", type=str, default="10.45.15.149")
    parser.add_argument("--MINIOPort", type=int, default=9000)
    parser.add_argument("--MINIOAccesskey", type=str, default="minioadmin")
    parser.add_argument("--MINIOSecret", type=str, default="minioadmin")
    parser.add_argument("--bucket_name", type=str, default="convmap")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--method", type=str, default="flat")
    parser.add_argument("--window_size", type=int, default=120)
    args = parser.parse_args()

    minioClient = {"host": args.MINIOHost, "port": args.MINIOPort, "access_key": args.MINIOAccesskey
                    , "secret_key": args.MINIOSecret} 
    
    dataModule = DatasetFromMinioBucket(minioClient, args.bucket_name,args.data_dir, args.batch_size, args.method, args.window_size)
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
