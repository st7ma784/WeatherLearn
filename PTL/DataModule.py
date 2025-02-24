from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch.utils.data import IterableDataset
import pydarnio
import numpy as np

class SuperDARNDataset(IterableDataset):
    def __init__(self,minioClient, minioBucket, batch_size):
        super().__init__()
        self.minioBucket = minioBucket
        self.batch_size = batch_size
        self.minioClient = minioClient
    

    def process_data(self, data1, data2):
        #use pydarnio to read the datafiles and plot as arrays.
        #return the arrays
                
        return data1, data2



    def __len__(self):
        #Return the number of batches
        return len(self.minioClient.list_objects(self.minioBucket)) // self.batch_size


    #The data loading step will find sequential timestamps in the data, and load them into memory
    def __iter__(self):
        #Get the list of objects in the bucket
        objects = self.minioClient.list_objects(self.minioBucket)
        #Iterate over the objects
        #sort the objects by name and take 2 adjacent objects at a time
        #yield the 2 objects as a tuple
        for i in range(0, len(objects), 2):
            #Get the object names
            obj1 = objects[i].object_name
            obj2 = objects[i+1].object_name
            #Get the data from the objects
            data1 = self.minioClient.get_object(self.minioBucket, obj1)
            data2 = self.minioClient.get_object(self.minioBucket, obj2)
            #yield the data
            data1,data2= self.process_data(data1, data2)

            yield (data1, data2)    


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