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
    

    def process_data(self, data1):
        #use pydarnio to read the datafiles and plot as arrays.
        #return the arrays
        ## FITACF
        # ptab: (8,)
        # ltab: (24, 2)
        # pwr0: (100,)
        ## RAWACF
        #ptab: (7,)
        #ltab: (19, 2)
        #slist: (70,)
        #pwr0: (70,)
        #acfd: (70, 18, 2)
        #xcfd: (70, 18, 2)

        #Grid data most useful 
        #because it has the flow velocities in it . 
        # \\luna.lancs.ac.uk\FST\PY\SPP\data\superdarn\new\cnvmap_TS18_range_limit\north\1999


        # start.year: 2009
        # start.month: 1
        # start.day: 3
        # start.hour: 1
        # start.minute: 20
        # start.second: 0.0
        # end.year: 2009
        # end.month: 1
        # end.day: 3
        # end.hour: 1
        # end.minute: 22
        # end.second: 0.0
        # stid: (1,)
        # channel: (1,)
        # nvec: (1,)
        # freq: (1,)
        # major.revision: (1,)
        # minor.revision: (1,)
        # program.id: (1,)
        # noise.mean: (1,)
        # noise.sd: (1,)
        # gsct: (1,)
        # v.min: (1,)
        # v.max: (1,)
        # p.min: (1,)
        # p.max: (1,)
        # w.min: (1,)
        # w.max: (1,)
        # ve.min: (1,)
        # ve.max: (1,)
        # vector.mlat: (6,)
        # vector.mlon: (6,)
        # vector.kvect: (6,)
        # vector.stid: (6,)
        # vector.channel: (6,)
        # vector.index: (6,)
        # vector.vel.median: (6,)
        # vector.vel.sd: (6,)
        #idea one use mlat, mlon to gaussian splat onto a x,y grid and stack
        #use grid index as collumn 
        #

        # return 2 adjacent time entries from the convmap file
        return data


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
            data1 = self.minioClient.get_object(self.minioBucket, obj.object_name)
            #yield the data
            data1= self.process_data(data1

            yield data1    


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