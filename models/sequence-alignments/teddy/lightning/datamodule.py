from teddy.data.Alphabet import Alphabet
from teddy.data.dataset import MsaLabels, CollateCustomfn,MsaName, CollateCustomfnNames, DistributedVariableBatchSampler
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl

class BDS_datamodule(pl.LightningDataModule):
    def __init__(self,
                 data_dir:str,  
                 alphabet:Alphabet, 
                 train_ratio:float, 
                 val_batch_size:int,
                 batch_size:int= None,
                 conditional_batch_sizes:list = None,
                 num_workers:int=4, 
                 limit_size:int = None,
                 prefetch_factor:int = 2,
                 persistent_workers:bool = True,
                 pin_memory:bool = True,
                 cache_dir:str = None,
                 seed:int = 42,
                 drop_last:bool = False,
                 shuffle:bool = True,
                 max_seq_len:int = 100,
                 max_sites_len:int = 1500,
                 scale_dates:float = 1.0,
                 scale_dur_target:float = 1.0,
                 max_size_dataset: int = None,
                 round_dates:bool = False
                 ):
        super().__init__()

        if batch_size is None:
            if conditional_batch_sizes is None:
                raise ValueError("batch_size or conditional_batch_sizes must be provided")
        else:
            assert val_batch_size==batch_size
            if conditional_batch_sizes is not None:
                raise ValueError("batch_size and conditional_batch_sizes cannot be used together")
        

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.alphabet = alphabet
        self.train_ration = train_ratio
        self.num_workers = num_workers
        self.limit_size = limit_size
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.cache_dir = cache_dir
        self.seed = seed
        self.drop_last = drop_last
        self.conditional_batch_sizes = conditional_batch_sizes
        self.shuffle = shuffle
        self.val_batch_size = val_batch_size
        self.max_seq_len = max_seq_len
        self.max_sites_len = max_sites_len
        self.scale_dur_target = scale_dur_target
        self.scale_dates = scale_dates
        self.max_size_dataset = max_size_dataset
        self.round_dates = round_dates

    def setup(self,stage=None):
        self.dataset = MsaLabels(self.data_dir, self.alphabet, limit_size=self.limit_size, cache_dir=self.cache_dir, scale_dur_target=self.scale_dur_target, scale_dates=self.scale_dates, max_size_dataset=self.max_size_dataset, round_dates=self.round_dates)
        self.train_dataset,self.val_dataset = random_split(self.dataset, [self.train_ration, 1-self.train_ration])

        
        


    def train_dataloader(self):

        if self.batch_size is not None:
            return DataLoader(self.train_dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers,
                                collate_fn=CollateCustomfn(self.alphabet.pad_idx, self.max_seq_len, self.max_sites_len),
                                prefetch_factor=self.prefetch_factor,
                                persistent_workers=self.persistent_workers,
                                pin_memory=self.pin_memory
                                )
        

        tsampler = DistributedVariableBatchSampler(
            dataset = self.train_dataset,
            conditional_batch_sizes=self.conditional_batch_sizes,
            shuffle=self.shuffle,
            seed = self.seed,
            drop_last=self.drop_last,
        )


        return DataLoader(dataset=self.train_dataset,
                              num_workers=self.num_workers,
                              collate_fn=CollateCustomfn(self.alphabet.pad_idx, self.max_seq_len, self.max_sites_len),
                                batch_sampler=tsampler,
                                prefetch_factor=self.prefetch_factor,
                                persistent_workers=self.persistent_workers,
                                pin_memory=self.pin_memory,
                                )
    

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.val_batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          collate_fn=CollateCustomfn(self.alphabet.pad_idx, self.max_seq_len, self.max_sites_len), 
                          prefetch_factor=self.prefetch_factor, 
                          persistent_workers=self.persistent_workers, 
                          pin_memory=self.pin_memory
                          )

class BDS_PredictDatamodule(pl.LightningDataModule):
    def __init__(self,
                 data_dir:str, 
                 batch_size:int, 
                 alphabet:Alphabet, 
                 num_workers:int=4, 
                 limit_size:int = None,
                 scale_dates:float = 1.0,
                 nb_resamples:int = 1,
                 size_samples:list = None,
                 round_dates:bool = False,
                 different_s: int = 1,
                 max_seq_len:int = 100,
                 max_sites_len:int = 1500
                 ):
        """Datamodule for prediction with BDS data"""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.alphabet = alphabet
        self.num_workers = num_workers
        self.limit_size = limit_size
        self.max_seq_len = max_seq_len
        self.max_sites_len = max_sites_len
        self.scale_dates = scale_dates
        self.nb_resamples = nb_resamples
        self.size_samples = size_samples
        self.round_dates = round_dates
        self.different_s = different_s

    def setup(self,stage=None):
        self.dataset = MsaName(dir=self.data_dir,
                                alphabet=self.alphabet, 
                                limit_size=self.limit_size,
                                scale_dates=self.scale_dates,
                                nb_resamples=self.nb_resamples,
                                size_samples=self.size_samples,
                                round_dates=self.round_dates,
                                different_s= self.different_s,
                                )

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,collate_fn=CollateCustomfnNames(self.alphabet.pad_idx, self.max_seq_len, self.max_sites_len))