from torch.utils.data import Dataset
import os
from teddy.data.prep_data import prep_data, resample_sites
import torch
from torch.nn.functional import pad
from torch.utils.data import DistributedSampler
import random
import math
import pandas as pd


def _extract_label_from_path(path,
                             scale_dur_target:float = 1.0,
                             ):
    """Extract label from path"""
    try:
        x = path.split('__')
        br,dr,s = float(x[1]),float(x[2]), float(".".join(x[3].split(".")[:2]))
        label = [br/dr, 1/dr * scale_dur_target,s]
        return label
    except:
        return [0,0,0]


class MsaLabels(Dataset):
    """TensorDataset for data and labels"""
    def __init__(self, 
                 dir:str ,
                 alphabet,
                 limit_size = None, 
                 cache_dir = None,
                 scale_dur_target:float = 1.0,
                 scale_dates:float = 1.0,
                 max_size_dataset: int = None,
                 round_dates:bool = False
                 ):
        self.dir = dir
        self.seq_files = [item for item in sorted(os.listdir(self.dir)) if item[-3:]=='.fa']
        if max_size_dataset is not None:
            assert max_size_dataset <= len(self.seq_files), "max_size_dataset should be less than or equal to the number of files in the directory"
            self.seq_files = random.sample(self.seq_files, max_size_dataset)
            for k in range(10):
                print(self.seq_files[k])
        self.alphabet = alphabet
        self.limit_size = limit_size
        self.cache_dir = cache_dir
        self.scale_dur_target = scale_dur_target
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir,exist_ok=True)
        self.scale_dates = scale_dates
        self.round_dates = round_dates

        # you need to have a design.csv file in folder below for this to work
        design_csv_path = self.dir + "design.csv"
        if os.path.exists(design_csv_path):
            self.design_params = pd.read_csv(design_csv_path, sep=';', index_col="output_file").sort_index().reset_index()
        else:
            self.design_params = None

    def __getitem__(self, index):
        name = self.seq_files[index]
        R_0,dur,s = _extract_label_from_path(name, 
                                             scale_dur_target=self.scale_dur_target)
        param_list = [R_0,dur]
        if self.design_params is not None:
            design_params = self.design_params.iloc[index][["sampling_proportion", "clock_rate"]].astype(float).values.tolist()
            param_list.extend(design_params)

        
        if self.cache_dir is not None:
            if os.path.exists(os.path.join(self.cache_dir,name+'.pt')):
                data,shape,minimum_date = torch.load(os.path.join(self.cache_dir,name+'.pt'), weights_only=True)
            else:
                data,shape, minimum_date = prep_data(os.path.join(self.dir,name),self.alphabet,s, scale_dates=self.scale_dates, round_dates=self.round_dates)
                torch.save((data,shape,minimum_date),os.path.join(self.cache_dir,name+'.pt'))
        else: 
            data,shape,minimum_date = prep_data(os.path.join(self.dir,name),self.alphabet,s, scale_dates=self.scale_dates, round_dates=self.round_dates)
        param_list.append(minimum_date)
        data,shape = resample_sites(data,shape,self.limit_size)
        return (data,shape), torch.Tensor(param_list)
    
    def __len__(self):
        return len(self.seq_files)
class CollateCustomfn:
    def __init__(self, pad_idx, max_seq_len, max_sites_len, flatten_mode=True):
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len + 1
        self.max_sites_len = max_sites_len + 2
        self.flatten_mode = flatten_mode
    
    def __call__(self, batch):
        data = [pad(item[0][0], (0, self.max_sites_len-item[0][0].shape[1], 
                                0, self.max_seq_len-item[0][0].shape[0]), 
                   value=self.pad_idx) for item in batch]
        data = torch.stack(data)
        
        shapes = torch.stack([item[0][1]
                             for item in batch])
        labels = torch.stack([item[1] for item in batch])

        if self.flatten_mode:
            data = torch.flatten(torch.as_tensor(data, dtype=torch.float32), start_dim=1)

        # return (data, shapes), labels
        return data, labels

class MsaName(Dataset):
    """TensorDataset for data and labels"""
    def __init__(self, 
                 dir,
                 alphabet,
                 limit_size = None,
                 scale_dates:float = 1.0,
                 nb_resamples:int = 1,
                 size_samples:list = None,
                 round_dates:bool = False,
                 different_s: int = 1,
                 ):
        self.dir = dir
        self.seq_files = [item for item in os.listdir(self.dir) if item[-3:]=='.fa']
        self.alphabet = alphabet
        self.limit_size = limit_size
        self.scale_dates = scale_dates
        self.nb_resamples = nb_resamples
        self.size_samples = size_samples
        self.round_dates = round_dates
        self.different_s = different_s

    def __getitem__(self, index):
        name = self.seq_files[index%len(self.seq_files)]
        if self.different_s > 1:
            s = 10**random.uniform(-2,0)
        else:
            _,_,s = _extract_label_from_path(name)
        if self.size_samples is not None:
            subsample = random.randint(self.size_samples[0], self.size_samples[1])
        else:
            subsample = None


        data,shape = prep_data(os.path.join(self.dir,name),
                               self.alphabet,
                               sampling_prop = s,
                               scale_dates=self.scale_dates,
                               subsample=subsample,
                               round_dates=self.round_dates
                               )
        data,shape = resample_sites(data,shape,self.limit_size)
        return name, (data,shape), s
    
    def __len__(self):
        return len(self.seq_files)*self.nb_resamples*self.different_s
    
class CollateCustomfnNames:
    def __init__(self,pad_idx,
                 max_seq_len,
                 max_sites_len):
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len +1 # +1 to account for the CLS token
        self.max_sites_len = max_sites_len +2 # +2 to account for the CLS token and the date
    
    def __call__(self,batch):
        names = ([item[0] for item in batch])
        data = [pad(item[1][0],(0,self.max_sites_len-item[1][0].shape[1],0,self.max_seq_len -item[1][0].shape[0]),value=self.pad_idx) for item in batch]
        data = [pad(item[1][0],(0,self.max_sites_len-item[1][0].shape[1],0,self.max_seq_len -item[1][0].shape[0]),value=self.pad_idx) for item in batch]
        data = torch.stack(data)
        shapes = torch.stack([torch.Tensor(item[1][1]) for item in batch])
        s = torch.Tensor([item[2] for item in batch])
        return names,(data,shapes), s


def verif_shape(shape,condition):
    """Check if the shape is in the condition"""
    condition_seq = condition["condition_seq"]
    condition_sites = condition["condition_sites"]
    if shape[0].item()>= condition_seq[0] and shape[0].item() <= condition_seq[1] and shape[2].item() >= condition_sites[0] and shape[2].item() <= condition_sites[1]:
        return True
    else:
        return False

def get_condition_index(shape,conditions):
    """Get the index of the condition that is verified"""
    for i,condition in enumerate(conditions):
        if verif_shape(shape,condition):
            return i
    raise ValueError("No condition verified")


class DistributedVariableBatchSampler(DistributedSampler):
    def __init__(self,
                dataset: Dataset,
                conditional_batch_sizes: list,
                shuffle: bool = True,
                seed: int = 0,
                drop_last: bool = False,
                
                ) -> None:
        super().__init__(dataset=dataset, 
                         shuffle=shuffle, 
                         seed=seed,
                         drop_last=drop_last,
                         )
        
        self.start_idx  = 0
        self.epoch = 0
        self.conditional_batch_sizes = conditional_batch_sizes
        self.__buckets = []
        self.__buckets = self.__prepare_buckets()
        self.__batches = []
        self.__batches = self.__prepare_batches(0)

    def __prepare_batches(self,epoch):
        __batches = []
        __leftovers = []
        minimal_size = min([self.conditional_batch_sizes[i]["batch_size"] for i in range(len(self.conditional_batch_sizes))])
        for i,bucket in enumerate(self.__buckets):
            if len(bucket) > 0:
                if self.shuffle:
                    g = torch.Generator()
                    g.manual_seed(self.seed + epoch + i)
                    indices = torch.randperm(len(bucket), generator=g).tolist()
                else:
                    indices = list(range(len(bucket)))
                steps = self.conditional_batch_sizes[i]["batch_size"]
                slices = [indices[j*steps:(j+1)*steps] for j in range(math.ceil(len(indices)/steps))]
                if len(slices[-1]) < self.conditional_batch_sizes[i]["batch_size"]:
                    __leftovers.extend(slices[-1])
                    slices = slices[:-1]
                __batches.extend(slices)
        if len(__leftovers) > 0:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + epoch + len(self.__buckets))
                indices = torch.randperm(len(__leftovers), generator=g).tolist()
            else:
                indices = list(range(len(__leftovers)))
            steps = minimal_size
            slices = [__leftovers[i*steps:(i+1)*steps] for i in range(math.ceil(len(indices)/steps))]
            if len(slices[-1]) < minimal_size:
                # sample already sampled to complete the last batch
                if self.shuffle:
                    random.seed(self.seed + epoch + len(self.__buckets) + 1)
                    l = random.sample(range(len(self.dataset)), minimal_size - len(slices[-1]))
                    slices[-1].extend(l)
                else:
                    l = range(minimal_size - len(slices[-1]))
                    slices[-1] = slices[-1] + l
            __batches.extend(slices)

        if len(__batches) % self.num_replicas != 0:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + epoch + len(self.__buckets) + 2)
                last_batches = random.sample(__batches, self.num_replicas - len(__batches) % self.num_replicas)
            else:
                last_batches = [__batches[i*self.num_replicas: (i+1)*self.num_replicas] for i in range(self.num_replicas-len(__batches)%self.num_replicas)]
            __batches.extend(last_batches)

        return __batches
        



    def __iter__(self):
        if self.shuffle:
            self.__batches = self.__prepare_batches(self.epoch)
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.__batches), generator=g).tolist()
        else:
            indices = list(range(len(self.__batches)))
        
        for index in indices[self.rank + self.start_idx::self.num_replicas]:
            yield self.__batches[index]

    def __prepare_buckets(self):
        if self.__buckets != []:
            return self.__buckets
        if self.rank == 0:
            print("Preparing buckets")
        buckets = [[] for _ in range(len(self.conditional_batch_sizes))]
        indices = list(range(len(self.dataset)))
        for idx in indices:
            x = self.dataset[idx][0][1]
            idx_bucket = get_condition_index(x, self.conditional_batch_sizes)
            buckets[idx_bucket].append(idx)
        if self.rank == 0:
            print("Buckets prepared")
            print("Number of buckets: ", sum([len(bucket)/self.conditional_batch_sizes[i]["batch_size"] for i,bucket in enumerate(buckets)]))
        return buckets

    def __len__(self) -> int:
        return math.ceil(len(self.__batches)/self.num_replicas)
    
    def state_dict(self):
        return dict(
            epoch=self.epoch,
            conditional_batch_sizes=self.conditional_batch_sizes,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            nb_replicas=self.num_replicas,
        )
    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self.conditional_batch_sizes = state["conditional_batch_sizes"]
        self.shuffle = state["shuffle"]
        self.drop_last = state["drop_last"]
        self.nb_replicas = state["nb_replicas"]

    @staticmethod
    def from_state_dict(dataset, state, start_step: int = 0):
        sampler = DistributedVariableBatchSampler(dataset, **state)
        sampler.set_starting_step(start_step, nb_replicas=state["num_replicas"])

        return sampler
    def set_starting_step(self, step: int, nb_replicas: int = 1):
        """Call this to set the starting step if resuming from a checkpoint"""
        self.start_idx = step * nb_replicas
    
