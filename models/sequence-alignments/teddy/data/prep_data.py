from Bio import SeqIO
import torch
import random as rd
import numpy as np
import math 

def _extract_data_from_path(path,
                            alphabet,
                            scale_dates:float = 1.0,
                            subsample:int = None,
                            process_dates:bool = True,
                            round_dates:bool = False,
                            ):
    test = SeqIO.parse(path, format="fasta")
    data = []
    nb_seq = 0
    dates = []
    for record in test:
        data.append(record.seq)
        if process_dates:
            date = float(record.name.split('|')[1])*scale_dates
            if round_dates:
                date = math.floor(date)
            dates.append(date)
        else:
            dates.append(0)
        nb_seq +=1
    data = alphabet.trad_vectorized(np.array(data))
    if subsample is not None:
        assert subsample > 0, "Subsample must be greater than 0"
        assert subsample <= nb_seq, "Subsample must be less than or equal to the number of sequences"
        indices = rd.sample(range(nb_seq), subsample)
        data = data[indices]
        dates = [dates[i] for i in indices]
    output = np.zeros((data.shape[0],data.shape[1]+2))
    output[:,2:] = data
    output[:,0] = dates
    output[:,1] = alphabet.cls_idx
    return output, nb_seq



def _add_row_cls(data,alphabet):
    """Add cls row to tokens"""
    cls = torch.Tensor([[alphabet.cls_idx]*len(data[0])])
    data = torch.cat([cls,data])
    return data

def prep_data(path: str,
              alphabet,
              sampling_prop:float,
              scale_dates:float = 1.0,
              subsample:int = None,
              process_dates:bool = True,
              round_dates:bool = False,
              ):
    data,nb_seq = _extract_data_from_path(path,alphabet, scale_dates=scale_dates, subsample=subsample, process_dates=process_dates, round_dates=round_dates)
    ratio = 1.0 if subsample is None else subsample / nb_seq
    nb_seq = subsample if subsample is not None else nb_seq
    s = sampling_prop * ratio
    data = torch.Tensor(data)
    
    # Remove the minimum date
    data[:,0] = data[:,0] - data[:,0].min()

    test = (data[1:,:] - data[:-1,:]).abs().sum(axis=0) != 0
    nb_sites = len(test) -2
    test[1] = True # to keep the "cls" column token (date is on the first column)
    test[0] = True # to keep the dates 
    # data = data[:,test]
    new_nb_sites = data.shape[1] -2

    data = _add_row_cls(data,alphabet)
    data[0,0] = s
    
    shape = (nb_seq,nb_sites,new_nb_sites)
    return (data,shape)

def resample_sites(data,
                   shape,
                   limit_size):
    nb_seq, nb_sites, new_nb_sites = shape
    if limit_size is not None and new_nb_sites > limit_size:
        resample = sorted(rd.sample(range(2, new_nb_sites + 2),limit_size))
        data = data[:,[0,1]+resample]
        new_nb_sites = limit_size
        shape = torch.Tensor([nb_seq,nb_sites,new_nb_sites])
        return (data, shape)
    shape = torch.Tensor([nb_seq,nb_sites,new_nb_sites])
    return (data, shape)