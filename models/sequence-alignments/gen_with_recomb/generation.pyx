# generation.pyx
from libcpp.string cimport string

cdef extern from "external_sim.cpp":
    void generate_all(
        double birth_rate,
        double death_rate,
        double sampling_prop,
        double coinfect_capacity,
        double recomb_prop,
        double superinfect_prop,
        double max_time,
        int max_trys,
        int number_of_samples,
        string tmp_dir,
        bint SIR,
        string evol_model,
        double clock_rate,
        string iqtree_path,
        string output_path,
        int num_sites,
        string recombinant_points_file,
        bint verbose_nb_diff,
        bint delete_files,
        string root_seq_path
    ) except +

import cython
from Bio import SeqIO
import torch
import random as rd
import numpy as np
import math 

@cython.boundscheck(False)
@cython.wraparound(False)
def generate(
    birth_rate,
    death_rate,
    sampling_prop,
    coinfect_capacity,
    recomb_prop,
    superinfect_prop,
    max_time,
    max_trys,
    number_of_samples,
    tmp_dir,
    SIR,
    evol_model,
    clock_rate,
    iqtree_path,
    output_path,
    num_sites,
    recombinant_points_file,
    verbose_nb_diff,
    delete_files,
    root_seq_path=""):

    cdef string c_tmp_dir = tmp_dir.encode('utf-8')
    cdef string c_evol_model = evol_model.encode('utf-8')
    cdef string c_iqtree_path = iqtree_path.encode('utf-8')
    cdef string c_output_path = output_path.encode('utf-8')
    cdef string c_recombinant_points_file = recombinant_points_file.encode('utf-8')
    cdef string c_root_seq_path = root_seq_path.encode('utf-8')
    
    generate_all(
        birth_rate,
        death_rate,
        sampling_prop,
        coinfect_capacity,
        recomb_prop,
        superinfect_prop,
        max_time,
        max_trys,
        number_of_samples,
        c_tmp_dir,
        SIR,
        c_evol_model,
        clock_rate,
        c_iqtree_path,
        c_output_path,
        num_sites,
        c_recombinant_points_file,
        verbose_nb_diff,
        delete_files,
        c_root_seq_path
    )


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
    data = data[:,test]
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

class Alphabet: 
    def __init__(self,
                standard_tokens:list,
                added_tokens:list = ['<s>','<cls>','<pad>','<nb_seq>', '<nb_sites>','<nb_nonconstant_sites>'],
                start_special_token:str = "<",
                end_special_token:str = ">",
                ):
        self.standard_tokens = standard_tokens
        self.added_tokens = added_tokens
        self.dictionary = {token: i for i, token in enumerate(standard_tokens+added_tokens)}
        self.all_tokens = standard_tokens+added_tokens
        self.s_idx = self.dictionary['<s>']
        self.cls_idx = self.dictionary['<cls>']
        self.pad_idx = self.dictionary['<pad>']
        self.seq_idx = self.dictionary['<nb_seq>']
        self.site_idx = self.dictionary['<nb_sites>']
        self.nonconstant_site_idx = self.dictionary['<nb_nonconstant_sites>']
        
        self.start_special_token = start_special_token
        self.end_special_token = end_special_token
        self.trad_vectorized = np.vectorize(self.trad)

    def trad(self,ch):
        return self.dictionary[ch.upper()]

    def __len__(self):
        return len(self.all_tokens)

DNA_alphabet = Alphabet(list("ATGCX-"))
