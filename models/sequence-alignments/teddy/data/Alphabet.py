import numpy as np

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

    def trad(self,char):
        return self.dictionary[char.upper()]

    def __len__(self):
        return len(self.all_tokens)
    
