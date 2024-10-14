from bin.common import OHEncoder
from torch.utils.data import Dataset
import torch
from bin.exceptions import *
from Bio import SeqIO

TRAIN_CHR=["1","2","3","4","5","6","7","8","9","10","11","12"]
VALID_CHR=["13","14","15","16","17"]
TEST_CHR=["18","19","20","21","22"]


class SeqDataset(Dataset):   
    def __init__(self, filename, train_chr=TRAIN_CHR, valid_chr=VALID_CHR, test_chr=TEST_CHR, noise=True, seq_len=200):
        
        self.ids=[]
        self.train_ids=[]
        self.test_ids=[]
        self.valid_ids=[]
        self.tensors={}
        self.info={}
        self.seq_len=seq_len


        OHE = OHEncoder(noise=noise)
        curr_id = 0
        for SR in SeqIO.parse(filename,"fasta"):
            encoded_seq = OHE(SR.seq)
            if not (encoded_seq is None) and len(SR.seq) == seq_len:
                X = torch.tensor(encoded_seq)
                # X = X.unsqueeze(0)  # shape becomes [1, batch_size, seq_len]
                X = X.reshape(1, *X.size())  # torch.Size([4, 200])
                print(X.shape)

                chrom=SR.id.split(":")[-2]
                self.info[curr_id]=SR.id
                if chrom in train_chr:
                    self.ids.append(curr_id)
                    self.train_ids.append(curr_id)
                    # self.tensors[curr_id]=X,y
                    self.tensors[curr_id] = X
                    curr_id += 1
                elif chrom in valid_chr:
                    self.ids.append(curr_id)
                    self.valid_ids.append(curr_id)
                    # self.tensors[curr_id]=X,y
                    self.tensors[curr_id] = X
                    curr_id += 1
                elif chrom in test_chr:
                    self.ids.append(curr_id)
                    self.test_ids.append(curr_id)
                    # self.tensors[curr_id]=X,y
                    self.tensors[curr_id] = X
                    curr_id += 1
                else:
                    print("wrong chromosome",repr(chrom),SR.id,chrom in TRAIN_CHR)
            else: # not encoded
                print("problem with seq", SR.id)    

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.tensors[index]
        
