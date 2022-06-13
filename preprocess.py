import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import string

def get_statistics(data):

    # Get statistics
    mean_df = data.mean(axis=1)
    median_df = data.median(axis=1)
    std_df = data.std(axis=1)
    cv_df = data.apply(lambda x: np.std(x, ddof=1) / np.mean(x), axis=1)
    summ_df = pd.concat([mean_df, median_df, std_df, cv_df], axis=1)
    summ_df.columns = ["mean", "median", "std", "cv"]

    #print("Writing Data")
    #summ_df.to_csv("data/summmary_statistics.csv")
    return summ_df

def onehote(sequence):

        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        seq2 = [mapping[i] for i in sequence]
        return np.eye(4)[seq2]

def ohe_pad(seqs):

    encodedDNA=[]
    for seq in seqs:
        encodedDNA.append(onehote(seq))
        
    X = tf.keras.preprocessing.sequence.pad_sequences(encodedDNA, padding="post")

    return X

def get_full_data(path="data/", path_data="data/gene_FPKM_200501.csv",
                  path_genes="data/ath_sequences.csv"):

    data = pd.read_csv(path_data, index_col=0)
    genes = pd.read_csv(path_genes, index_col=0)

    # Normalize
    data = data*10**6 / data.sum(axis=0)
    
    print("Reading done...")
    summ_df = preprocess(data)

    full_df = genes.join(data)
    full_df = full_df.join(summ_df)

    print("Merge done...")
    full_df.to_csv(path + "full.csv", index = False)

def genes_with_N(df):

    amb_list = list(string.ascii_uppercase)
    amb_list.remove('A')
    amb_list.remove('C')
    amb_list.remove('T')
    amb_list.remove('G')

    term = np.concatenate( np.where(
            df.terminator_seq.str.contains('|'.join(amb_list))) )
    prom = np.concatenate( np.where(
            df.promoter_seq.str.contains('|'.join(amb_list))) )
    
    genes = df.gene_id[ np.unique(np.concatenate((term, prom))) ]

    return genes

def save_df(df_seq, df_exp, path='data/', pkl_flag=False):

    print("Preprocessing data...")

    summmary_statistics = get_statistics(df_exp)
    df = pd.merge(summmary_statistics, df_seq, 
                left_index=True, right_on='entrez_id')

    genes = genes_with_N(df)

    print(genes)
    # Filter genes with ambiguous values
    df = df[~df.gene_id.isin(genes)]

    df["full_seq"] = df["promoter_seq"] + df["terminator_seq"]

    X_full = ohe_pad( df.full_seq )
    X_prom = ohe_pad( df.promoter_seq )
    X_term = ohe_pad( df.terminator_seq )
    y = df[["mean", "median", "std", "cv"]].values


    if pkl_flag:

        seq_path   = path + 'ohe_seq.pkl'
        prom_path  = path + 'ohe_prom_seq.pkl'
        term_path  = path + 'ohe_term_seq.pkl'
        stats_path = path + 'statistics.pkl'

        with open(seq_path,'wb') as f:
            pickle.dump(X_full, f)

        with open(prom_path,'wb') as f:
            pickle.dump(X_prom, f)

        with open(term_path,'wb') as f:
            pickle.dump(X_term, f)

        with open(stats_path,'wb') as f:
            pickle.dump(y, f)

    
    print("Writing data...")

    np.savez(path + 'preprocessed.npz', full_seq=X_full, promoter_seq=X_prom, 
              terminator_seq=X_term, statistics=y)


if __name__ == "__main__":

    path_exp  = "data/data_atted/original_files/Ath-r.c5-0.expression.combat.txt" 
    path_seq  = "data/data_atted/original_files/atted_r_seqs.csv"

    print("Reading data...")
    df_exp = pd.read_csv(path_exp, sep = "\t")
    df_seq = pd.read_csv(path_seq)

    save_df(df_seq, df_exp, path='data/data_atted/preprocessed/')

    

    
