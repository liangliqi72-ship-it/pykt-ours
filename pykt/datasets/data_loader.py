#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


def parse_group_ids(s, seq_len, unknown_group=8):
    if not isinstance(s, str) or len(s.strip()) == 0 or s.strip().lower() == "nan":
        return [unknown_group] * seq_len

    vals = []
    for item in s.split(","):
        item = item.strip()
        if len(item) == 0:
            continue

        try:
            v = int(float(item))
            if v < 0 or v > unknown_group:
                v = unknown_group
        except Exception:
            v = unknown_group

        vals.append(v)

    vals = vals[:seq_len]
    if len(vals) < seq_len:
        vals += [unknown_group] * (seq_len - len(vals))

    return vals


def parse_dugp_alpha_feats(s, seq_len, feat_dim=5):
    if not isinstance(s, str) or len(s.strip()) == 0 or s.strip().lower() == "nan":
        return [[0.0] * feat_dim for _ in range(seq_len)]

    steps = []
    for item in s.split(","):
        item = item.strip()
        if len(item) == 0:
            continue

        vals = []
        for x in item.split("|"):
            x = x.strip()
            if len(x) == 0:
                continue

            try:
                v = float(x)
                if not np.isfinite(v):
                    v = 0.0
            except Exception:
                v = 0.0

            vals.append(v)

        vals = vals[:feat_dim]
        if len(vals) < feat_dim:
            vals += [0.0] * (feat_dim - len(vals))

        steps.append(vals)

    steps = steps[:seq_len]
    if len(steps) < seq_len:
        steps += [[0.0] * feat_dim for _ in range(seq_len - len(steps))]

    return steps


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

class KTDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).
    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds, qtest=False, force_reprocess=False):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        self.group_k = 8
        self.unknown_group = self.group_k
        self.alpha_feat_dim = 5
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + f"_dugp_k{self.group_k}_a{self.alpha_feat_dim}_qtest.pkl"
        else:
            processed_data = file_path + folds_str + f"_dugp_k{self.group_k}_a{self.alpha_feat_dim}.pkl"

        if force_reprocess or (not os.path.exists(processed_data)):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                for key in self.dori:
                    self.dori[key] = self.dori[key]#[:100]
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length
        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get
        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue

            # ours: group_id is now a causal prefix sequence, not a scalar
            # original length: [200], after slicing: [199]
            if key == "group_id":
                seqs = self.dori[key][index][:-1].long()
                shft_seqs = self.dori[key][index][1:].long()

                mask_bool = mseqs.bool()

                seqs = torch.where(
                    mask_bool,
                    seqs,
                    torch.full_like(seqs, self.unknown_group),
                )

                shft_seqs = torch.where(
                    mask_bool,
                    shft_seqs,
                    torch.full_like(shft_seqs, self.unknown_group),
                )

                dcur[key] = seqs
                dcur["shft_" + key] = shft_seqs
                continue

            # ours: alpha_feat shape is [200, 5], after slicing [199, 5]
            if key == "alpha_feat":
            # print(f"key: {key}, len: {len(self.dori[key])}")
                seqs = self.dori[key][index][:-1].float()
                shft_seqs = self.dori[key][index][1:].float()
                mask_float = mseqs.float().unsqueeze(-1)
                dcur[key] = seqs * mask_float
                dcur["shft_" + key] = shft_seqs * mask_float
                continue
            
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs

            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs

        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]

        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest


    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.
        Returns: 
            (tuple): tuple containing
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": [], "group_id": [], "alpha_feat": []}

        # seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        df = pd.read_csv(sequence_path)#[0:1000]
        #==============================================
        df["_orig_idx"] = range(len(df))
        #===========================================
        df = df[df["fold"].isin(folds)]

        # ours: load student group mapping
        #group_path = os.path.join(os.path.dirname(sequence_path), "student_group_k8.csv")

        #if os.path.exists(group_path):
            #group_df = pd.read_csv(group_path)
            #uid2group = dict(
                #zip(
                    #group_df["uid"].astype(str),
                    #group_df["group_id"].astype(int)
                #)
            #)
            #print(f"Loaded student group file: {group_path}")
        #else:
            #uid2group = {}
            #print(f"[Warning] student_group_k8.csv not found at {group_path}. Use group_id=0.")

        seq_path = Path(sequence_path)
        dugp_group_path = seq_path.with_name(
            seq_path.stem + "_dugp_causal_group_k8" + seq_path.suffix
        )

        use_dugp_group = dugp_group_path.exists()

        if use_dugp_group:
            print(f"Read DUGP causal group from: {dugp_group_path}")
            group_df = pd.read_csv(dugp_group_path)

            raw_row_count = len(pd.read_csv(sequence_path, usecols=["fold"]))
            if len(group_df) != raw_row_count:
                raise ValueError(
                    f"DUGP group file row count mismatch: "
                    f"group_df={len(group_df)}, sequence_df={raw_row_count}"
                )

            group_col = "dugp_group_ids"
            alpha_col = "dugp_alpha_feats"

            if group_col not in group_df.columns:
                raise ValueError(f"{group_col} not found in {dugp_group_path}")

            if alpha_col not in group_df.columns:
                raise ValueError(f"{alpha_col} not found in {dugp_group_path}")

            group_seq_list = group_df[group_col].astype(str).tolist()
            alpha_feat_list = group_df[alpha_col].astype(str).tolist()
        else:
            raise FileNotFoundError(
                f"DUGP causal group file not found at {dugp_group_path}. "
                "Please run generate_dugp_causal_group.py first."
            )


        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        for i, row in df.iterrows():

            # ours: get causal prefix group_id sequence and alpha_feat sequence
            orig_i = int(row["_orig_idx"]) if "_orig_idx" in row else i
            seq_len = len(str(row["responses"]).split(","))

            if use_dugp_group:
                gids = parse_group_ids(
                    group_seq_list[orig_i],
                    seq_len=seq_len,
                    unknown_group=self.unknown_group,
                )

                afeats = parse_dugp_alpha_feats(
                    alpha_feat_list[orig_i],
                    seq_len=seq_len,
                    feat_dim=self.alpha_feat_dim,
                )

                if len(gids) != seq_len:
                    raise ValueError(
                        f"group_id length mismatch at row {orig_i}: "
                        f"{len(gids)} vs {seq_len}"
                    )

                if len(afeats) != seq_len:
                    raise ValueError(
                        f"alpha_feat length mismatch at row {orig_i}: "
                        f"{len(afeats)} vs {seq_len}"
                    )

                dori["group_id"].append(gids)
                dori["alpha_feat"].append(afeats)

            else:
                dori["group_id"].append([self.unknown_group] * seq_len)
                dori["alpha_feat"].append([[0.0] * self.alpha_feat_dim] * seq_len)


            #use kc_id or question_id as input
            dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])
        for key in dori:
            if key in ["rseqs", "alpha_feat"]:#in ["smasks", "tseqs"]:
                dori[key] = FloatTensor(dori[key])
            else:
                dori[key] = LongTensor(dori[key])

        mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            
            return dori, dqtest
        return dori
