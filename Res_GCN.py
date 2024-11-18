
import torch
import torch.nn as nn


class GCB(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):

        super(GCB, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()
    def init_weights(self):
      nn.init.xavier_uniform_(self.fc_full.weight)
      nn.init.constant_(self.fc_full.bias, 0)
      nn.init.constant_(self.bn1.weight, 1)
      nn.init.constant_(self.bn1.bias, 0)
      nn.init.constant_(self.bn2.weight, 1)
      nn.init.constant_(self.bn2.bias, 0)

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out

class GRCB(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.conv = GCB(self.atom_fea_len, self.nbr_fea_len)
        self.bn = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus = nn.Softplus()

    def init_weights(self):
        self.conv.init_weights()
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N,M = nbr_fea_idx.shape
        res_atom_out_fea = self.conv(atom_in_fea, nbr_fea, nbr_fea_idx)
        atom_out_fea = atom_in_fea + res_atom_out_fea
        atom_out_fea = self.softplus(self.bn(atom_out_fea))
        return atom_out_fea

class Res_GCN(nn.Module):

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=1, h_fea_len=128, n_h=1, n_resconv=3,
                 classification=False):
        super(Res_GCN, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([GCB(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.res_convs = nn.ModuleList([GRCB(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_resconv)])
        
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for i in range(n_h-1):
                out_features = h_fea_len // 2
                self.fcs.append(nn.Linear(h_fea_len, out_features))
                self.bns.append(nn.BatchNorm1d(out_features))
                h_fea_len = out_features
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        for resconv_func in self.res_convs:
            atom_fea = resconv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out, crys_fea
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for conv_func in self.convs:
            conv_func.init_weights()
        for resconv_func in self.res_convs:
            resconv_func.init_weights()
        nn.init.xavier_uniform_(self.conv_to_fc.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

        if hasattr(self, 'fcs'):
            for fc, bn in zip(self.fcs, self.bns):
                nn.init.xavier_uniform_(fc.weight)
                nn.init.constant_(fc.bias, 0)
                nn.init.constant_(bn.weight, 1)
                nn.init.constant_(bn.bias, 0)
    
    def pooling(self, atom_fea, crystal_atom_idx):
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
