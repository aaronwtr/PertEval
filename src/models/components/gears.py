from typing import Optional
from gears.utils import get_similarity_network, GeneSimNetwork
from gears import PertData

import torch
import torch.nn as nn

from torch_geometric.nn import SGConv


class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.network(x)


class GEARSNetwork(torch.nn.Module):
    """
    GEARS model

    """

    def __init__(
            self,
            hidden_size: int,
            num_go_gnn_layers: int,
            num_gene_gnn_layers: int,
            decoder_hidden_size: int,
            num_similar_genes_go_graph: int,
            num_similar_genes_co_express_graph: int,
            coexpress_threshold: float,
            uncertainty: bool,
            uncertainty_reg: float,
            direction_lambda: float,
            G_go: Optional[torch.Tensor] = None,
            G_go_weight: Optional[torch.Tensor] = None,
            G_coexpress: Optional[torch.Tensor] = None,
            G_coexpress_weight: Optional[torch.Tensor] = None,
            no_perturb: bool = False,
            pert_emb_lambda: float = 0.2,
            num_genes: int = None,
            num_perts: int = None
    ):
        super(GEARSNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.uncertainty = uncertainty
        self.num_layers = num_go_gnn_layers
        self.indv_out_hidden_size = decoder_hidden_size
        self.num_similar_genes_go_graph = num_similar_genes_go_graph
        self.num_similar_genes_co_express_graph = num_similar_genes_co_express_graph
        self.G_go = G_go
        self.G_go_weight = G_go_weight
        self.G_coexpress = G_coexpress
        self.G_coexpress_weight = G_coexpress_weight
        self.coexpress_threshold = coexpress_threshold
        self.uncertainty_reg = uncertainty_reg
        self.direction_lambda = direction_lambda
        self.num_layers_gene_pos = num_gene_gnn_layers
        self.no_perturb = no_perturb
        self.pert_emb_lambda = pert_emb_lambda
        self.num_genes = num_genes
        self.num_perts = num_perts

        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, self.hidden_size)

        # gene/globel perturbation embedding dictionary lookup
        self.gene_emb = nn.Embedding(self.num_genes, self.hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, self.hidden_size, max_norm=True)

        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([self.hidden_size, self.hidden_size, self.hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([self.hidden_size, self.hidden_size, self.hidden_size], last_layer_act='ReLU')

        # gene co-expression GNN
        self.G_coexpress = self.G_coexpress
        self.G_coexpress_weight = self.G_coexpress_weight

        self.emb_pos = nn.Embedding(self.num_genes, self.hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(self.hidden_size, self.hidden_size, 1))

        ### perturbation gene ontology GNN
        self.G_sim = self.G_go
        self.G_sim_weight = self.G_go_weight

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(self.hidden_size, self.hidden_size, 1))

        # decoder shared MLP
        self.recovery_w = MLP([self.hidden_size, self.hidden_size * 2, self.hidden_size], last_layer_act='linear')

        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               self.hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)

        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, self.hidden_size,
                                     self.hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                               self.hidden_size + 1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)

        # batchnorms
        self.bn_emb = nn.BatchNorm1d(self.hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(self.hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(self.hidden_size)

        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([self.hidden_size, self.hidden_size * 2, self.hidden_size, 1],
                                     last_layer_act='linear')

    def forward(self, data):
        """
        Forward pass of the model
        """
        x, pert_idx = data.x, data.pert_idx
        if self.no_perturb:
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique())

            ## get base gene embeddings
            emb = self.gene_emb(
                torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ))
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)

            pos_emb = self.emb_pos(
                torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ))
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T

            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))))

            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_go, self.G_go_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP
            base_emb = self.transform(base_emb)
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis=2)
            out = w + self.indv_b1

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs, self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)

            return torch.stack(out)

    def model_initialize(self, pertmodule) -> None:
        """Initialize the model"""
        pert_data = pertmodule.pert_data
        if self.G_coexpress is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(
                network_type='co-express',
                adata=pert_data.adata,
                threshold=self.coexpress_threshold,
                k=self.num_similar_genes_co_express_graph,
                data_path=pert_data.data_path,
                data_name='',
                split=pert_data.split,
                seed=pert_data.seed,
                train_gene_set_size=pert_data.train_gene_set_size,
                set2conditions=pert_data.set2conditions
            )

            sim_network = GeneSimNetwork(edge_list, pertmodule.gene_list, node_map=pert_data.node_map)
            self.G_coexpress = sim_network.edge_index
            self.G_coexpress_weight = sim_network.edge_weight

        if self.G_go is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(
                network_type='go',
                adata=pert_data.adata,
                threshold=self.coexpress_threshold,
                k=self.num_similar_genes_go_graph,
                pert_list=pertmodule.pert_list,
                data_path=pert_data.data_path,
                data_name='',
                split=pert_data.split,
                seed=pert_data.seed,
                train_gene_set_size=pert_data.train_gene_set_size,
                set2conditions=pert_data.set2conditions,
                default_pert_graph=pert_data.default_pert_graph
            )

            sim_network = GeneSimNetwork(edge_list, pertmodule.pert_list, node_map=pert_data.node_map_pert)
            self.G_go = sim_network.edge_index
            self.G_go_weight = sim_network.edge_weight
