import sys,os
sys.path.append("../../utilities")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.model_selection import train_test_split
from data_preprocessing.initial_embeddings.GraphBERTEmbedder import GraphBERTEmbedder
import os
import json
import numpy as np
import utilities as u
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class GraphDataPreparation:
    def __init__(self, kg_name,model_name_init = "sentence-transformers/all-MiniLM-L6-v2", entities_embd_path = None, edges_embd_path=None, is_directed=True, with_self_loop= False, emb_dim = 256):
        self.entities_embd_path = entities_embd_path
        self.edges_embd_path = edges_embd_path
        if model_name_init and model_name_init.startswith("random_"):
            self.random_embd_seed = int(model_name_init.split("_", 1)[1])
            self.model_name_init = "random"
        else:
            self.random_embd_seed = 42
            self.model_name_init = model_name_init
        self.dataset = kg_name
        self.kg_path = "datasets/"+kg_name+".json"
        self.emb_dim = emb_dim
        self.built_graph = None
        self.nxGraph = None
        self.nodes_index = None
        self.is_directed = is_directed
        self.predicate_to_id = {}  # Dictionary to store predicate type mappings
        self.with_self_loop = with_self_loop
        self.label_encoder = None
        self.gs_to_graph = None
        self.graph_to_gs = None

    def decode_indexes(self):
        inverted_dict = {value: key for key, value in self.nodes_index.items()}
        return inverted_dict

    def build_networkx_graph(self):
        print("Building NetworkX graph")
        graph_data = u.read_json_file(self.kg_path)
        entities_embeddings, edge_embeddings = self._resolve_embeddings(graph_data)

        # Create a NetworkX graph
        if self.is_directed:
            self.nxGraph = nx.DiGraph()
        else:
            self.nxGraph = nx.Graph()

        # Add nodes with their embeddings
        for node, emb in entities_embeddings.items():
            self.nxGraph.add_node(node, emb=emb.clone().detach())

        # Add edges with or without embeddings
        for i in range(len(graph_data)):
            subject = graph_data[i]['subject']
            obj = graph_data[i]['object']
            pred = graph_data[i]['predicate']
            edge_data = edge_embeddings[pred] if edge_embeddings is not None else None
            self.nxGraph.add_edge(subject, obj, emb=edge_data)

        # Create node to index mapping
        self.nodes_index = {node: i for i, node in enumerate(self.nxGraph.nodes())}

    def build_torch_geometric_data(self):
        print("Building PyTorch Geometric Data object")
        # Extract node features
        node_features = [self.nxGraph.nodes[node]['emb'] for node in self.nxGraph.nodes()]
        # Extract edge index and edge attributes (if available)
        edge_index = [(self.nodes_index[u], self.nodes_index[v]) for u, v in self.nxGraph.edges()]
        edge_attr = [self.nxGraph.edges[u, v]['emb'] for u, v in self.nxGraph.edges()] if self.edges_embd_path else None
        # Convert to PyTorch tensors
        x = torch.stack(node_features).squeeze(1)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # Handle edge attributes if they exist
        edge_attr_tensor = torch.stack(edge_attr).squeeze(1) if edge_attr is not None else None
        if not self.is_directed:
            edge_index = to_undirected(edge_index)
        # Include edge attributes in the Data object if they exist
        self.built_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_tensor)
        return self.built_graph
    def prepare_graph(self):
        self.build_networkx_graph()
        return self.build_torch_geometric_data()
    def split_data(self):
        transform = RandomLinkSplit(is_undirected=not self.is_directed, add_negative_train_samples=False)
        return transform(self.built_graph)
    def get_connected_components(self):
        connected_components = list(nx.connected_components(self.nxGraph.to_undirected()))
        return connected_components
    def get_subgraph_for_component(self, component):
        subgraph_nodes = list(component)
        subgraph = self.nxGraph.subgraph(subgraph_nodes)
        # Create node to index mapping for the subgraph
        subgraph_index = {node: i for i, node in enumerate(subgraph.nodes())}
        # Extract node features and edge index
        node_features = [subgraph.nodes[node]['emb'] for node in subgraph.nodes()]
        edge_index = [(subgraph_index[u], subgraph_index[v]) for u, v in subgraph.edges()]
        edge_attr = [subgraph.edges[u, v]['emb'] for u, v in subgraph.edges()] if self.edges_embd_path else None
        # Convert to PyTorch tensors
        x = torch.stack(node_features).squeeze(1)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.stack(edge_attr).squeeze(1) if edge_attr is not None else None
        if not self.is_directed:
            edge_index = to_undirected(edge_index)
        # Include edge attributes in the Data object if they exist
        subgraph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_tensor)
        return subgraph_data

    def remove_self_loops_from_json(self,graph_data):
        """
        Supprime les relations réflexives (self-loops) d'un graphe représenté en JSON.

        Args:
            graph_data (list[dict]): Liste de dictionnaires représentant les relations du graphe.

        Returns:
            list[dict]: Graphe nettoyé sans relations réflexives.
        """
        cleaned_graph = []

        for triplet in graph_data:
            # Vérifie si le sujet est différent de l'objet
            if triplet["subject"] != triplet["object"]:
                cleaned_graph.append(triplet)
        return cleaned_graph

    def _resolve_embeddings(self, graph_data):
        # Priority 1: explicit paths provided
        if self.entities_embd_path and self.edges_embd_path:
            print(f"[INFO] Loading embeddings from explicit paths: {self.entities_embd_path}, {self.edges_embd_path}")
            return u.read_pickle_file(self.entities_embd_path), u.read_pickle_file(self.edges_embd_path)

        # Priority 2: model name provided → check cache, else embed and persist
        if self.model_name_init and self.model_name_init != "random":
            output_init_embeddings_path = self.dataset + "_init_embd"
            model_short_name = self.model_name_init.split("/")[-1]
            auto_entities_path = os.path.join(output_init_embeddings_path, f"Entities_{model_short_name}.pickle")
            auto_edges_path = os.path.join(output_init_embeddings_path, f"Predicates_{model_short_name}.pickle")

            if os.path.exists(auto_entities_path) and os.path.exists(auto_edges_path):
                print(f"[INFO] Found cached embeddings for '{model_short_name}', loading from {output_init_embeddings_path}")
                return u.read_pickle_file(auto_entities_path), u.read_pickle_file(auto_edges_path)

            print(f"[INFO] No cached embeddings found, running GraphBERTEmbedder with '{self.model_name_init}'")
            os.makedirs(output_init_embeddings_path, exist_ok=True)
            gbe = GraphBERTEmbedder(self.kg_path, output_init_embeddings_path, self.model_name_init)
            return gbe.run()

        # Priority 3: no model name → random embeddings
        print(f"[INFO] Using RANDOM embeddings (seed={self.random_embd_seed})")
        nodes = sorted(set(s["subject"] for s in graph_data) | set(s["object"] for s in graph_data))
        predicates = sorted({entry["predicate"] for entry in graph_data})
        gen = torch.Generator().manual_seed(self.random_embd_seed)
        return (
            {node: torch.rand(1, self.emb_dim, generator=gen) for node in nodes},
            {p: torch.rand(1, self.emb_dim, generator=gen) for p in predicates},
        )

    def build_networkx_graph_type(self):
        print("Building NetworkX graph with unique relation type IDs")
        graph_data = u.read_json_file(self.kg_path)
        entities_embeddings, edge_embeddings = self._resolve_embeddings(graph_data)





        # Initialize NetworkX graph based on directed flag
        #self.nxGraph = nx.DiGraph() if self.is_directed else nx.Graph()
        self.nxGraph = nx.MultiDiGraph() if self.is_directed else nx.MultiGraph()
        unique_subjects = set(s["subject"] for s in graph_data)
        unique_objects = set(s["object"] for s in graph_data)
        # Add nodes with embeddings
        for node, emb in entities_embeddings.items():
            if (node in unique_subjects or node in unique_objects):
                self.nxGraph.add_node(node, emb=emb.clone().detach())
        # Generate a unique ID for each predicate
        unique_predicates = sorted({entry['predicate'] for entry in graph_data})  # Sorted for determinism
        self.predicate_to_id = {predicate: idx for idx, predicate in enumerate(unique_predicates)}
        # unique_predicates = {entry['predicate'] for entry in graph_data}
        # self.predicate_to_id = {predicate: idx for idx, predicate in enumerate(unique_predicates)}

        print("----------- Unique relation ------->",len(self.predicate_to_id.keys()))
        # Add edges with embeddings and relation type ID
        for entry in graph_data:
            subject = entry['subject']
            obj = entry['object']
            pred = entry['predicate']
            edge_type = self.predicate_to_id[pred]  # Get unique ID for this predicate
            edge_data = edge_embeddings[pred] if edge_embeddings is not None else None
            self.nxGraph.add_edge(subject, obj, emb=edge_data, type=edge_type)

        # Create node to index mapping
        self.nodes_index = {node: i for i, node in enumerate(self.nxGraph.nodes())}

    def build_torch_geometric_data_with_types(self):
        print("Building PyTorch Geometric Data object with unique relation type IDs")

        # Récupérer les embeddings des nœuds
        node_features = [self.nxGraph.nodes[node]['emb'] for node in self.nxGraph.nodes()]

        edge_index = []
        edge_type = []
        edge_attr = []

        # Parcourir toutes les arêtes dans MultiDiGraph
        for u, v, key, data in self.nxGraph.edges(keys=True, data=True):
            edge_index.append((self.nodes_index[u], self.nodes_index[v]))
            edge_type.append(data["type"])  # ID unique pour chaque relation
            edge_attr.append(data["emb"] if "emb" in data else None)

        # Conversion en tenseurs PyTorch
        x = torch.stack(node_features).squeeze(1)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = torch.stack(edge_attr).squeeze(1) if edge_attr[0] is not None else None

        # Convertir en graphe non dirigé si nécessaire
        if not self.is_directed:
            edge_index = to_undirected(edge_index)

        # Stocker le graphe sous PyTorch Geometric Data
        self.built_graph = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr)
        return self.built_graph


    def prepare_graph_with_type(self):
        self.build_networkx_graph_type()
        return self.build_torch_geometric_data_with_types()

    def get_list_indexes(self, list):
        """
        gdp:
        CC_list:
        """
        indexes = [self.nodes_index[cc] for cc in list]
        return indexes

    def prepare_graph_with_type_label(self, excel_path: str):
        """
        Prépare le graphe PyTorch Geometric avec types de relations et ajoute les labels
        des termes à partir d’un fichier Excel unique.

        Le fichier Excel doit contenir au moins deux colonnes :
        - 'term' : nom du nœud
        - 'label' : label associé
        """

        # Étape 1 : Construire le graphe avec types
        data = self.prepare_graph_with_type()

        # Étape 2 : Charger le fichier Excel
        df = pd.read_excel(excel_path).dropna(subset=['term', 'label'])

        # Étape 3 : Encodage des labels
        self.label_encoder = LabelEncoder()
        df['encoded_label'] = self.label_encoder.fit_transform(df['label'])
        # Mapping GS index → Graph index
        df["graph_idx"] = df["term"].map(self.nodes_index)
        # Sauvegarder le mapping complet dans l'objet
        self.gs_to_graph = df["graph_idx"].tolist()
        self.graph_to_gs = {g: i for i, g in enumerate(self.gs_to_graph) if g != -1}
        # Initialisation du vecteur des labels
        num_nodes = data.num_nodes
        y = torch.full((num_nodes,), -1, dtype=torch.long)

        # Mapping node → index
        term_to_index = self.nodes_index

        # Assignation des labels aux nœuds
        for _, row in df.iterrows():
            term, label = row['term'], row['encoded_label']
            if term in term_to_index:
                idx = term_to_index[term]
                y[idx] = label
            else:
                print(f"[WARN] Term not found in graph: {term}")

        # Ajout au Data object
        data.y = y
        data.label_encoder = self.label_encoder
        data.num_classes = len(self.label_encoder.classes_)

        return data

    from sklearn.model_selection import train_test_split

    def split_data_with_labels(self, data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42,
                               stratify=True):
        """
        Split des nœuds annotés en train/val/test et ajoute les masques au Data object.

        Args:
            data (torch_geometric.data.Data): graphe avec `y` déjà rempli.
            train_ratio (float): proportion d'exemples en train.
            val_ratio (float): proportion d'exemples en validation.
            test_ratio (float): proportion d'exemples en test.
            random_state (int): seed pour reproductibilité.
            stratify (bool): stratifier par classe ou non.

        Returns:
            data (torch_geometric.data.Data): avec `train_mask`, `val_mask`, `test_mask`.
        """
        num_nodes = data.num_nodes
        annotated_idx = (data.y != -1).nonzero(as_tuple=True)[0].tolist()
        labels = data.y[annotated_idx].tolist()

        if stratify and len(set(labels)) > 1:
            stratify_labels = labels
        else:
            stratify_labels = None

        # Split train vs (val+test)
        idx_train, idx_temp, y_train, y_temp = train_test_split(
            annotated_idx, labels,
            train_size=train_ratio,
            random_state=random_state,
            stratify=stratify_labels
        )

        # Split val vs test
        relative_val = val_ratio / (val_ratio + test_ratio)
        if stratify and len(set(y_temp)) > 1:
            stratify_temp = y_temp
        else:
            stratify_temp = None
        idx_val, idx_test, y_val, y_test = train_test_split(
            idx_temp, y_temp,
            train_size=relative_val,
            random_state=random_state,
            stratify=stratify_temp
        )

        # Init masques
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        # Ajout au Data
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        print(
            f"[INFO] Split effectué → Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
        return data

    # def generate_split(self, data, seed=42, train_ratio=0.20, val_ratio=0.05, test_ratio=0.75, stratify=True):
    #     """
    #     Génère un seul split train/val/test sur les nœuds annotés.
    #
    #     Args:
    #         data (torch_geometric.data.Data): graphe avec `y` rempli.
    #         seed (int): seed de randomisation.
    #         train_ratio (float): proportion train.
    #         val_ratio (float): proportion validation.
    #         test_ratio (float): proportion test.
    #         stratify (bool): stratification des classes si possible.
    #
    #     Returns:
    #         dict: dictionnaire contenant train/val/test indices.
    #     """
    #     annotated_idx = (data.y != -1).nonzero(as_tuple=True)[0].tolist()
    #     labels = data.y[annotated_idx].tolist()
    #
    #     if stratify and len(set(labels)) > 1:
    #         stratify_labels = labels
    #     else:
    #         stratify_labels = None
    #
    #     # Split train vs (val+test)
    #     idx_train, idx_temp, y_train, y_temp = train_test_split(
    #         annotated_idx, labels,
    #         train_size=train_ratio,
    #         random_state=seed,
    #         stratify=stratify_labels
    #     )
    #
    #     # Split val vs test
    #     relative_val = val_ratio / (val_ratio + test_ratio)
    #     if stratify and len(set(y_temp)) > 1:
    #         stratify_temp = y_temp
    #     else:
    #         stratify_temp = None
    #
    #     idx_val, idx_test, y_val, y_test = train_test_split(
    #         idx_temp, y_temp,
    #         train_size=relative_val,
    #         random_state=seed,
    #         stratify=stratify_temp
    #     )
    #
    #     return {
    #         "train_idx": idx_train,
    #         "val_idx": idx_val,
    #         "test_idx": idx_test
    #     }

    def generate_split(self, data, seed=42,
                       train_ratio=0.20, val_ratio=0.05, test_ratio=0.75,
                       stratify=True):

        # Liste GS des termes annotés
        gs_indices = [i for i, gidx in enumerate(self.gs_to_graph) if gidx != -1]
        graph_indices = [self.gs_to_graph[i] for i in gs_indices]
        labels = data.y[graph_indices].tolist()

        if stratify and len(set(labels)) > 1:
            stratify_labels = labels
        else:
            stratify_labels = None

        # Split TRAIN
        gs_train, gs_temp, y_train, y_temp = train_test_split(
            gs_indices, labels,
            train_size=train_ratio,
            random_state=seed,
            stratify=stratify_labels
        )

        # Split VAL / TEST
        relative_val = val_ratio / (val_ratio + test_ratio)

        if stratify and len(set(y_temp)) > 1:
            stratify_temp = y_temp
        else:
            stratify_temp = None

        gs_val, gs_test, _, _ = train_test_split(
            gs_temp, y_temp,
            train_size=relative_val,
            random_state=seed,
            stratify=stratify_temp
        )

        return {
            "train_idx": gs_train,  # INDEX GS !!!
            "val_idx": gs_val,  # INDEX GS !!!
            "test_idx": gs_test  # INDEX GS !!!
        }

    def generate_and_save_splits(self, data, save_dir, n_splits=5, seeds=None,
                                 train_ratio=0.20, val_ratio=0.05, test_ratio=0.75, stratify=True):
        """
        Génère et sauvegarde plusieurs splits train/val/test.
        """
        if seeds is None:
            seeds = list(range(n_splits))

        os.makedirs(save_dir, exist_ok=True)
        splits_summary = {}

        for split_id, seed in enumerate(seeds):
            split_data = self.generate_split(
                data, seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                stratify=stratify
            )

            # Sauvegarde JSON
            with open(os.path.join(save_dir, f"split_{split_id}.json"), "w") as f:
                json.dump(split_data, f)

            # Sauvegarde NPY
            np.save(os.path.join(save_dir, f"train_idx_{split_id}.npy"), np.array(split_data["train_idx"]))
            np.save(os.path.join(save_dir, f"val_idx_{split_id}.npy"), np.array(split_data["val_idx"]))
            np.save(os.path.join(save_dir, f"test_idx_{split_id}.npy"), np.array(split_data["test_idx"]))

            splits_summary[f"split_{split_id}"] = {
                "seed": seed,
                "train_size": len(split_data["train_idx"]),
                "val_size": len(split_data["val_idx"]),
                "test_size": len(split_data["test_idx"])
            }

            print(f"[INFO] Split {split_id} (seed={seed}) → Train: {len(split_data['train_idx'])}, "
                  f"Val: {len(split_data['val_idx'])}, Test: {len(split_data['test_idx'])}")

        # Sauvegarde du résumé global
        with open(os.path.join(save_dir, "splits_summary.json"), "w") as f:
            json.dump(splits_summary, f, indent=2)

        print(f"[INFO] {n_splits} splits sauvegardés dans {save_dir}")


    # def load_split(self, data, split_path):
    #     """
    #     Recharge un split train/val/test depuis un fichier JSON et met à jour les masques dans data.
    #     """
    #
    #     with open(split_path, "r") as f:
    #         split_data = json.load(f)
    #
    #     num_nodes = data.num_nodes
    #     train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    #     val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    #     test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    #
    #     train_mask[split_data["train_idx"]] = True
    #     val_mask[split_data["val_idx"]] = True
    #     test_mask[split_data["test_idx"]] = True
    #
    #     data.train_mask = train_mask
    #     data.val_mask = val_mask
    #     data.test_mask = test_mask
    #
    #     print(f"[INFO] Split chargé depuis {split_path}")
    #     return data
    def load_split(self, data, split_path):

        with open(split_path, "r") as f:
            split_data = json.load(f)

        # Convertir GS → Graphe
        train_graph_idx = [self.gs_to_graph[i] for i in split_data["train_idx"]]
        val_graph_idx   = [self.gs_to_graph[i] for i in split_data["val_idx"]]
        test_graph_idx  = [self.gs_to_graph[i] for i in split_data["test_idx"]]

        num_nodes = data.num_nodes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_graph_idx] = True
        val_mask[val_graph_idx] = True
        test_mask[test_graph_idx] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        print(f"[INFO] Split chargé depuis {split_path}")
        return data