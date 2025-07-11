import torch
import os
import joblib
import warnings
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_scatter import scatter_mean, scatter_sum, scatter_max
from encoder.gvp import AutoGraphEncoder
from utils.data_utils import convert_graph, BatchSampler, extract_seq_from_pdb
from build_graph import generate_graph
from build_subgraph import generate_pos_subgraph
from pathlib import Path
import gc
import signal
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings("ignore")

# Set up timeout handling
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def safe_parallel_map(func, data, workers=2, use_threads=True, timeout=300):
    """Safe parallel mapping to avoid hanging"""
    results = []
    
    if len(data) == 1 or workers == 1:
        # Single-threaded processing to avoid parallel issues
        for item in tqdm(data, desc="Sequential processing", disable=True):
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                print(f"Error processing {item}: {e}")
                results.append(None)
        return results
    
    # Use thread pool instead of process pool, safer
    if use_threads:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(func, item): item for item in data}
            for future in tqdm(as_completed(futures, timeout=timeout), total=len(data), desc="Parallel processing", disable=True):
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except Exception as e:
                    item = futures[future]
                    print(f"Error processing {item}: {e}")
                    results.append(None)
    else:
        # Sequential processing as fallback
        for item in tqdm(data, desc="Sequential fallback", disable=True):
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                print(f"Error processing {item}: {e}")
                results.append(None)
    
    return results


class SimpleDataset(Dataset):
    """Simple dataset class to avoid complex batch processing"""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def predict_sturcture(model, cluster_models, dataloader, device):
    """Predict structure with better error handling"""
    struc_label_dict = {}
    cluster_model_dict = {}

    for cluster_model_path in cluster_models:
        cluster_model_name = cluster_model_path.split("/")[-1].split(".")[0]
        struc_label_dict[cluster_model_name] = []
        cluster_model_dict[cluster_model_name] = joblib.load(cluster_model_path)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Predicting structures", disable=True)):
            try:
                batch.to(device)
                h_V = (batch.node_s, batch.node_v)
                h_E = (batch.edge_s, batch.edge_v)

                node_emebddings = model.get_embedding(h_V, batch.edge_index, h_E)
                graph_emebddings = scatter_mean(node_emebddings, batch.batch, dim=0).cpu()
                norm_graph_emebddings = F.normalize(graph_emebddings, p=2, dim=1)
                
                for name, cluster_model in cluster_model_dict.items():
                    batch_structure_labels = cluster_model.predict(
                        norm_graph_emebddings
                    ).tolist()
                    struc_label_dict[name].extend(batch_structure_labels)
                
                # Timely cleanup
                del batch, h_V, h_E, node_emebddings, graph_emebddings, norm_graph_emebddings
                if torch.cuda.is_available() and i % 5 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                # Add empty results to avoid length mismatch
                for name in cluster_model_dict.keys():
                    struc_label_dict[name].extend([0] * batch.num_graphs)
                continue

    return struc_label_dict


def process_single_pdb_safe(pdb_file, subgraph_depth, max_distance, num_threads=1):
    """Safely process single PDB file to avoid parallel issues"""
    result_dict = {}
    result_dict["name"] = os.path.basename(pdb_file)
    
    try:
        # Build graph
        graph = generate_graph(pdb_file, max_distance)
        result_dict["aa_seq"] = graph.aa_seq
        anchor_nodes = list(range(0, len(graph.node_s), 1))
        
        subgraphs = []
        
        # Process subgraphs serially to avoid threading issues
        for anchor_node in tqdm(anchor_nodes, desc=f"Processing {os.path.basename(pdb_file)}", leave=False, disable=True):
            try:
                subgraph = generate_pos_subgraph(
                    graph,
                    subgraph_depth,
                    max_distance,
                    anchor_node,
                    verbose=False,
                    pure_subgraph=True,
                )[anchor_node]
                subgraph = convert_graph(subgraph)
                subgraphs.append(subgraph)
            except Exception as e:
                print(f"Error processing anchor {anchor_node}: {e}")
                # Add an empty subgraph as placeholder
                subgraphs.append(None)
        
        # Filter out None values
        subgraphs = [sg for sg in subgraphs if sg is not None]
        return subgraphs, result_dict, len(subgraphs)
        
    except Exception as e:
        result_dict["error"] = str(e)
        print(f"Error processing {pdb_file}: {e}")
        return None, result_dict, 0


def simple_pdb_converter(pdb_files, subgraph_depth, max_distance, max_batch_nodes, error_file=None):
    """Simplified PDB converter to avoid complex parallel processing"""
    all_subgraphs = []
    results = []
    node_counts = []
    errors = []
    
    # Process each PDB file serially
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files", disable=True):
        subgraphs, result_dict, node_count = process_single_pdb_safe(
            pdb_file, subgraph_depth, max_distance
        )
        
        if subgraphs is None:
            errors.append(result_dict)
            continue
            
        all_subgraphs.extend(subgraphs)
        results.append(result_dict)
        node_counts.append(node_count)
    
    # Save errors
    if errors and error_file:
        error_df = pd.DataFrame(errors)
        error_df.to_csv(error_file, index=False)
    
    def simple_collate_fn(batch):
        """Simplified collate function"""
        batch_graphs = Batch.from_data_list(batch)
        batch_graphs.node_s = torch.zeros_like(batch_graphs.node_s)
        return batch_graphs
    
    # Create simple dataset
    dataset = SimpleDataset(all_subgraphs)
    
    # Use simple batching
    batch_size = max(1, max_batch_nodes // 1000)  # Estimate batch size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Do not use multiprocessing
        collate_fn=simple_collate_fn
    )
    
    return dataloader, results


class SSTPredictor:
    def __init__(
        self,
        model_path=None,
        cluster_dir=None,
        cluster_model=None,
        max_distance=10,
        subgraph_depth=None,
        max_batch_nodes=10000,
        num_processes=1,  # Default single process
        num_threads=1,   # Default single thread
        device=None,
        structure_vocab_size=2048,
        safe_mode=True,  # Add safe mode
    ) -> None:
        """Initialize the SST predictor.
        
        Args:
            safe_mode: If True, use safer but potentially slower processing
        """
        assert structure_vocab_size in [20, 64, 128, 512, 1024, 2048, 4096]
        
        if model_path is None:
            self.model_path = str(Path(__file__).parent / "static" / "AE.pt")
        else:
            self.model_path = model_path
            
        if cluster_dir is None:
            self.cluster_dir = str(Path(__file__).parent / "static")
            self.cluster_model = [f"{structure_vocab_size}.joblib"]
        else:
            self.cluster_dir = cluster_dir
            self.cluster_model = cluster_model if cluster_model is not None else [f"{structure_vocab_size}.joblib"]
            
        self.max_distance = max_distance
        self.subgraph_depth = subgraph_depth
        self.max_batch_nodes = max_batch_nodes
        self.num_processes = 1 if safe_mode else min(num_processes, 2)  # Limit number of processes
        self.num_threads = 1 if safe_mode else min(num_threads, 4)    # Limit number of threads
        self.structure_vocab_size = structure_vocab_size
        self.safe_mode = safe_mode
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Reset CUDA state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Load model
        node_dim = (256, 32)
        edge_dim = (64, 2)
        model = AutoGraphEncoder(
            node_in_dim=(20, 3),
            node_h_dim=node_dim,
            edge_in_dim=(32, 1),
            edge_h_dim=edge_dim,
            num_layers=6,
        )
        
        try:
            if self.device == "cpu":
                model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            else:
                model.load_state_dict(torch.load(self.model_path))
                model.to(self.device)
            model.eval()
            self.model = model
            
            params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self.cluster_models = [os.path.join(self.cluster_dir, m) for m in self.cluster_model]
        
        # Verify cluster models exist
        for cm in self.cluster_models:
            if not os.path.exists(cm):
                print(f"Warning: Cluster model {cm} not found")

    def clear_all_cache(self):
        """Thoroughly clear all caches"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(0.1)  # Give system a moment

    def predict_from_pdb(self, pdb_files, error_file=None, cache_subgraph_dir=None):
        """Safe PDB prediction method"""
        try:
            if isinstance(pdb_files, str):
                pdb_files = [pdb_files]
            
            # Check file existence
            valid_files = []
            for pdb_file in pdb_files:
                if os.path.exists(pdb_file):
                    valid_files.append(pdb_file)
                else:
                    print(f"Warning: File {pdb_file} not found")
            
            if not valid_files:
                return []
            
            # Use safe converter
            if self.safe_mode:
                data_loader, results = simple_pdb_converter(
                    valid_files,
                    self.subgraph_depth,
                    self.max_distance,
                    self.max_batch_nodes,
                    error_file
                )
            else:
                # Original method but with reduced parallelism
                from . import pdb_conventer
                data_loader, results = pdb_conventer(
                    valid_files,
                    self.subgraph_depth,
                    self.max_distance,
                    self.max_batch_nodes,
                    error_file,
                    self.num_processes,
                    self.num_threads,
                    cache_subgraph_dir
                )
            
            if not results:
                return []
            
            # Predict structures
            structures = predict_sturcture(self.model, self.cluster_models, data_loader, self.device)
            
            # Assemble results
            start, end = 0, 0
            for result in results:
                seq_len = len(result["aa_seq"])
                end += seq_len
                for cluster_name, structure_labels in structures.items():
                    if end <= len(structure_labels):
                        result[f"{cluster_name}_sst_seq"] = structure_labels[start:end]
                    else:
                        print(f"Warning: Structure labels length mismatch for {result['name']}")
                        result[f"{cluster_name}_sst_seq"] = structure_labels[start:] + [0] * (end - len(structure_labels))
                start = end
            
            return results
            
        except Exception as e:
            print(f"Error in predict_from_pdb: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            # Thorough cleanup
            self.clear_all_cache()

    def __del__(self):
        """Destructor"""
        try:
            self.clear_all_cache()
        except:
            pass