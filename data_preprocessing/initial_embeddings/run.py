# Entiy Embedding 
import sys
sys.path.append('../../utilities')
import utilities as u
from GraphBERTEmbedder import GraphBERTEmbedder
import BertEmbedder as bem
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    import torch

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(cuda_available)

    parser = argparse.ArgumentParser(description="Triplets Elements Embedding: Embedding of subject and predicate and object ")
    parser.add_argument("--input_kg", type=str, help="name of the input json file inside the data directory (without extensions)")
    parser.add_argument("--model", type=str, help="pretrained_model_name_or_path")
    args = parser.parse_args()
    kg_name = args.input_kg
    output_path = "../outputs"
    KG_path = "../../data/" + kg_name +".json"
    model_name = args.model
    gbe = GraphBERTEmbedder(KG_path, output_path, model_name)
    gbe.run()