"""
GOReasoner: Main inference pipeline for protein GO term prediction.
Integrates GOExplainer model with hierarchical propagation.
"""

import argparse
import json
import os
import sys
from typing import Dict, List
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.GOExplainer import GOExplainer
from src.propagation import GOPropagator
from ontology.utils import GeneOntology


class GOReasoner:
    """
    Main reasoning pipeline for GO term prediction.
    """
    
    def __init__(self,
                 model_path: str,
                 ontology_path: str,
                 go_definitions_path: str,
                 lora_path: str = None):
        """
        Initialize GOReasoner pipeline.
        
        Args:
            model_path: Path to LLM model
            ontology_path: Path to GO ontology file
            go_definitions_path: Path to GO definitions
            lora_path: Optional LoRA adapter path
        """
        # Initialize GO ontology
        self.ontology = GeneOntology(ontology_path)
        
        # Load GO definitions
        self.go_definitions = np.load(
            go_definitions_path, allow_pickle=True
        ).item()
        
        # Initialize GOExplainer model
        self.model = GOExplainer(
            model_path=model_path,
            lora_path=lora_path
        )
        
        # Initialize propagator
        self.propagator = GOPropagator(self.ontology)
        
        # Statistics
        self.total_terms = 0
        self.calculated_terms = 0
    
    def predict(self,
               input_file: str,
               output_file: str,
               domain: str,
               use_propagation: bool = True):
        """
        Run prediction pipeline on input data.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output results
            domain: GO domain (mf/bp/cc)
            use_propagation: Whether to use hierarchical propagation
        """
        # Load cache if exists
        cache_file = output_file.replace(".txt", ".npy")
        if os.path.exists(cache_file):
            self.model.load_cache(cache_file)
        
        results = []
        total_tp, total_fp, total_fn = 0, 0, 0
        
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            with open(output_file, 'w') as wf:
                for item in tqdm(data, desc="Processing proteins"):
                    # Extract protein information
                    protein_id = item['proid']
                    statement = item.get('desc', '')
                    
                    # Extract GO candidates and ground truth
                    if item.get('go'):
                        if item['go'][0].get('goid'):
                            go_list = [_.get('goid') for _ in item['go']]
                            true_go = [
                                _.get('goid') for _ in item['go'] 
                                if _.get('label') == True
                            ]
                        else:
                            go_list = [list(_.keys())[0] for _ in item['go']]
                            true_go = [
                                list(_.keys())[0] for _ in item['go'] 
                                if list(_.values())[0] == True
                            ]
                    elif item.get('goids'):
                        go_list = list(set(item['goids']))
                        true_go = go_list
                    else:
                        continue
                    
                    # Skip short statements
                    if len(statement) < 10:
                        print(f"Skipping {protein_id}: statement too short")
                        continue
                    
                    # Filter valid GO terms
                    go_list = [
                        goid for goid in go_list 
                        if self.go_definitions.get(goid)
                    ]
                    
                    # Get scores with/without propagation
                    if use_propagation:
                        score_dict, reason_dict = self.propagator.propagate_scores(
                            protein_id=protein_id,
                            statement=statement,
                            domain=domain,
                            go_term_list=go_list,
                            scorer=self.model,
                            go_definitions=self.go_definitions
                        )
                    else:
                        score_dict, reason_dict = self.model.get_score(
                            protein_id=protein_id,
                            statement=statement,
                            domain=domain,
                            go_list=go_list,
                            go_definitions=self.go_definitions
                        )
                    
                    # Process predictions
                    predictions = []
                    predicted_positive = []
                    
                    for goid in score_dict:
                        # Write to output file
                        wf.write(f"{protein_id}\t{goid}\t{score_dict[goid]}\n")
                        
                        predictions.append({
                            'goid': goid,
                            'answer': reason_dict.get(goid, f'{goid}\tFalse\tError'),
                            'label': goid in true_go,
                            'score': score_dict[goid]
                        })
                        
                        if score_dict[goid] > 0:
                            predicted_positive.append(goid)
                    
                    # Sort by score
                    predictions = sorted(
                        predictions, 
                        key=lambda x: x['score'], 
                        reverse=True
                    )
                    
                    results.append({
                        'proid': protein_id,
                        'predictions': predictions
                    })
                    
                    # Calculate metrics
                    metrics = self._compute_metrics(true_go, predicted_positive)
                    total_tp += metrics['tp']
                    total_fp += metrics['fp']
                    total_fn += metrics['fn']
                    
                    print(f"{protein_id}: {metrics}")
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            self.model.save_cache(cache_file)
            raise
        
        # Save results
        self._save_results(results, output_file)
        
        # Calculate and print overall metrics
        self._print_overall_metrics(total_tp, total_fp, total_fn)
        
        # Save cache
        self.model.save_cache(cache_file)
    
    def _compute_metrics(self, 
                        golden: List[str], 
                        predicted: List[str]) -> Dict:
        """
        Compute precision, recall, F1 metrics.
        
        Args:
            golden: Ground truth GO terms
            predicted: Predicted GO terms
            
        Returns:
            Dictionary of metrics
        """
        golden_set = set(golden)
        predicted_set = set(predicted)
        
        tp = len(golden_set & predicted_set)
        fp = len(predicted_set - golden_set)
        fn = len(golden_set - predicted_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall) 
              if (precision + recall) > 0 else 0.0)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    def _save_results(self, results: List[Dict], output_file: str):
        """Save results to JSON file."""
        json_file = output_file.replace(".txt", ".json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {json_file}")
    
    def _print_overall_metrics(self, tp: int, fp: int, fn: int):
        """Print overall evaluation metrics."""
        micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / 
                   (micro_precision + micro_recall) 
                   if (micro_precision + micro_recall) > 0 else 0.0)
        
        print("\n" + "="*50)
        print("Overall Metrics:")
        print(f"Micro-Precision: {micro_precision:.4f}")
        print(f"Micro-Recall: {micro_recall:.4f}")
        print(f"Micro-F1: {micro_f1:.4f}")
        print("="*50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GOReasoner: GO term prediction with LLM"
    )
    parser.add_argument('--model_path', type=str, required=True, default="whitneyyan0122/GOExplainer-Qwen3-8B",
                       help='Path to base LLM model')
    parser.add_argument('--lora_path', type=str, default=None,
                       help='Path to LoRA adapter')
    parser.add_argument('--ontology_path', type=str, 
                       default='src/ontology/go-basic.obo',
                       help='Path to GO ontology file')
    parser.add_argument('--go_definitions', type=str,
                       default='dependency_file/go.npy',
                       help='Path to GO definitions file')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input JSON file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to output results file')
    parser.add_argument('--domain', type=str, choices=['mf', 'bp', 'cc'],
                       default='mf', help='GO domain')
    parser.add_argument('--no_propagation', action='store_true', default=False,
                       help='Disable hierarchical propagation')
    
    args = parser.parse_args()
    
    # Initialize reasoner
    reasoner = GOReasoner(
        model_path=args.model_path,
        ontology_path=args.ontology_path,
        go_definitions_path=args.go_definitions,
        lora_path=args.lora_path
    )
    
    # Run prediction
    reasoner.predict(
        input_file=args.input_file,
        output_file=args.output_file,
        domain=args.domain,
        use_propagation=not args.no_propagation
    )


if __name__ == "__main__":
    main()