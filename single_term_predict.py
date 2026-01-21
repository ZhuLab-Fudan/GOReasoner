"""
Single GO term prediction script.
Predict whether a specific GO term applies to a protein based on literature evidence.

Usage:
    python single_term_predict.py \
        --protein_name "P12345" \
        --pmids "12345678,23456789" \
        --goid "GO:0003674" \
        --domain "mf"
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.GOExplainer.model import GOExplainer
from src.data_processor import ProteinDataProcessor


class SingleTermPredictor:
    """
    Predict a single GO term for a given protein.
    """
    
    def __init__(self,
                 model_path: str,
                 ontology_path: str,
                 go_definitions_path: str,
                 lora_path: Optional[str] = None):
        """
        Initialize single term predictor.
        
        Args:
            model_path: Path to base LLM model
            ontology_path: Path to GO ontology file
            go_definitions_path: Path to GO definitions
            lora_path: Optional LoRA adapter path
        """
        # Load GO definitions
        self.go_definitions = np.load(
            go_definitions_path, 
            allow_pickle=True
        ).item()
        
        # Initialize GOExplainer
        self.model = GOExplainer(
            model_path=model_path,
            lora_path=lora_path
        )
        
        # Initialize data processor
        self.data_processor = ProteinDataProcessor()
        
        print("Single term predictor initialized successfully!")
    
    def fetch_literature(self, 
                        pmids: List[str]) -> str:
        """
        Fetch literature abstracts from PubMed IDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            Combined literature text
        """
        # TODO: Implement PubMed fetching via Entrez API
        # For now, return placeholder
        print(f"Fetching literature for PMIDs: {pmids}")
        
        try:
            from Bio import Entrez
            Entrez.email = "your_email@example.com"  # Required by NCBI
            
            abstracts = []
            for pmid in pmids:
                try:
                    handle = Entrez.efetch(
                        db="pubmed",
                        id=pmid,
                        rettype="abstract",
                        retmode="text"
                    )
                    abstract = handle.read()
                    abstracts.append(abstract)
                    handle.close()
                except Exception as e:
                    print(f"Failed to fetch PMID {pmid}: {e}")
            
            return "\n\n".join(abstracts)
        
        except ImportError:
            print("Warning: Biopython not installed. Using placeholder text.")
            return (
                "Placeholder literature text. "
                "Install Biopython to fetch actual abstracts: "
                "pip install biopython"
            )
    
    def predict_single_term(self,
                           protein_name: str,
                           literature: str,
                           goid: str,
                           domain: str,
                           pmids: Optional[List[str]] = None) -> dict:
        """
        Predict whether a GO term applies to a protein.
        
        Args:
            protein_name: Protein identifier or name
            literature: Literature text describing the protein
            goid: GO term ID to verify
            domain: GO domain (mf/bp/cc)
            pmids: Optional list of PubMed IDs for reference
            
        Returns:
            Dictionary containing prediction results
        """
        # Validate GO term
        if goid not in self.go_definitions:
            return {
                "error": f"GO term {goid} not found in definitions",
                "goid": goid,
                "prediction": None
            }
        
        # Prepare literature chunks if text is long
        if len(literature) > 1000:
            chunks_data = self.data_processor.get_chunks(
                text=literature,
                protein_id=protein_name
            )
            # Use first chunk for simplicity (could aggregate later)
            statement = chunks_data[0]['text']
        else:
            statement = literature
        
        print(f"\n{'='*60}")
        print(f"Predicting GO term for protein: {protein_name}")
        print(f"GO Term: {goid} - {self.go_definitions[goid]}")
        print(f"Domain: {domain}")
        if pmids:
            print(f"PMIDs: {', '.join(pmids)}")
        print(f"{'='*60}\n")
        
        # Get prediction
        scores, reasons = self.model.get_score(
            protein_id=protein_name,
            statement=statement,
            domain=domain,
            go_list=[goid],
            go_definitions=self.go_definitions,
            batch_size=1
        )
        
        # Parse results
        score = scores.get(goid, 0.0)
        reason = reasons.get(goid, "No reason available")
        
        # Determine prediction
        is_positive = score > 0
        confidence = abs(score)
        
        # Parse reasoning
        try:
            parts = reason.split('\t')
            if len(parts) >= 3:
                go_id_parsed = parts[0]
                judgement = parts[1]
                explanation = '\t'.join(parts[2:])
            else:
                go_id_parsed = goid
                judgement = "True" if is_positive else "False"
                explanation = reason
        except Exception:
            go_id_parsed = goid
            judgement = "True" if is_positive else "False"
            explanation = reason
        
        result = {
            "protein_name": protein_name,
            "goid": goid,
            "go_definition": self.go_definitions[goid],
            "domain": domain,
            "prediction": judgement,
            "confidence_score": confidence,
            "raw_score": score,
            "explanation": explanation,
            "pmids": pmids or [],
            "literature_length": len(literature)
        }
        
        return result
    
    def predict_from_pmids(self,
                          protein_name: str,
                          pmids: List[str],
                          goid: str,
                          domain: str) -> dict:
        """
        Predict GO term by fetching literature from PubMed.
        
        Args:
            protein_name: Protein identifier
            pmids: List of PubMed IDs
            goid: GO term ID
            domain: GO domain
            
        Returns:
            Prediction results
        """
        # Fetch literature
        literature = self.fetch_literature(pmids)
        
        if not literature or len(literature) < 50:
            return {
                "error": "Failed to fetch sufficient literature",
                "protein_name": protein_name,
                "goid": goid,
                "pmids": pmids
            }
        
        # Make prediction
        return self.predict_single_term(
            protein_name=protein_name,
            literature=literature,
            goid=goid,
            domain=domain,
            pmids=pmids
        )
    
    def predict_batch_terms(self,
                           protein_name: str,
                           literature: str,
                           goids: List[str],
                           domain: str) -> List[dict]:
        """
        Predict multiple GO terms for one protein.
        
        Args:
            protein_name: Protein identifier
            literature: Literature text
            goids: List of GO term IDs
            domain: GO domain
            
        Returns:
            List of prediction results
        """
        # Filter valid GO terms
        valid_goids = [
            goid for goid in goids 
            if goid in self.go_definitions
        ]
        
        if not valid_goids:
            return [{
                "error": "No valid GO terms found",
                "provided_goids": goids
            }]
        
        print(f"\nPredicting {len(valid_goids)} GO terms for {protein_name}...")
        
        # Get predictions
        scores, reasons = self.model.get_score(
            protein_id=protein_name,
            statement=literature,
            domain=domain,
            go_list=valid_goids,
            go_definitions=self.go_definitions
        )
        
        # Format results
        results = []
        for goid in valid_goids:
            score = scores.get(goid, 0.0)
            reason = reasons.get(goid, "No reason available")
            
            is_positive = score > 0
            confidence = abs(score)
            
            # Parse reasoning
            try:
                parts = reason.split('\t')
                if len(parts) >= 3:
                    judgement = parts[1]
                    explanation = '\t'.join(parts[2:])
                else:
                    judgement = "True" if is_positive else "False"
                    explanation = reason
            except Exception:
                judgement = "True" if is_positive else "False"
                explanation = reason
            
            results.append({
                "goid": goid,
                "go_definition": self.go_definitions[goid],
                "prediction": judgement,
                "confidence_score": confidence,
                "raw_score": score,
                "explanation": explanation
            })
        
        # Sort by confidence
        results = sorted(results, key=lambda x: x['confidence_score'], reverse=True)
        
        return results


def print_result(result: dict):
    """Pretty print prediction result."""
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Protein: {result['protein_name']}")
    print(f"GO Term: {result['goid']}")
    print(f"Definition: {result['go_definition']}")
    print(f"Domain: {result['domain']}")
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence_score']:.4f}")
    print(f"Raw Score: {result['raw_score']:.4f}")
    
    if result.get('pmids'):
        print(f"PMIDs: {', '.join(result['pmids'])}")
    
    print(f"\nExplanation:")
    print(result['explanation'])
    print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Predict single GO term for a protein"
    )
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to base LLM model')
    parser.add_argument('--lora_path', type=str, default=None,
                       help='Path to LoRA adapter')
    parser.add_argument('--ontology_path', type=str,
                       default='./data/go-basic.obo',
                       help='Path to GO ontology file')
    parser.add_argument('--go_definitions', type=str,
                       default='./data/go_definitions.npy',
                       help='Path to GO definitions file')
    
    # Prediction arguments
    parser.add_argument('--protein_name', type=str, required=True,
                       help='Protein identifier or name')
    parser.add_argument('--goid', type=str, required=True,
                       help='GO term ID (e.g., GO:0003674)')
    parser.add_argument('--domain', type=str, required=True,
                       choices=['mf', 'bp', 'cc'],
                       help='GO domain')
    
    # Literature input (one of these is required)
    parser.add_argument('--pmids', type=str, default=None,
                       help='Comma-separated PubMed IDs (e.g., 12345678,23456789)')
    parser.add_argument('--literature', type=str, default=None,
                       help='Direct literature text')
    parser.add_argument('--literature_file', type=str, default=None,
                       help='Path to file containing literature text')
    
    # Optional: batch prediction
    parser.add_argument('--goids_file', type=str, default=None,
                       help='Path to file with multiple GO IDs (one per line)')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results as JSON')
    
    args = parser.parse_args()
    
    # Validate literature input
    if not any([args.pmids, args.literature, args.literature_file]):
        parser.error("One of --pmids, --literature, or --literature_file is required")
    
    # Initialize predictor
    print("Initializing predictor...")
    predictor = SingleTermPredictor(
        model_path=args.model_path,
        ontology_path=args.ontology_path,
        go_definitions_path=args.go_definitions,
        lora_path=args.lora_path
    )
    
    # Get literature
    if args.literature:
        literature = args.literature
        pmids = None
    elif args.literature_file:
        with open(args.literature_file, 'r') as f:
            literature = f.read()
        pmids = None
    else:  # args.pmids
        pmids = args.pmids.split(',')
        result = predictor.predict_from_pmids(
            protein_name=args.protein_name,
            pmids=pmids,
            goid=args.goid,
            domain=args.domain
        )
        print_result(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        
        return
    
    # Batch prediction mode
    if args.goids_file:
        with open(args.goids_file, 'r') as f:
            goids = [line.strip() for line in f if line.strip()]
        
        results = predictor.predict_batch_terms(
            protein_name=args.protein_name,
            literature=literature,
            goids=goids,
            domain=args.domain
        )
        
        print(f"\n{'='*60}")
        print(f"BATCH PREDICTION RESULTS ({len(results)} terms)")
        print(f"{'='*60}\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['goid']}: {result['prediction']} "
                  f"(confidence: {result['confidence_score']:.4f})")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    "protein_name": args.protein_name,
                    "domain": args.domain,
                    "predictions": results
                }, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    # Single prediction mode
    else:
        result = predictor.predict_single_term(
            protein_name=args.protein_name,
            literature=literature,
            goid=args.goid,
            domain=args.domain,
            pmids=pmids
        )
        
        print_result(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()