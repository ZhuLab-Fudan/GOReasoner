import json
import os
import argparse
from collections import defaultdict
from typing import Dict

class GOReasonerConsensus:
    """
    Consensus Refinement Module based on GOReasoner Equation (1).
    """
    
    def __init__(self, alpha: float = 1.0):
        # Alpha controls the relative contribution of GOExplainer [cite: 69]
        self.alpha = alpha 

    def load_init_results(self, file_path: str) -> Dict[str, Dict[str, float]]:
        """
        Load initial predictions (f_init) from sequence-based or text-based models[cite: 61, 69].
        Format: protein_id \t go_id \t score
        """
        preds = defaultdict(dict)
        if not os.path.exists(file_path):
            return preds
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    pro_id, go_id, score = parts
                    preds[pro_id][go_id] = float(score)
        return preds

    def load_refine_results(self, file_path: str) -> Dict[str, Dict[str, float]]:
        """
        Load refined predictions (f_exp) from GOExplainer[cite: 63, 69].
        Applies 0-1 normalization and filters by judgement.
        """
        preds = defaultdict(dict)
        if not os.path.exists(file_path):
            return preds
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                pro_id = item['proid']
                for p in item.get('preds', []):
                    # Score is positive only when the judgement is True [cite: 139]
                    if '\tNo' in p.get('answer', ''):
                        score = 0.0
                    else:
                        # Normalize JRJ scores from [-1, 1] to [0, 1] [cite: 140]
                        score = (p.get('DIG_score', 0.0) + 1) / 2
                    preds[pro_id][p['goid']] = score
        return preds

    def fuse(self, f_init_data: Dict, f_exp_data: Dict) -> Dict[str, Dict[str, float]]:
        """
        Perform Probabilistic Fusion (Noisy-OR).
        S(x) = 1 - (1 - f_init) * (1 - alpha * f_exp)
        """
        fused = defaultdict(dict)
        all_pros = set(f_init_data.keys()) | set(f_exp_data.keys())
        
        for pro_id in all_pros:
            go_terms = set(f_init_data.get(pro_id, {}).keys()) | set(f_exp_data.get(pro_id, {}).keys())
            
            for go_id in go_terms:
                f_init = f_init_data.get(pro_id, {}).get(go_id, 0.0)
                f_exp = f_exp_data.get(pro_id, {}).get(go_id, 0.0)
                
                # Equation (1) 
                combined_score = 1 - (1 - f_init) * (1 - self.alpha * f_exp)
                fused[pro_id][go_id] = combined_score
        return fused

    def save(self, results: Dict, output_path: str):
        with open(output_path, 'w') as f:
            for pro_id, gos in sorted(results.items()):
                for go_id, score in sorted(gos.items()):
                    f.write(f"{pro_id}\t{go_id}\t{score:.6f}\n")

def main():
    parser = argparse.ArgumentParser(description="GOReasoner Consensus Fusion")
    # Updated parameter names as requested
    parser.add_argument('--init_res', type=str, required=True, 
                        help='Directory containing initial result files (e.g., baseline_bp.txt)')
    parser.add_argument('--refine_res', type=str, required=True, 
                        help='Directory containing refinement result files (e.g., explainer_bp.json)')
    parser.add_argument('--alpha', type=float, default=1.0, 
                        help='Scaling factor (alpha) for GOExplainer contribution (default: 1.0) [cite: 69]')
    args = parser.parse_args()

    fusioner = GOReasonerConsensus(alpha=args.alpha)
    domains = ['bp', 'mf', 'cc']

    for domain in domains:
        print(f"Processing domain: {domain.upper()}...")
        
        # Construct paths using requested arguments
        init_file = os.path.join(args.init_res, f"baseline_{domain}.txt")
        refine_file = os.path.join(args.refine_res, f"explainer_{domain}.json")
        output_file = os.path.join(args.init_res, f"fused_{domain}.txt")

        # 1. Load data
        f_init_data = fusioner.load_init_results(init_file)
        f_exp_data = fusioner.load_refine_results(refine_file)

        # 2. Fuse
        if not f_init_data and not f_exp_data:
            print(f"  Warning: No data found for {domain}. Skipping.")
            continue
            
        fused_results = fusioner.fuse(f_init_data, f_exp_data)

        # 3. Save
        fusioner.save(fused_results, output_file)
        print(f"  Saved fused results to {output_file}")

if __name__ == "__main__":
    main()