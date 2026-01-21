"""
GO term hierarchical propagation module.
Implements score propagation through the GO DAG structure.
"""

from typing import Dict, List, Set, Tuple
from collections import deque


class GOPropagator:
    """
    Handle GO term score propagation through ontology hierarchy.
    """
    
    def __init__(self, ontology, root_terms: Tuple[str, ...] = ('GO:0008150', 'GO:0003674', 'GO:0005575')):
        """
        Initialize GO propagator.
        
        Args:
            ontology: GeneOntology object
            root_terms: Root GO terms to exclude
        """
        self.ontology = ontology
        self.root_terms = set(root_terms)
    
    def find_leaf_nodes(self, terms: List[str]) -> List[str]:
        """
        Identify leaf nodes in given GO term set.
        
        Args:
            terms: List of GO term IDs
            
        Returns:
            List of leaf node GO term IDs
        """
        term_set = set(terms)
        leaf_nodes = []
        
        for term in terms:
            if term not in self.ontology:
                continue
                
            # Check if term has no children or all children are outside term_set
            if not self.ontology[term].children:
                leaf_nodes.append(term)
            else:
                has_child_in_set = any(
                    child in term_set 
                    for child in self.ontology[term].children
                )
                if not has_child_in_set:
                    leaf_nodes.append(term)
        
        return leaf_nodes
    
    def transfer_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Transfer scores through GO hierarchy using negative propagation.
        
        Args:
            scores: Dictionary mapping GO term IDs to scores
            
        Returns:
            Updated scores after propagation
        """
        return self.ontology.transfer_scores_with_negatives(scores)
    
    def propagate_scores(self,
                        protein_id: str,
                        statement: str,
                        domain: str,
                        go_term_list: List[str],
                        scorer,
                        go_definitions: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Hierarchical score propagation through GO DAG.
        
        Algorithm:
        1. Process leaf nodes iteratively
        2. Calculate scores for leaf nodes
        3. Propagate positive scores to ancestors
        4. Recalculate low-scoring propagated nodes
        
        Args:
            protein_id: Protein identifier
            statement: Protein description text
            domain: GO domain (mf/bp/cc)
            go_term_list: List of candidate GO terms
            scorer: GOScorer instance for calculating term scores
            go_definitions: Mapping of GO IDs to definitions
            
        Returns:
            Tuple of (score_dict, reason_dict)
        """
        score_dict = {}
        reason_dict = {}
        score_to_goid = {}
        calculated_terms = set()
        
        # Filter valid GO terms
        remaining_terms = [
            goid for goid in go_term_list 
            if go_definitions.get(goid)
        ]
        
        # Step 1-3: Iteratively process leaf nodes
        while remaining_terms:
            # 1) Get leaf nodes
            leaves = self.find_leaf_nodes(remaining_terms)
            if not leaves:
                break
            
            # 2) Calculate scores for leaf nodes
            scores_to_propagate = {}
            _scores, _reasons = scorer.get_score(
                protein_id, statement, domain, leaves, go_definitions
            )
            
            for leaf in leaves:
                score = _scores[leaf]
                if score > 0:
                    scores_to_propagate[leaf] = score
                
                calculated_terms.add(leaf)
                score_dict[leaf] = score
                reason_dict[leaf] = _reasons[leaf]
                score_to_goid[score] = leaf
            
            # 3) Propagate scores
            propagated = {}
            if scores_to_propagate:
                propagated = self.transfer_scores(scores_to_propagate)
                
                for term, score in propagated.items():
                    if term not in score_dict:
                        score_dict[term] = score
                    else:
                        score_dict[term] = max(score_dict[term], score)
                    
                    # Update reason
                    if score_dict[term] == float('-inf'):
                        reason_dict[term] = 'False\tError\n'
                    else:
                        reference = reason_dict[score_to_goid[score_dict[term]]]
                        reason = ' '.join(reference.split()[2:])
                        judgement = score_dict[term] > 0
                        
                        if score_to_goid[score_dict[term]] != term:
                            reason_dict[term] = (
                                f'{term}\t{judgement}\t'
                                f'[Transfer from {score_to_goid[score_dict[term]]}] {reason}'
                            )
            
            remaining_terms = [
                term for term in remaining_terms 
                if term not in (set(propagated.keys()) | calculated_terms)
            ]
        
        # Step 4: Recalculate low-scoring propagated terms
        positive_scores = [s for s in score_dict.values() if s > 0]
        
        if not positive_scores:
            average_score = sum(score_dict.values()) / len(score_dict)
            terms_to_recalculate = [
                term for term, score in score_dict.items()
                if term not in calculated_terms and score < average_score
            ]
        else:
            average_score = sum(positive_scores) / len(positive_scores)
            terms_to_recalculate = [
                term for term, score in score_dict.items()
                if term not in calculated_terms and score < average_score
            ]
        
        if terms_to_recalculate:
            _score, _reasons = scorer.get_score(
                protein_id, statement, domain, 
                terms_to_recalculate, go_definitions
            )
            
            for term in terms_to_recalculate:
                new_score = _score[term]
                score_dict[term] = new_score
                reason_dict[term] = _reasons[term]
                calculated_terms.add(term)
                score_to_goid[new_score] = term
        
        print(f"{len(go_term_list)} => {len(calculated_terms)}")
        
        return score_dict, reason_dict