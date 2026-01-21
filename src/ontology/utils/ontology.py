#!/usr/bin/env python3
# -*- coding: utf-8

from collections import defaultdict
import sys
sys.path.append("../goatools")
from goatools.obo_parser import GODag
from utils import NS_ID_FC2S
from typing import List, Set, Dict, Tuple

__all__ = ['GeneOntology']


class GOTerm(object):
    """

    """

    def __init__(self, go_term):
        """
        :param go_term: instance of the GOTerm
        :param godag: instance of the GODag
        :return:
        """
        self.id = go_term.id
        self.parents = {p.id for p in go_term.parents}
        if hasattr(go_term, 'relationship'):
            for parent in go_term.relationship.get('part_of', set()):
                if parent.namespace == go_term.namespace:
                    self.parents.add(parent.id)
        self.name = go_term.name
        self.namespace = self.ns = NS_ID_FC2S[go_term.namespace]
        self.children = set()
        self.depth = 0


class GeneOntology(dict):
    """

    """

    def __init__(self, obo_file_path):
        """
        :param obo_file_path: the ontology obo file
        :return:
        """
        super(GeneOntology, self).__init__()
        go_dag = GODag(obo_file_path, 'relationship')
        for go_id, go_term in go_dag.items():
            self[go_id] = GOTerm(go_term)
        self.get_children()
        self.root_term = ['GO:0008150', 'GO:0003674', 'GO:0005575']
        self.get_depth()

    def transfer(self, go_list):
        """
        :param go_list: the go terms which should be transferred
        :return:
        """
        go_list = list(filter(lambda x: x in self, go_list))
        ancestors, now = set(go_list), set(go_list)
        while len(now) > 0:
            next = set()
            for go_term in now:
                if go_term in self:
                    next |= self[go_term].parents - ancestors
            now = next
            ancestors |= now
        return ancestors

    def get_children(self):
        for go_id in self:
            for parent in self[go_id].parents:
                self[parent].children.add(go_id)

    def get_depth(self):
            # 初始化所有节点的深度为0
            for term in self:
                self[term].depth = 0
            
            # 根节点深度为1
            for root in self.root_term:
                self[root].depth = 1
            
            # 使用队列进行BFS
            from collections import deque
            queue = deque(self.root_term)
            
            while queue:
                current = queue.popleft()
                
                for child in self[current].children:
                    # 更新子节点的深度：取当前深度和父节点深度+1中的较大值
                    new_depth = self[current].depth + 1
                    if new_depth > self[child].depth:
                        self[child].depth = new_depth
                    
                    # 将子节点加入队列继续处理
                    queue.append(child)
    
    # def get_depth(self):
    #     for root in self.root_term:
    #         self[root].depth = 1
    #     now = set(self.root_term)
    #     while len(now) > 0:
    #         next = set()
    #         for go_term in now:
    #             for child in self[go_term].children:
    #                 if self[child].depth == 0:
    #                     next.add(child)
    #                     self[child].depth = self[go_term].depth + 1
    #         now = next


    def get_sibling_terms(self, go_id: str) -> Set[str]:
        """
        Get sibling terms for a given GO term.
        
        :param go_id: The GO term ID.
        :return: Set of sibling GO term IDs.
        """
        if go_id not in self:
            return set()
        siblings = set()
        for parent_id in self[go_id].parents:
            siblings.update(self[parent_id].children)
        siblings.discard(go_id)  # remove the term itself from its siblings
        return siblings

    def generate_negative_samples(self, input_dict: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, Set[str]]]:
        """
        Generate negative samples for each protein in the input dictionary.

        :param input_dict: Dictionary of proteins and their GO terms by namespace.
        :return: Dictionary of proteins and their negative GO term samples by namespace.
        """
        negative_samples = defaultdict(set)
        # for ns, proteins in input_dict.items():
        for protein, go_terms in input_dict.items():
            for go_id in go_terms:
                siblings = self.get_sibling_terms(go_id)
                # Exclude the positive samples from negative samples
                siblings.difference_update(go_terms)
                #negative_samples[ns][protein].update(siblings)
                negative_samples[protein].update(siblings)
        return negative_samples
    
    def transfer_scores(self, term_scores):
        scores = defaultdict(float)
        for go_term in sorted(self.transfer(term_scores.keys()), key=lambda x: self[x].depth, reverse=True):
            scores[go_term] = max(scores[go_term], term_scores.get(go_term, 0))
            for parent_id in self[go_term].parents:
                scores[parent_id] = max(scores[parent_id], scores[go_term])
        return scores
    
    # def transfer_scores(self, term_scores):
    #     scores = defaultdict(float)
    #     for go_term in term_scores:
    #         for parent_id in self[go_term].parents:
    #             scores[parent_id] = max(scores[parent_id], scores[go_term], term_scores.get(parent_id, 0), term_scores.get(go_term, 0))            
    #     for go_term in sorted(self.transfer(term_scores.keys()), key=lambda x: self[x].depth, reverse=True):
    #         scores[go_term] = max(scores[go_term], term_scores.get(go_term, 0))
    #         for parent_id in self[go_term].parents:
    #             scores[parent_id] = max(scores[parent_id], scores[go_term], term_scores.get(parent_id, 0), term_scores.get(go_term, 0))
    #     return scores
    
    def transfer_scores_with_negatives(self, term_scores):
        scores = {}
        for go_term in term_scores:
            for parent_id in self[go_term].parents:
                scores[parent_id] = max(scores.get(parent_id, -float('inf')) , scores.get(parent_id, -float('inf')), term_scores.get(parent_id, -float('inf')), term_scores.get(go_term, -float('inf')))            

        for go_term in sorted(self.transfer(term_scores.keys()), key=lambda x: self[x].depth, reverse=True):
            if scores.get(go_term) is None:
                scores[go_term] = term_scores.get(go_term, -float('inf'))
            else:
                scores[go_term] = max(scores[go_term], term_scores.get(go_term, -float('inf')))
            for parent_id in self[go_term].parents:
                if scores.get(parent_id) is None:
                    scores[parent_id] = scores[go_term]
                else:
                    scores[parent_id] = max(scores[parent_id], scores[go_term], term_scores.get(parent_id, -float('inf')), term_scores.get(go_term, -float('inf')))            
        return scores


if __name__ == '__main__':
    # ontology = GeneOntology('go-basic.obo')
    # print(ontology['GO:0090645'].id)
    ontology = GeneOntology('/home/zhushanfeng/liuhc/CAFA5/go-basic.obo')
    input_dict = {
        # "MF": {"protein1": ["GO:0044877", "GO:0045735"]},
        # "BP": {"protein2": ["GO:0010431", "GO:0009792"]},
        "CC": {"protein3": ["GO:0005938", "GO:0005737", "GO:0005634"]}
    }
    negative_samples = ontology.generate_negative_samples(input_dict)
    for ns, proteins in negative_samples.items():
        for protein, samples in proteins.items():
            print(f"{ns} - {protein}: {samples}")
            print(len(samples))
    for ns in input_dict:
        for protein in input_dict[ns]:
            pos_func = ontology.transfer(input_dict[ns][protein])
            print(f"{ns} - {protein} pos_func: {pos_func}")