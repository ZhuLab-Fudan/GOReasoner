#!/usr/bin/env python3
# -*- coding: utf-8
import click
from collections import defaultdict
import sys
from utils import GeneOntology


__all__ = ['get_label_list']


def get_label_list(term_file, ontology):
    if term_file is None:
        return None
    label_list = defaultdict(list)
    with open(term_file) as fp:
        for line in fp:
            go_term = line.strip()
            if go_term in ontology and go_term not in ontology.root_term:
                label_list[ontology[go_term].ns].append(go_term)
    return label_list



@click.command()
@click.option('--annotation-file')
@click.option('--ontology-file')
@click.option('--label-file')
def main(annotation_file, ontology_file, label_file):
    label_list = set()
    with open(annotation_file) as fp:
        for line in fp:
            label_list.add(line.split()[1])
    ontology = GeneOntology(ontology_file)
    with open(label_file, 'w') as fp:
        for go_term in ontology.transfer(label_list):
            print(go_term, file=fp)


if __name__ == '__main__':
    main()
