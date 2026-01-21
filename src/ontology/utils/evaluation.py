#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2016/8/1
@author yrh

"""

import os
import sys
import configparser
import math
import random
import numpy as np
sys.path.append("/home/yanhy/")
sys.path.append("/home/yanhy/goatools")
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as area_under_curve
from statsmodels.stats.weightstats import ttost_paired
from multiprocessing import Pool
#from plot import plot
from utils import (EvalAnnotation, NS_ID,
                       GeneOntology,
                       get_protein_species,
                       get_type_list, get_label_list,
                       get_sample)


__all__ = ['get_term_scores',
           'get_y']


result_key_words = ('AUTHOR', 'MODEL', 'KEYWORDS', 'ACCURACY', 'END', 'mf', 'bp','cc','=====')
root_term = ('GO:0008150', 'GO:0003674', 'GO:0005575')


def get_term_scores(res_file, ontology):
    if res_file is None:
        return None
    term_scores = defaultdict(dict)
    for ns in NS_ID:
        term_scores[ns] = defaultdict(dict)
    with open(res_file) as fp:
        for line in fp:
            if line.startswith(result_key_words):
                continue
            line_list = line.split()
            pid, go_term, score, *_ = line_list
            if go_term in ontology:
                term_scores[ontology[go_term].ns][pid][go_term] = float(score)
                term_scores[pid][go_term] = float(score)
        return term_scores


def fscore(p, r):
    return 2*p*r/(p+r) if (p+r) > 0.0 else 0.0


def fmax(term_scores, pro_anno, type_list):
    fmax = 0.0, 1.0
    for cut in (c / 100 for c in range(101)):
        psum, pnum, rsum, rnum = 0.0, 0, 0.0, 0
        for seq in type_list:
            if seq in pro_anno and len(pro_anno[seq]) > 1:
                rnum += 1
                if seq in term_scores:
                    total, correct = 0, 0
                    for go_term in term_scores[seq]:
                        if go_term not in root_term and term_scores[seq][go_term] >= cut:
                            total += 1
                            if go_term in pro_anno[seq]:
                                correct += 1
                    rsum += correct / (len(pro_anno[seq]) - 1)
                    if total > 0:
                        pnum += 1
                        psum += correct / total
        if pnum > 0:
            # fmax = max(fmax, (fscore(psum / pnum, rsum / rnum), psum / pnum, rsum / rnum, cut))
            fmax = max(fmax, (fscore(psum / pnum, rsum / rnum), cut))
    return fmax

# def fmax(term_scores, pro_anno, type_list):
#     fmax = 0.0, 1.0
#     topk = []
#     f1 = []
#     precision = []
#     recall = []
#     for cut in (c for c in range(40)):
#     #for cut in (c / 100 for c in range(101)):
#         psum, pnum, rsum, rnum = 0.0, 0, 0.0, 0
#         for seq in type_list:
#             if seq in pro_anno and len(pro_anno[seq]) > 1:
#                 rnum += 1
#                 if seq in term_scores:
#                     total, correct = 0, 0
#                     for go_term in term_scores[seq]:
#                         sort_term_scores = list(term_scores[seq].values())
#                         top = min(len(sort_term_scores) - 1,cut)
#                         if go_term not in root_term and term_scores[seq][go_term] >= sort_term_scores[top]:
#                         #if go_term not in root_term and term_scores[seq][go_term] >= cut:
#                             total += 1
#                             if go_term in pro_anno[seq]:
#                                 correct += 1
#                     rsum += correct / (len(pro_anno[seq]) - 1)
#                     if total > 0:
#                         pnum += 1
#                         psum += correct / total
#         if pnum > 0:
#             # fmax = max(fmax, (fscore(psum / pnum, rsum / rnum), psum / pnum, rsum / rnum, cut))
            
#             print(f"top:{cut}|F1:{fscore(psum / pnum, rsum / rnum):.3f}|p:{psum / pnum:.3f}|r:{rsum / rnum:.3f}")
#             #print(f"cutoff:{cut:.2f}|F1:{fscore(psum / pnum, rsum / rnum):.3f}|p:{psum / pnum:.3f}|r:{rsum / rnum:.3f}")
#             topk.append(cut+1)
#             #topk.append(cut)
#             f1.append(round(fscore(psum / pnum, rsum / rnum),3))
#             precision.append(round(psum / pnum,3))
#             recall.append(round(rsum / rnum,3))
#             fmax = max(fmax, (fscore(psum / pnum, rsum / rnum), cut))
#     plot(topk,f1,precision,recall)
#     return fmax


def smin(term_scores, pro_anno, type_list, ic):
    if ic is None:
        return 0
    # func_scores = {seq: {func: func_scores[seq][func] for func in func_scores[seq] if func_scores[seq][func] >= 0.1}
    #                for seq in func_scores}
    smin = float('inf'), 1.0
    for cut in (c / 100 for c in range(101)):
        num, ru, mi = 0, 0.0, 0.0
        for seq in type_list:
            if seq in pro_anno and len(pro_anno[seq]) > 1:
                num += 1
                for func in pro_anno[seq]:
                    if seq not in term_scores or func not in term_scores[seq] or term_scores[seq][func] < cut:
                        ru += ic.get(func, 0)
                if seq in term_scores:
                    for func in term_scores[seq]:
                        if term_scores[seq][func] >= cut and func not in pro_anno[seq]:
                            mi += ic.get(func, 0)
        smin = min(smin, (math.sqrt(ru * ru + mi * mi) / num, cut))
    return smin[0]


def get_y(term_scores, pro_anno, type_list, label_list, top=100):
    # print(len(func_scores), len(seq_funcs), len(type_list), len(label_list))
    if top is not None:
        term_scores = {pid: {go_term: score for go_term, score in sorted(term_scores[pid].items(),
                                                                         key=lambda x: x[1], reverse=True)[:top]}
                       for pid in term_scores}
    y_true, y_pred = [], []
    for pid in type_list:
        if pid not in pro_anno or len(pro_anno[pid]) <= 1:
            continue
        for go_term in label_list:
            y_true.append(1.0 if go_term in pro_anno[pid] else 0.0)
            y_pred.append(term_scores[pid][go_term] if pid in term_scores and go_term in term_scores[pid] else 0.0)
    return y_true, y_pred


def fmax_and_smin_and_auc_and_aupr(term_scores, pro_anno, type_list, label_list, ic):
    try:
        y_true, y_pred = get_y(term_scores, pro_anno, type_list, label_list)
        """Some models like Naive and KNN may have many same scores, so we should random rank them."""
        y_true, y_pred = tuple(zip(*[(y, idx)
                                     for idx, (_, y) in enumerate(sorted(zip(y_pred, y_true), key=lambda x:x[0]))]))
        p, r, _ = precision_recall_curve(y_true, y_pred)
        return (fmax(term_scores, pro_anno, type_list),
                smin(term_scores, pro_anno, type_list, ic),
                roc_auc_score(y_true, y_pred),
                area_under_curve(r[1:], p[1:]))
    except:
        return 0, 0, 0, 0


def evaluate(term_scores, pro_anno, type_list, label_list, ic):
    """
    :param res_file: CAFA2 submission format file
    :param std_files: dict, keys are bp, cc or mf and values are std ontology files
    :return: fmax, auc, and aupr
    """
    return tuple(zip(*list(fmax_and_smin_and_auc_and_aupr(term_scores[ns], pro_anno[ns], type_list[ns],
                                                          label_list[ns], ic)
                           for ns in NS_ID)))


def bootstrap(term_scores, pro_anno, type_list, label_list, rep=10000):
    if isinstance(term_scores, dict):
        term_scores = (term_scores,)
    res, type_list = [], list(type_list)
    for _ in range(rep):
        sampling = [random.choice(type_list) for _ in range(len(type_list))]
        aupr = []
        for cate_term_scores in term_scores:
            y_true, y_pred = get_y(cate_term_scores, pro_anno, sampling, label_list)
            p, r, _ = precision_recall_curve(y_true, y_pred)
            aupr.append(area_under_curve(r, p))
        res.append(aupr)
        print(aupr)
    return list(zip(*res)) if len(term_scores) > 1 else list(zip(*res))[0]


def ttest(term_scores0, term_scores1, pro_anno, type_list, label_list, rep=100):
    res = []
    for ns in NS_ID:
        x0, x1 = bootstrap((term_scores0[ns], term_scores1[ns]), pro_anno[ns], type_list[ns], label_list[ns], rep)
        pvalue, _, _ = ttost_paired(np.asarray(x0), np.asarray(x1), 0, float('inf'))
        res.append(pvalue)
    return res


def term_aupr(term_scores, pro_anno, type_list, label_list, res_file=None):
    if res_file is not None:
        res_file = open(res_file, 'w')
    res = []
    for ns in NS_ID:
        auprs = []
        for label in label_list[ns]:
            y_true, y_pred = get_y(term_scores[ns], pro_anno[ns], type_list[ns], {label}, None)
            if sum(y_true) >= 3:
                aupr = 0
                if len(y_pred) - y_pred.count(0) >= 3:
                    p, r, _ = precision_recall_curve(y_true, y_pred)
                    aupr = area_under_curve(r, p)
                auprs.append(aupr)
                # auprs.append(roc_auc_score(y_true, y_pred))
                # print(label, aupr)
                if res_file is not None:
                    print(label, aupr, file=res_file)
        res.append(sum(auprs) / len(auprs) if len(auprs) > 0 else 0.0)
    if res_file is not None:
        res_file.close()
    return res


def get_seq_aupr(term_scores, pro_anno, type_list, label_list):
    auprs = []
    for ns in NS_ID:
        aupr = 0
        for seq in type_list[ns]:
            if seq in term_scores[ns]:
                y_true, y_pred = get_y({seq: term_scores[ns][seq]}, pro_anno[ns], [seq], label_list[ns])
                if sum(y_true) == 0: continue
                p, r, _ = precision_recall_curve(y_true, y_pred)
                if len(p) > 2:
                    aupr += area_under_curve(r[1:], p[1:])
        auprs.append(aupr / len(type_list[ns]))
    return auprs


def main(argv):
    if len(argv) == 0:
        print('please input one or more configure files')
        print('annotations.bp annotations.cc annotations.mf ', 'the annotations of bp, cc and mf')
        print('type_file.bp type_file.cc type_file.mf       ', 'the evaluated sequences of bp, cc and mf')
        print('output.results                               ', 'the results file')

    conf = configparser.ConfigParser()
    print(conf.read(argv))
    ontology = GeneOntology(conf.get('annotations', 'ontology'))
    pro_anno = {ns: Annotation.load(conf.get('annotations', ns), ontology) for ns in NS_ID}
    type_list = {ns: get_type_list(conf.get('type_file', ns, fallback=None)) for ns in NS_ID}
    label_list = get_label_list(conf.get('label', 'label_list'), ontology)
    if conf.has_option('annotations', 'ic'):
        ic = {}
        with open(conf.get('annotations', 'ic')) as fp:
            for line in fp:
                go_term, ic_score = line.split()
                ic[go_term] = float(ic_score)
    else:
        ic = None
    print(conf.get('output', 'top-results', fallback=None))
    term_scores = get_term_scores(conf.get('output', 'top-results', fallback=None), ontology)

    if conf.has_section('t-test'):
        # print(conf.get('t-test', 'results'), conf.get('output', 'top-results'))
        # func_scores0 = get_func_scores(conf.get('t-test', 'results'), ontology)
        # print(ttest(func_scores0, term_scores, pro_anno, type_list, label_list))
        times = conf.getint('t-test', 'times')
        type_list = {ns: get_sample(conf.get('t-test', ns), times) for ns in NS_ID}
        print(conf.get('output', 'sample'))
        with open(conf.get('output', 'sample'), 'w') as fp:
            for i in range(times):
                fmax, smin, auc, aupr = evaluate(term_scores, pro_anno,
                                                 {ns: type_list[ns][i] for ns in NS_ID}, label_list, ic)
                print(*(f[0] for f in fmax), *smin, *auc, *aupr)
                print(*(f[0] for f in fmax), *smin, *auc, *aupr, file=fp)
    elif conf.getboolean('label', 'term-centric', fallback=False):
        print('term-centric', conf.get('label', 'label_list'), conf.get('output', 'results'))
        term_scores = get_term_scores(conf.get('output', 'results'), ontology)
        print(term_aupr(term_scores, pro_anno, type_list, label_list, conf.get('output', 'aupr', fallback=None)))
    elif conf.has_option('species', 'specific'):
        # print('species:', conf.get('species', 'specific'))
        protein_species = get_protein_species(conf.get('species', 'protein_species'))
        for sp in conf.get('species', 'specific').split():
            t_list = defaultdict(set)
            for ns in NS_ID:
                for seq in type_list[ns]:
                    if protein_species[seq] == sp:
                        t_list[ns].add(seq)
            print(sp + ':', evaluate(term_scores, pro_anno, t_list, label_list, ic))
    elif conf.has_option('evaluation', 'results'):
        print(conf.get('evaluation', 'results').split())
        for res_file in conf.get('evaluation', 'results').split():
            term_scores = get_term_scores(res_file, ontology)
            print(os.path.splitext(os.path.basename(res_file))[0],
                  evaluate(term_scores, pro_anno, type_list, label_list, ic))
    else:
        print(evaluate(term_scores, pro_anno, type_list, label_list, ic))


if __name__ == '__main__':
    main(sys.argv[1:])

