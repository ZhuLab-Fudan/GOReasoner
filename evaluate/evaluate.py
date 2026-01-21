#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2022/10/22
@author wangsj
weighted F-measure

"""

import os
import sys
import configparser
import math
import random
import numpy as np
# sys.path.append("/home/zhushanfeng/liuhc/NG/goatools/")
# sys.path.append("/home/zhushanfeng/liuhc/NG")
sys.path.append("/home/yanhy")
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as area_under_curve
from statsmodels.stats.weightstats import ttost_paired
from multiprocessing import Pool
from collections import deque
from tqdm import tqdm
# from plot import plot
from utils import (Annotation, NS_ID,
                       GeneOntology,
                       get_protein_species,
                       get_type_list, get_label_list,
                       get_sample)


__all__ = ['get_term_scores',
           'get_y']


result_key_words = ('AUTHOR', 'MODEL', 'KEYWORDS', 'ACCURACY', 'END')
root_term = ('GO:0008150', 'GO:0003674', 'GO:0005575')

def filter_root(ontology, term_scores, root='GO:0005488'):
    descendants = set()
    queue = deque([root])  
    while queue:
        current = queue.popleft()  
        if current not in descendants:  
            descendants.add(current)
            queue.extend(child for child in ontology[current].children if child in term_scores)
    filtered_term_scores = {k: v for k, v in term_scores.items() if k not in descendants}
    return filtered_term_scores

def get_term_scores(res_file, ontology):
    if res_file is None:
        return None
    term_scores = defaultdict(dict)
    for ns in NS_ID:
        term_scores[ns] = defaultdict(dict)
    with open(res_file) as fp:
        for line in fp:
            try:
                if line.startswith(result_key_words):
                    continue
                line_list = line.split()
                pid, go_term, score, *_ = line_list
                if go_term in ontology:
                    term_scores[ontology[go_term].ns][pid][go_term] = float(score)
                    # term_scores[pid][go_term] = float(score)
            except Exception:
                print(Exception)
                pass
        for ns in ['mf', 'bp', 'cc']:
            for pid in term_scores[ns]:
                # if len(term_scores[ns][pid]) > 50:
                #     term_scores[ns][pid] = dict(list(term_scores[ns][pid].items())[:50])
                # if len(term_scores[ns][pid]) > 50:
                #     term_scores[ns][pid] = dict(
                #         sorted(term_scores[ns][pid].items(), key=lambda item: item[1], reverse=True)
                #     )
                term_scores[ns][pid] = ontology.transfer_scores_with_negatives(term_scores[ns][pid])
                # if ns == 'mf':
                #     term_scores[ns][pid] = filter_root(ontology, term_scores[ns][pid])
        return term_scores


def fscore(p, r):
    return 2*p*r/(p+r) if (p+r) > 0.0 else 0.0

import numpy as np

def fmax(term_scores, pro_anno, type_list, ic, n_cuts=500):
    fmax = (0.0, 0.0)
    best_prec, best_recall = 0.0, 0.0  # Variables to store precision and recall at max F1

    all_scores = []
    for seq in type_list:
        if seq in term_scores:
            all_scores.extend(term_scores[seq].values())

    if len(all_scores) == 0:
        return fmax

    all_scores = np.array(all_scores)


    qs = np.linspace(0, 1, n_cuts + 1)
    cuts = np.quantile(all_scores, qs)
    cuts = np.unique(cuts)   

    
    for cut in cuts:
        psum, pnum, rsum, rnum = 0.0, 0, 0.0, 0
        predicted_num = 0

        for seq in type_list:
            if seq in pro_anno and len(pro_anno[seq]) > 1:
                rnum += 1
                if seq in term_scores:
                    predicted_num += 1
                    total, correct = 0.0, 0.0

                    for go_term, score in term_scores[seq].items():
                        if go_term not in root_term and score >= cut:
                            w = ic.get(go_term, 0.0)
                            total += w
                            if go_term in pro_anno[seq]:
                                correct += w

                    weighted_true_go = 0.0
                    for go_term in pro_anno[seq]:
                        if go_term not in root_term:
                            weighted_true_go += ic.get(go_term, 0.0)

                    if weighted_true_go > 0:
                        rsum += correct / weighted_true_go

                    if total > 0:
                        pnum += 1
                        psum += correct / total

        if pnum > 0 and rnum > 0:
            prec = psum / pnum
            recall = rsum / rnum
            f1 = fscore(prec, recall)
            if f1 > fmax[0]:
                fmax = (f1, cut)
                best_prec, best_recall = prec, recall  # Update best precision and recall

    print("rnum:", rnum, "predicted_num:", predicted_num)
    print("Best F1:", fmax[0], "Precision:", best_prec, "Recall:", best_recall)  # Print the values
    return fmax



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
        return (fmax(term_scores, pro_anno, type_list, ic),
                # 0, 0, 0
                smin(term_scores, pro_anno, type_list, ic),
                roc_auc_score(y_true, y_pred),
                area_under_curve(r[1:], p[1:])
                )
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
                aupr = 0.0
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


def get_seq_aupr(term_scores, pro_anno, type_list, label_list, res_file=None):
    auprs = []
    for ns in NS_ID:
        if res_file is not None:
            res_file_in = open(res_file+f'.{ns}', 'w')
        aupr = 0
        # res_file_inn = open(res_file+f'.ttm{ns}', 'w')
        for seq in type_list[ns]:
            aupr_per = 0
            if seq in term_scores[ns]:
                # print(seq, file=res_file_inn)
                y_true, y_pred = get_y({seq: term_scores[ns][seq]}, pro_anno[ns], [seq], label_list[ns])
                if sum(y_true) == 0: continue
                p, r, _ = precision_recall_curve(y_true, y_pred)
                if len(p) > 2:
                    aupr_per = area_under_curve(r[1:], p[1:])
                    aupr += area_under_curve(r[1:], p[1:])
            if res_file is not None:
                print(seq, aupr_per, file=res_file_in)
        auprs.append(aupr / len(type_list[ns]))
        if res_file is not None:
            res_file_in.close()
    return auprs


def _main(argv):
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
                go_term, ic_score = line.strip().split()
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
        # print('term-centric', conf.get('label', 'label_list'), conf.get('output', 'results'))
        # term_scores = get_term_scores(conf.get('output', 'results'), ontology)
        # print('term-centric', conf.get('label', 'label_list'), conf.get('evaluation', 'results'))
        term_scores = get_term_scores(conf.get('evaluation', 'results'), ontology)
        #print(term_aupr(term_scores, pro_anno, type_list, label_list, conf.get('output', 'aupr', fallback=None)))
        print(get_seq_aupr(term_scores, pro_anno, type_list, label_list, conf.get('output', 'aupr', fallback=None)))
    elif conf.has_option('species', 'specific'):
        # print('species:', conf.get('species', 'specific'))
        protein_species = get_protein_species(conf.get('species', 'protein_species'))
        for sp in conf.get('species', 'specific').split():
            t_list = defaultdict(set)
            for ns in NS_ID:
                for seq in type_list[ns]:
                    if protein_species[seq] == sp:
                        t_list[ns].add(seq)
            term_scores = get_term_scores(conf.get('evaluation', 'results'), ontology)
            print(f"{sp}=mf:{len(t_list['mf'])}bp:{len(t_list['bp'])}cc:{len(t_list['cc'])}")
            if len(t_list['mf'])+len(t_list['bp'])+len(t_list['cc'])>0:
                print(sp + ':', evaluate(term_scores, pro_anno, t_list, label_list, ic))
    elif conf.has_option('evaluation', 'results'):
        print(conf.get('evaluation', 'results').split())
        for res_file in conf.get('evaluation', 'results').split():
            term_scores = get_term_scores(res_file, ontology)
            print(os.path.splitext(os.path.basename(res_file))[0],
                  evaluate(term_scores, pro_anno, type_list, label_list, ic))
    else:
        print(evaluate(term_scores, pro_anno, type_list, label_list, ic))

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
                go_term, ic_score = line.strip().split()
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
        # print('term-centric', conf.get('label', 'label_list'), conf.get('output', 'results'))
        # term_scores = get_term_scores(conf.get('output', 'results'), ontology)
        # print('term-centric', conf.get('label', 'label_list'), conf.get('evaluation', 'results'))
        term_scores = get_term_scores(conf.get('evaluation', 'results'), ontology)
        #print(term_aupr(term_scores, pro_anno, type_list, label_list, conf.get('output', 'aupr', fallback=None)))
        print(get_seq_aupr(term_scores, pro_anno, type_list, label_list, conf.get('output', 'aupr', fallback=None)))
    elif conf.has_option('species', 'specific'):
        # print('species:', conf.get('species', 'specific'))
        protein_species = get_protein_species(conf.get('species', 'protein_species'))
        for sp in conf.get('species', 'specific').split():
            t_list = defaultdict(set)
            for ns in NS_ID:
                for seq in type_list[ns]:
                    if protein_species[seq] == sp:
                        t_list[ns].add(seq)
            term_scores = get_term_scores(conf.get('evaluation', 'results'), ontology)
            print(f"{sp}=mf:{len(t_list['mf'])}bp:{len(t_list['bp'])}cc:{len(t_list['cc'])}")
            if len(t_list['mf'])+len(t_list['bp'])+len(t_list['cc'])>0:
                print(sp + ':', evaluate(term_scores, pro_anno, t_list, label_list, ic))
    elif conf.has_option('evaluation', 'results'):
        print(conf.get('evaluation', 'results').split())
        for res_file in conf.get('evaluation', 'results').split():
            term_scores = get_term_scores(res_file, ontology)
            fmax_res, smin, auc, aupr = evaluate(term_scores, pro_anno, type_list, label_list, ic)
            print(os.path.splitext(os.path.basename(res_file))[0],
                  (fmax_res, smin, auc, aupr))
            # write thresholded predictions if output dir configured
            if conf.has_option('output', 'dir'):
                out_dir = conf.get('output', 'dir')
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, os.path.basename(res_file))
                # map ns -> cut (threshold from fmax)
                ns_to_cut = {ns: fmax_res[i][1] for i, ns in enumerate(NS_ID)}
                with open(out_path, 'w') as wf:
                    # optional header with thresholds
                    print('# thresholds\t' + '\t'.join(f'{ns}={ns_to_cut[ns]:.6f}' for ns in NS_ID), file=wf)
                    for ns in NS_ID:
                        cut = ns_to_cut[ns]
                        # only consider sequences in type_list for consistency with evaluation
                        valid_pids = set(type_list[ns]) if isinstance(type_list[ns], (set, list, tuple)) else set(type_list[ns])
                        for pid, gos in term_scores[ns].items():
                            if pid not in valid_pids:
                                continue
                            for go_term, score in gos.items():
                                if go_term in root_term:
                                    continue
                                exceed = 1 if score >= cut else 0
                                annotated = (pid in pro_anno[ns] and go_term in pro_anno[ns][pid])
                                correct = 'T' if (exceed == 1 and annotated) or (exceed == 0 and not annotated) else 'F'
                                print(ns, exceed, correct, pid, go_term, score, sep='\t', file=wf)
                                
    else:
        eval_res = evaluate(term_scores, pro_anno, type_list, label_list, ic)
        print(eval_res)
        # write thresholded predictions for top-results if configured
        if conf.has_option('output', 'dir') and conf.has_option('output', 'top-results'):
            res_file = conf.get('output', 'top-results')
            if res_file:
                fmax_res, smin, auc, aupr = eval_res
                out_dir = conf.get('output', 'dir')
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, os.path.basename(res_file))
                ns_to_cut = {ns: fmax_res[i][1] for i, ns in enumerate(NS_ID)}
                with open(out_path, 'w') as wf:
                    print('# thresholds\t' + '\t'.join(f'{ns}={ns_to_cut[ns]:.6f}' for ns in NS_ID), file=wf)
                    for ns in NS_ID:
                        cut = ns_to_cut[ns]
                        valid_pids = set(type_list[ns]) if isinstance(type_list[ns], (set, list, tuple)) else set(type_list[ns])
                        for pid, gos in term_scores[ns].items():
                            if pid not in valid_pids:
                                continue
                            for go_term, score in gos.items():
                                if go_term in root_term:
                                    continue
                                exceed = 1 if score >= cut else 0
                                correct = 'T' if (pid in pro_anno[ns] and go_term in pro_anno[ns][pid]) else 'F'
                                print(ns, pid, go_term, exceed, correct, sep='\t', file=wf)

if __name__ == '__main__':
    main(sys.argv[1:])
