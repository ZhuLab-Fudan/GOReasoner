#!/usr/bin/env python3
# -*- coding: utf-8
import os
import sys
import datetime
import configparser
from Bio import SeqIO
from collections import defaultdict
import json
import requests
import aiohttp
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
__all__ = ['EvalAnnotation',
           'Annotation',
           'parse_fasta',
           'write_fasta',
           'get_data',
           'get_target_data',
           'generate_target_data',
           'output_res',
           'split_annotations_by_namespace',
           'get_protein_species',
           'fetch_protein_species',
           'get_type_list',
           'get_blast_identity',
           'get_now',
           'NS_ID',
           'NS_ID_FC2S',
           'get_term_scores',
           'get_pmid_consensus_score',
           'log']


NS_ID = ['mf', 'bp', 'cc']
NS_ID_FC2S = {'molecular_function': 'mf', 'biological_process': 'bp', 'cellular_component': 'cc',
              'F': 'mf', 'P': 'bp', 'C': 'cc'}
result_key_words = ('AUTHOR', 'MODEL', 'KEYWORDS', 'ACCURACY', 'END')
root_term = ('GO:0008150', 'GO:0003674', 'GO:0005575')


# class Annotation(defaultdict):
#     """
#     Protein Annotations.
#     """

#     def __init__(self):
#         super(Annotation, self).__init__(set)

#     def combine(self, *args):
#         """
#         :param args: one or more ProteinAnno
#         :return: combined ProteinAnno
#         """
#         for pro_anno in args:
#             for name in pro_anno:
#                 self[name] = self[name] | pro_anno[name]
#         return self

#     @staticmethod
#     def load(anno_files=None, ontology=None):
#         """
#         :param anno_files: the BP, MF, CC functions files, the file have two columns: ID  GO
#         :param ontology: if it is not None, the GO will propagate with it
#         :return: instance of ProteinAnno
#         """
#         if anno_files is None:
#             return None
#         if isinstance(anno_files, str):
#             anno_files = [anno_files]
#         pro_anno = Annotation()
#         for funcs_file in anno_files:
#             try:
#                 with open(funcs_file) as fp:
#                     for line in fp:
#                         pid, go_term = line.split()[:2]
#                         pro_anno[pid].add(go_term)
#             except Exception:
#                 pass
#         if ontology is not None:
#             for pid in pro_anno:
#                 pro_anno[pid] = ontology.transfer(pro_anno[pid])
#         else:
#             print('Warning! Don\'t transfer the annotations!')
#         return pro_anno

class Annotation(defaultdict):
    """
    Protein Annotations.
    """
    root = ('GO:0008150', 'GO:0003674', 'GO:0005575')
    def __init__(self):
        super(Annotation, self).__init__(set)

    def combine(self, *args):
        """
        :param args: one or more ProteinAnno
        :return: combined ProteinAnno
        """
        for pro_anno in args:
            for name in pro_anno:
                self[name] = self[name] | pro_anno[name]
        return self

    @staticmethod
    def load(anno_files=None, ontology=None):
        """
        :param anno_files: the BP, MF, CC functions files, the file have two columns: ID  GO
        :param ontology: if it is not None, the GO will propagate with it
        :return: instance of ProteinAnno
        """
        if anno_files is None:
            return None
        if isinstance(anno_files, str):
            anno_files = [anno_files]
        pro_anno = Annotation()
        for funcs_file in anno_files:
            try:
                with open(funcs_file) as fp:
                    for line in fp:
                        pid, go_term = line.split()[:2]
                        pro_anno[pid].add(go_term)
            except Exception:
                pass
        if ontology is not None:
            for pid in pro_anno:
                pro_anno[pid] = ontology.transfer(pro_anno[pid]) - set(Annotation.root)
                # pro_anno[pid] = ontology.transfer(pro_anno[pid])
        else:
            print('Warning! Don\'t transfer the annotations!')
        return pro_anno

class EvalAnnotation(defaultdict):
    """
    Protein Annotations.
    """

    root = ('GO:0008150', 'GO:0003674', 'GO:0005575')
    def __init__(self):
        super(EvalAnnotation, self).__init__(set)

    def combine(self, *args):
        """
        :param args: one or more ProteinAnno
        :return: combined ProteinAnno
        """
        for pro_anno in args:
            for name in pro_anno:
                self[name] = self[name] | pro_anno[name]
        return self

    @staticmethod
    def load(anno_files=None, ontology=None):
        """
        :param anno_files: the BP, MF, CC functions files, the file have two columns: ID  GO
        :param ontology: if it is not None, the GO will propagate with it
        :return: instance of ProteinAnno
        """
        if anno_files is None:
            return None
        if isinstance(anno_files, str):
            anno_files = [anno_files]
        pro_anno = EvalAnnotation()
        for funcs_file in anno_files:
            try:
                with open(funcs_file) as fp:
                    for line in fp:
                        pid, go_term = line.split()[:2]
                        pro_anno[pid].add(go_term)
            except Exception:
                pass
        if ontology is not None:
            for pid in pro_anno:
                pro_anno[pid] = ontology.transfer(pro_anno[pid]) - set(EvalAnnotation.root)
        else:
            print('Warning! Don\'t transfer the annotations!')
        return pro_anno

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
                    term_scores[pid][go_term] = float(score)
            except Exception:
                pass
        return term_scores


def parse_fasta(fasta_file):
    return SeqIO.parse(fasta_file, 'fasta')


def write_fasta(seqs, fasta_file):
    SeqIO.write(seqs, fasta_file, 'fasta')


def get_data(train_seq_file, train_anno_file=None, ontology=None):
    return list(parse_fasta(train_seq_file)), Annotation.load(train_anno_file, ontology)


def get_target_data(targets_file, targets_dir):
    if not os.path.exists(targets_file):
        seqs = generate_target_data(targets_dir)
        write_fasta(seqs, targets_file)
    return parse_fasta(targets_file)


def generate_target_data(targets_dir):
    seqs = set()
    for dirpath, dirnames, filenames in os.walk(targets_dir):
        for file in filenames:
            if file.endswith('.tfa'):
                seqs |= set(parse_fasta(os.path.join(dirpath, file)))
        for dname in dirnames:
            seqs |= generate_target_data(os.path.join(dirpath, dname))
    return seqs


def output_res(res_file, ontology, pred, author='FDUPFP', model_id=1, keywords='machine learning'):
    if res_file is None: return
    with open(res_file, 'w') as fp:
        # print('AUTHOR', author, file=fp)
        # print('MODEL', model_id, file=fp)
        # print('KEYWORDS', keywords + '.', file=fp)
        for seq in pred:
            #scores = ontology.transfer_scores(pred[seq.id])
            for func, score in sorted(pred[seq].items(), key=lambda x: x[1], reverse=True)[:200]:
                print(seq, func, round(score, 3), ontology[func].namespace, ontology[func].name, sep='\t', file=fp)
        # print('END', file=fp)


def split_annotations_by_namespace(data_file, bp_file, cc_file, mf_file):
    annotations = {}
    with open(data_file) as fp:
        for line in fp:
            pid, acc, np = line.strip().split()[:3]
            if pid not in annotations:
                annotations[pid] = {'bp': set(), 'cc': set(), 'mf': set()}
            annotations[pid][np].add(acc)
    with open(bp_file, 'w') as fbp, open(cc_file, 'w') as fcc, open(mf_file, 'w') as fmf:
        fp = {'bp': fbp, 'cc': fcc, 'mf': fmf}
        for pid in annotations:
            for np in annotations[pid]:
                if annotations[pid][np] == {'GO:0005515'}:
                    continue
                for acc in annotations[pid][np]:
                    print(pid, acc, file=fp[np])


def get_protein_species(protein_species_file):
    protein_species = {}
    with open(protein_species_file) as fp:
        for line in fp:
            pid, sp = line.split()[:2]
            protein_species[pid] = sp
    return protein_species


def get_type_list(type_file):
    if type_file is None:
        return None
    type_list = set()
    with open(type_file) as fp:
        for line in fp:
            type_list.add(line.strip())
    return type_list

# def fetch_protein_species(protein_ids):
#     """
#     Fetches species IDs for a list of protein UniProt IDs and returns them as a dictionary.

#     Args:
#     - protein_ids (list): A list of UniProt IDs.

#     Returns:
#     - dict: A dictionary with protein IDs as keys and species IDs as values.
#     """
#     species_dict = {}
#     for protein_id in protein_ids:
#         url = f"https://rest.uniprot.org/uniprotkb/{protein_id.id}.json"
#         response = requests.get(url, proxies={"http": None, "https": None})
        
#         if response.status_code == 200:
#             data = response.json()
#             species_name = data['organism']['taxonId']
#             species_dict[protein_id.id] = species_name
#             print(species_name)
#         else:
#             print('Warning! Species id not recognised!')
#             species_dict[protein_id.id] = '0000'

#     return species_dict

# 缓存机制，避免重复查询
@lru_cache(maxsize=1000)
def cache_species_lookup(protein_id):
    return None  # 初始值为None，未缓存任何数据

async def fetch_species_async(protein_id, session):
    """
    异步请求单个protein_id的species信息
    """
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
    try:
        async with session.get(url, proxy=None) as response:  # proxy=None 禁用代理
            if response.status == 200:
                data = await response.json()
                species_name = data.get('organism', {}).get('taxonId', 'Unknown')
                return species_name
            else:
                species_name = '0000'
                print(f"Warning! Failed to fetch {protein_id}: HTTP {response.status}")
                return species_name
    except Exception as e:
        species_name = '0000'
        print(f"Error fetching {protein_id}: {e}")
        return species_name

async def fetch_species(protein_records, rate_limit=10):
    """
    异步批量获取多个protein的species信息，同时控制速率以避免被封。

    Args:
    - protein_records (list of SeqRecord): SeqRecord对象列表。
    - rate_limit (int): 每秒请求数量的限制。

    Returns:
    - dict: protein_id到species_id的映射
    """
    species_dict = {}
    tasks = []

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        for record in protein_records:
            
            protein_id = record.id  # 提取SeqRecord的id属性作为UniProt ID

            # 检查缓存
            cached_result = cache_species_lookup(protein_id)
            if cached_result is not None:
                species_dict[protein_id] = cached_result
                continue

            # 添加异步任务
            tasks.append(fetch_species_async(protein_id, session))

            # 限制请求速率
            if len(tasks) >= rate_limit:
                results = await asyncio.gather(*tasks)
                for pid, species in zip([rec.id for rec in protein_records[:len(results)]], results):
                    species_dict[pid] = species
                tasks = []  # 重置任务列表
                await asyncio.sleep(1)  # 间隔1秒以控制请求速率

        # 处理剩余的任务
        if tasks:
            results = await asyncio.gather(*tasks)
            for pid, species in zip([rec.id for rec in protein_records[-len(results):]], results):
                species_dict[pid] = species
    print(species_dict)
    # 缓存查询结果
    for protein_id, species_name in species_dict.items():
        cache_species_lookup.cache_clear()  # 清除缓存中旧的记录
        cache_species_lookup(protein_id)  # 更新缓存

    return species_dict

def fetch_protein_species(protein_records):
    """
    同步主函数，方便调用异步API。
    """
    return asyncio.get_event_loop().run_until_complete(fetch_species(protein_records))

def get_blast_identity(file, binary=None):
    if file is None:
        return None
    blast_id = {}
    with open(file) as fp:
        for line in fp:
            pid, id = line.split()
            blast_id[pid] = float(id) if binary is None else int(float(id) >= binary)
    return blast_id

def generate_pmid_emb(data_files, pid_file, output_file):
    # 示例数据
    transformed_data = {}
    #pid_set = get_type_list(pid_file)
    for file in data_files:
        with open(file, 'r') as fp:
            for line in fp.readlines():
                json_data = json.loads(line)
            # 将数据转换为新的格式
                # if json_data["paper_id"] in pid_set:
                #if json_data["pid"] in pid_set:
                #transformed_data.update({json_data["paper_id"]: json_data["embedding"]})
                with open(output_file+json_data["paper_id"], 'w') as file:
                    #json.dump(transformed_data, file, indent=4)
                    json.dump({json_data["paper_id"]: json_data["embedding"]}, file)

def get_pmid_term_score(res_file):
    term_scores = defaultdict(dict)
    with open(res_file) as fp:
        for line in fp:
            if line.startswith(result_key_words):
                continue
            line_list = line.split()
            pid, pmid, go_term, score = line_list
            # if go_term in ontology:
            term_scores[(pid,pmid)][go_term] = float(score)
    return term_scores


# def get_pmid_consensus_score(res_file, alpha=1):
#     term_scores = get_pmid_term_score(res_file)
#     res = defaultdict(dict)
#     for seq, pmid in term_scores:
#         for go_term in term_scores[(seq,pmid)]:
#             try:
#                 print(res[seq][go_term])
#             except:
#                 res[seq][go_term] = 1.0
#             res[seq][go_term] *= (1.0 - alpha * term_scores.get((seq,pmid), {}).get(go_term, 0.0))
#     for seq in res:
#         for go_term in res[seq]:
#              res[seq][go_term] = 1.0 - res[seq][go_term]
#     return res

def get_pmid_consensus_score(res_file, alpha=1):
    # Assume get_pmid_term_score is defined elsewhere and retrieves a dictionary
    # Mapping from (sequence, PMID) tuple to a dictionary of GO terms and their scores
    term_scores = get_pmid_term_score(res_file)  # This function needs to be defined or mocked for full functionality
    res = defaultdict(lambda: defaultdict(lambda: 1.0))  # Using nested defaultdict for automatic handling of missing keys
    
    # Iterate over each (sequence, PMID) pair and their corresponding GO term scores
    for (seq, pmid), scores in term_scores.items():
        for go_term, score in scores.items():
            # Update the GO term score for the sequence based on the current term score and alpha
            # Multiplicative update based on the formula provided
            res[seq][go_term] *= (1 - alpha * score)

    # Adjust final scores by converting them to 1.0 minus the accumulated product
    for seq in res:
        for go_term in res[seq]:
            res[seq][go_term] = 1.0 - res[seq][go_term]

    return res

def get_now():
    return datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S')

def log(*args):
    print(f'[{datetime.datetime.now()}]', *args)

def main(argv):
    conf = configparser.ConfigParser()
    print(conf.read(argv))
    split_annotations_by_namespace(conf.get('annotations', 'data'),
                                   conf.get('annotations', 'bp'),
                                   conf.get('annotations', 'cc'),
                                   conf.get('annotations', 'mf'))


if __name__ == '__main__':
    #main(sys.argv[1:])
    data_files = ['/home1/liuhc/protein/PFP/src/TextGO/text/data/SP/protein_pmid_em_sp20230819.json']
    pid_file = "/home/zhushanfeng/storage/liuhc/baseline/TextGO/text/pmid/testall.pmid"
    output_file = '/home1/liuhc/protein/PFP/data/NG4/mem/emb/pmid_emb/'
    generate_pmid_emb(data_files, pid_file, output_file)
    # res = get_pmid_consensus_score("/home/zhushanfeng/storage/liuhc/CAFA5/data20230819_ontology20230101/res/cafa5_test240530/lrmem_basic+trembl_res.txt")
    # output_res("/home/zhushanfeng/storage/liuhc/CAFA5/data20230819_ontology20230101/res/cafa5_test240530/lrmem+trembl_res.txt", res)
