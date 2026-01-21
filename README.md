# GOReasoner

Official code for GOReasoner: Enhancing Protein Function Annotation through Explainable Reasoning with LLM

# Requirements

```shell
uv pip install -r requirements.txt
```

# Data Preprocessing

```shell
python get_testdata.py --input test_proteins.txt --output testdata.json
```

For test_proteins.txt, each line is ```[protein id]\t[goid]\t[pmid]```, for example,

```text
P12345	GO:0042802	12345678
P67890	GO:0003674	11111111
```

# Prediction

## prediction and propagation

```shell
python GOReasoner.py \
    --input_file testdata/GOR2023/bp_with_desc.json \
    --output_file results/GOR2023/bp_results.json \
    --domain bp
```

## consensus fusion

```shell
python consensus_fusion.py \
    --init_res init_res.txt \
    --refine_res refine_res.txt
```