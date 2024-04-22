# reproduction

To reproduce the results in our paper, follow the steps below, starting from the main directory.

## install

```
pip install -r requirements.txt
```

## prepare datasets

Download datasets and save two copies, one in UTF8 and the other using the gpt2 tokenizer:

```
python3 prepare.py pg19 --out_dir=datasets/pg19
python3 prepare.py pg19 --out_dir=datasets/pg19 --tokenizer=gpt2
python3 prepare.py lucadiliello/STORIES --out_dir=datasets/STORIES
python3 prepare.py lucadiliello/STORIES --out_dir=datasets/STORIES --tokenizer=gpt2
python3 prepare.py monology/pile-uncopyrighted --out_dir=datasets/arxiv --max_data_bytes=20e9 --filter=meta.pile_set_name.ArXiv
python3 prepare.py monology/pile-uncopyrighted --out_dir=datasets/arxiv --max_data_bytes=20e9 --filter=meta.pile_set_name.ArXiv --tokenizer=gpt2
python3 prepare.py monology/pile-uncopyrighted --out_dir=datasets/github --max_data_bytes=20e9 --filter=meta.pile_set_name.Github
python3 prepare.py monology/pile-uncopyrighted --out_dir=datasets/github --max_data_bytes=20e9 --filter=meta.pile_set_name.Github --tokenizer=gpt2
```

In order to prepare the datasets using the sentencepiece tokenizer, you'll need to install sentencepiece with the spm_train command to train the sentencepiece tokenizer.
See [sentencepiece installation](https://github.com/google/sentencepiece?tab=readme-ov-file#installation).
To train the sentencepiece tokenizers:

```
for dir in pg19 arxiv github; do
    cd datasets/$dir
    spm_train --input=train.txt --model_prefix=sp --model_type=bpe --vocab_size=50257 --num_threads=32 --allow_whitespace_only_pieces=True --remove_extra_whitespaces=False --byte_fallback=True --normalization_rule_name=identity --input_sentence_size=10000000
    cd -
done
```

Download datasets and save using sentencepiece (sp) tokens:

```
python3 prepare.py pg19 --out_dir=datasets/pg19 --tokenizer=sp
python3 prepare.py monology/pile-uncopyrighted --out_dir=datasets/arxiv --max_data_bytes=20e9 --filter=meta.pile_set_name.ArXiv --tokenizer=sp
python3 prepare.py monology/pile-uncopyrighted --out_dir=datasets/github --max_data_bytes=20e9 --filter=meta.pile_set_name.Github --tokenizer=sp
```

## SpaceByte-793M+184M training

To train SpaceByte-793M+184M on pg19, STORIES, arxiv, and github, run:

```
for dataset in pg19 STORIES arxiv github; do
    python3 train.py --dataset=$dataset --model=SpaceByte --context_size=8192 --global_context_size=1344 --d_model=1536 --d_local=768 --n_layers=28 --n_local_layers=26 --local_attention_window=768 --rope --batch_size=64 --iters='30e9/tokens' --lr='0.5e-2*B**0.5' --beta2=0.98 --patch_method=utf8 --micro_batch_size=2 --out_dir=spacebyte-793M184M
done
```

The trained models will appear in a `spacebyte-793M184M` subdirectory.
wandb will be used for logging into a project named `spacebyte-793M184M`.
See `test bits per byte' in wandb for the bits-per-byte for the test split.

## Transformer-1B training

To train the subword Transformer-1B models on pg19, STORIES, arxiv, and github, run:

```
python3 train.py --batch_size=64 --beta2=0.98 --context_size=2048 --d_model=1536 --iters='7.49e9/tokens' --lr=0.5e-2*B**0.5 --rope=True --micro_batch_size=2 --tokenizer=sp --dataset=pg19 --n_layers=40 --out_dir=spacebyte8_medium2
python3 train.py --batch_size=64 --beta2=0.98 --context_size=2048 --d_model=1536 --iters='6.83e9/tokens' --lr=0.5e-2*B**0.5 --rope=True --micro_batch_size=2 --tokenizer=gpt2 --dataset=STORIES --n_layers=44 --out_dir=spacebyte8_medium2
python3 train.py --batch_size=64 --beta2=0.98 --context_size=2048 --d_model=1536 --iters='8.10e9/tokens' --lr=0.5e-2*B**0.5 --rope=True --micro_batch_size=2 --tokenizer=sp --dataset=arxiv --n_layers=37 --out_dir=spacebyte8_medium2
python3 train.py --batch_size=64 --beta2=0.98 --context_size=2048 --d_model=1536 --iters='9.52e9/tokens' --lr=0.5e-2*B**0.5 --rope=True --micro_batch_size=2 --tokenizer=sp --dataset=github --n_layers=31 --out_dir=spacebyte8_medium2
```

## Pareto frontier grid search

To train the Pareto frontier models using the grid search, you'll first want to set the SUBMIT_COMMAND variable in `reproduce/jobs.py` so that the training runs aren't all done locally.
Then you can launch the grid search using

```
python3 reproduce/jobs.py
```

The trained models will appear in a `spacebyte` subdirectory.
wandb will be used for logging into a project named `spacebyte`.

To create the plots and table data in the paper, move `reproduce/plots.py` and the `reproduce/experiments.ipynb` jupyter notebook into the main directory and run the jupyter notebook.
