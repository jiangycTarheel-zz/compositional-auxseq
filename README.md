# compositional-auxseq

This repo contains the source code of the models described in the following paper 
* *"Inducing Transformer's Compositional Generalization Ability via Auxiliary Sequence Prediction Tasks"* in Proceedings of EMNLP, 2021. ([paper](https://arxiv.org/abs/2109.15256)).

The basic code structure was adapted from the HuggingFace [Transformers](https://github.com/huggingface/transformers).

## 0. Preparation
### 0.1 Dependencies
* PyTorch 1.4.0/1.6.0/1.8.0

### 0.2 Data

* Download the original SCAN [data](https://github.com/brendenlake/SCAN)
* Download the SCAN [MCD splits](https://github.com/google-research/google-research/tree/master/cfq)
* Organize the data into `data/scan` and make sure it follows such a structure:
```
------ data
--------- scan
------------ tasks_test_mcd1.txt
------------ tasks_train_mcd1.txt
------------ tasks_val_mcd1.txt
```

## 2. Training
* Train the model on the SCAN MCD1 splits by running:
```
./train_scan_scripts/train_auxseq_mcd1.sh
```
* By defaults, the top-5 best model checkpoints will be saved in `out/scan/auxseq-00`.

## 3. Evaluation
* Set the `EVAL_EPOCH` parameter in the `eval_scan_scripts/eval_auxseq_mcd1.sh`.
* Evaluate the model on the SCAN MCD1 splits by running:
```
./eval_scan_scripts/eval_auxseq_mcd1.sh
```

## Citation
```
@inproceedings{jiang-bansal-2021-enriching,
    title = "Inducing Transformer's Compositional Generalization Ability via Auxiliary Sequence Prediction Tasks",
    author = "Jiang, Yichen and Bansal, Mohit",
    booktitle = "Proceedings of the EMNLP 2021",
    year = "2021",
}