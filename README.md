# TemBERTure

We  developed TemBERTure, a deep-learning package for protein thermostability prediction. It consists of three components: 
(i) TemBERTureDB, a large curated database of thermophilic and non-thermophilic sequences;
(ii) TemBERTureCLS, a classifier  which predicts  the thermal class (non-thermophilic or thermophilic) of a protein sequence;
(iii) TemBERTureTm, a regression model, which predicts the melting temperature of a protein, based on its primary sequence. 

Both models are built upon the existing protBERT-BFD language model [1] and fine-tuned through an adapter-based approach [2], [3]. 

This repository provides implementations and weights for both tasks, allowing users to leverage these models for various protein-related predictive tasks.

## How to use TemBERTure

#### Download
```
git clone https://github.com/ibmm-unibe-ch/TemBERTure.git
cd TemBERTure
git filter-branch --subdirectory-filter temBERTure -- --all
```



```
seq = 'MEKVYGLIGFPVEHSLSPLMHNDAFARLGIPARYHLFSVEPGQVGAAIAGVRALGIAGVNVTIPHKLAVIPFLDEVDEHARRIGAVNTIINNDGRLIGFNTDGPGYVQALEEEMNITLDGKRILVIGAGGGARGIYFSLLSTAAERIDMANRTVEKAERLVREGEGGRSAYFSLAEAETRLDEYDIIINTTSVGMHPRVEVQPLSLERLRPGVIVSNIIYNPLETKWLKEAKARGARVQNGVGMLVYQGALAFEKWTGQWPDVNRMKQLVIEALRR'
```
#### TemBERTure_CLS:
```
model = TemBERTure(adapter_path='./temBERTure/temBERTure_CLS/', device='cuda:6',batch_size=16, task = 'classification')) # temberture_cls
model.predict([seq])
```
#### TemBERTure_TM:
```
model_replica1 = TemBERTure(adapter_path='./temBERTure/temBERTure_TM/replica1/', device='cuda:6',batch_size=16, task = 'regression') # temberture_Tm
model_replica2 = TemBERTure(adapter_path='./temBERTure/temBERTure_TM/replica2/', device='cuda:6',batch_size=16, task = 'regression') # temberture_Tm
model_replica3 = TemBERTure(adapter_path='./temBERTure/temBERTure_TM/replica3/', device='cuda:6',batch_size=16, task = 'regression') # temberture_Tm
```


## Database Content


The database included in this repository comprises:

- **[Database Name]**: A brief description of the main database used.
- **[File Names]**: List of files containing data in the database.
- **[Data Format]**: Description of the data format used in the database.

## Training Data

In addition to the database, we also provide specific data for model training. These data have been prepared and pre-processed for direct use in training the models offered in this repository.

- **[Dataset Name]**: A description of the training data.
- **[Dataset Link]**: Datasets that were used to train, validate, and test TemStaPro are available in Zenodo.

[1] A. Elnaggar et al., “ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 10, pp. 7112–7127, Oct. 2022, doi: 10.1109/TPAMI.2021.3095381.  
[2]	N. Houlsby et al., “Parameter-Efficient Transfer Learning for NLP.” arXiv, Jun. 13, 2019. Accessed: Feb. 14, 2024. [Online]. Available: http://arxiv.org/abs/1902.00751  
[3]	C. Poth et al., “Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning,” 2023, doi: 10.48550/ARXIV.2311.11077.
