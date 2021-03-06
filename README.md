# BERT_Chinese_NER
Implementing BERT + CRF with PyTorch for Chinese named entity recognition (NER).

## Quickstart
### Prerequisites
#### virtualenv option
* Create a python virtual environment `virtualenv venv`
* Source `source venv/bin/activate`

#### conda option
* Create a python virtual environment 
* `conda create --name venv python=3.7`
* Source `conda activate venv`

### Installing
* Install required python package `pip install -r requirements.txt`

### Training
* Download People’s Daily dataset from https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER and place the data files in `data` folder.
* Run the following command
```bash
python run_ner.py
```

## Implementation detail
### model list
* BERT + CRF

### Run NER
## Results
#### BERT + CRF 
The losses and F1-score
![](https://github.com/RocioLiu/bert_chinese_ner/blob/main/outputs/images/loss_metric.png)
