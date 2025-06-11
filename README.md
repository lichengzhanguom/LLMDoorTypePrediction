## Introduction
The repository is leveraging LLMs (here GPT-4.1) to predict door type after door detection in [our another repository](https://github.com/lichengzhanguom/Co-DETR/tree/own).
As a result, we generate a dataset of different door types as well as their locations with trivial manual labeling effort.

## Running
### 1.Run predict.py
```shell
python predict.py
```
### 2.Run refine.py
```shell
python refine.py
```
### 3.Run removeduplicateboxes.py
```shell
python removeduplicateboxes.py
```
### 4. Run filterwindow.py
```shell
python filterwindow.py
```
### 5.Run balconyemergencyprediction.py
```shell
python balconyemergencyprediction.py
```
### Run baseline
```shell
python LLMemergencyexitprediction.py
```
### Calculate accuracy
```shell
python calculateaccuracy.py
```
