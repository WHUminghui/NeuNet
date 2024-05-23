# NeuNet
Code and Data about 《Joint Learning Neuronal Skeleton and Brain Circuit Topology with Permutation Invariant Encoders for Neuron Classification》accepted by AAAI 2024. The data were derived from our reprocessed data, which we put in [Here](https://drive.google.com/drive/folders/1adpq49VKfUyH7SXh-G5DznlIXsGVj1Eu?usp=drive_link).

## Installation
### Main Requeriments
* Linux
* CUDA environment
* Python==3.7
* Torch==1.11
* torch-cluster==1.6.0
* torch-geometric==2.0.4
* torch-scatter==2.0.9
* torch-sparse==0.6.14
* CUDA==11.1
### Code Structure
```
Source Code
├── data
|   ├──Hemibrain
|   |   ├── Connectome
|   |   |    ├──raw
|   |   |    └──processed
|   |   └── Skeleton
|   |   |    ├──raw
|   |   |    └──processed
|   └──H01
|   |   ├── Connectome
|   |   |    ├──raw
|   |   |    └──processed
|   |   └── Skeleton
|   |   |    ├──raw
|   |   |    └──processed
...
```

Download relative datasets from [here](https://drive.google.com/drive/folders/1adpq49VKfUyH7SXh-G5DznlIXsGVj1Eu?usp=drive_link). And put it(H01 and HemiBrain) on 'datasets/'. 
### Run
python main.py

