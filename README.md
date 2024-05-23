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

Download relative datasets from [here](https://drive.google.com/drive/folders/1adpq49VKfUyH7SXh-G5DznlIXsGVj1Eu?usp=drive_link). And put it(H01 and HemiBrain) on 'data/'. 

For each dataset it should be constructed separately for its Skeleton data and Connectome data, and then each of them corresponds to a floder that includes '... /raw/' and ' .../processed/' files.

The raw data will be different from one data to another, so I don't provide how to process from a raw data to ' ... /raw/' form. But for each dataset I have provided how to process the data from '... /raw/' to '.../processed/' , e.g. 'datasets/HemiSkeleton.py'. You can delete the ' .../processed' folder of the data or change the name, and 'datasets/HemiSkeleton.py' will demonstrate how to get 'HemiBrain/Skeleton/processed/' from 'HemiBrain/Skeleton/raw/' that can be used directly by NeuNet. In summary, if you want to swap in your own dataset, you'll want to prepare your own '... /raw/' folder for Skeleton and Connectome, and then similar to HemiSkeleton.py and HemiConnectome.py, build your own '... /processed/' file for Skeleton and Connectome, respectively.

### Run
python main.py

