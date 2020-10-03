# question-answering
This repository includes the implementation of the paper “[Development of Hierarchical Attention Network Based Architecture for Cloze-Style Question Answering](https://link.springer.com/chapter/10.1007/978-3-030-60036-5_14)”. A blog post illustrating the model could be found [here](https://fahadsahli.com/han/).

The model's implemented using TensorFlow 2, and its implementation's on HAN_Model.py. Config.py sets some necessary parameters, utility_methods.py contains some utility methods, and HAN.py includes the main method.

Here's an example to the run the code in training mode:
```
python3 HAN.py --training True --data path/to/data/ --embedding_file path/to/embeddings
```
The required commands are data and embedding_file. Other parameters will be set to their default values if you don't specify the values you want.

The folder notebooks includes a notebook version of the model implemented using TensorFlow 1, data pre-processing, and word embeddings extracted from BERT.

Data and embeddings can be accessed from [here](https://drive.google.com/drive/folders/1cQkHI0jc11z1Huuf9Th3moOvm5YrohSO?usp=sharing)
