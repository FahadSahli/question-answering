# question-answering
This repository includes the implementation of the paper “XXXXX”. A blog post illustrating the model could be found at “XXXX”.

The model is implemented using TensorFlow 2, and it is on HAN_Model.py. Config.py sets some necessary parameters, utility_methods.py contains some utility methods, and HAN.py includes the main method.

Here is an example to the run the code in training mode:
	python3 HAN.py --training True --data ./data/ --embedding_file ./glove.6B.200d.txt
The required commands are data and embedding_file

The folder notebooks includes a notebook version of the model implemented using TensorFlow 1, data pre-processing, and word embeddings extracted from BERT.
