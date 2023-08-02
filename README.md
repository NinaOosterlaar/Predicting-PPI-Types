# Predicting-PPI-Types

## Introduction

This github provides algorithms that are able to classify PPIs as either activating, inhibiting, or neutral based on the GO terms of the proteins. Four algorithms use the Transformer algoirthm based on the [TransformerGO](https://github.com/Ieremie/TransformerGO) algorithm. Two algorithms use a Recurrent neural network algorithm created by TensorFlow.

To run the code, a sqlite database needs to be created. The sqlite file used is too large to commit to github. You can contact me for the sqlite file, or create your own. The database and data_retriever folder provide code needed to create your own sqlite database, but does not provide you with all the necessary data, since this would be too large. An image with the overview of the structure of the database can also be found in the database folder.

### Node2Vec

The folder Node2Vec_Transformer_multiclass contains all the necessary code for classifiying PPIs as activating, inhibiting, or neutral when the GO terms are processed by the Node2Vec algorithm. Node2Vec_Transformer_pos_neg contains all the necessary code for classifying PPIs as either activating or inhibiting with Node2Vec processed GO terms. Both folders contain three files with the vectors produced when running Node2Vec on all the GO terms. If a new file with vectors needs to be created, the algorithm to do that can be found [here](https://github.com/Ieremie/TransformerGO).

### Sent2Vec

The folder Sent2Vec_Transformer_multiclass contains all the necessary code for classigying PPIs as activating, inhibiting, or neutral when the GO terms are processed by the Sent2Vec algorithm. Sent2Vec_Transformer_pos_neg contains all the necessary code for classigying PPIs as either activating or inhibiting with Sent2Vec processed GO terms. The folder containing the vectors of the Sent2Vec processed GO terms is too large to commit to github. When needed you can contact me, or you can create your own file. Create_sentence_embeddings contains the python file needed to create the sentence embeddings. The sentence embedding algorithm then needs to be downloaded from the [BioSent2Vec github](https://github.com/ncbi-nlp/BioSentVec). BioSent2Vec uses the following [Sent2Vec](https://github.com/epfml/sent2vec) algorithm, which needs to be installed beforehand. 

### RNN

The folder recurrent_nn contains all the necessary code for classifying PPIs as activating, inhibiting, or neutral or just as activating or inhibiting using the RNN instead of the Transformer neural network. It uses Node2Vec processed GO terms. 


