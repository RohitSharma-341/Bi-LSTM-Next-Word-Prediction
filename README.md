# Bi-LSTM-Next-Word-Prediction
Bi LSTM (Bi directional Long Short Term Memory) Neural Networks.


It is a type of recurrent neural network (RNN) that processes sequential data in both forward and backward directions.In this model captures both past and future context of input sequence .

It consists of two LSTM layers: one processing the sequence in the forward direction and other in the backward direction . Each layer maintains its own hidden states and memory cells.

Architecture of Bi-LSTM:

1.Input Sequence: The input sequence is a sequence of data points, such as words in a sentence or characters in a text. Each data point is typically represented as a vector or embedded representation.

2.Embedding: The input sequence is often transformed into dense vector representations called embeddings. Embeddings capture the semantic meaning of the data points and provide a more compact and meaningful representation for the subsequent layers.

3.Bi-LSTM: The Bi-LSTM layer is the core component of the architecture. It consists of two LSTM layers: one processing the input sequence in the forward direction and the other in the backward direction. Each LSTM layer has its own set of parameters.

4.Output: The output of the Bi-LSTM layer is the combination of the hidden states from both the forward and backward LSTM layers at each time step. The specific combination method can vary, such as concatenating the hidden states or applying a different transformation.


https://github.com/RohitSharma-341/Bi-LSTM-Next-Word-Prediction/assets/139057604/96b00cfa-9cfd-421a-85c4-8bc895488757
