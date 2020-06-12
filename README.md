# Robustness of decentralized compressed SGD with gossip communication

To run an instance of our network : 

`pip install -r requirements.txt`
`python run.py`


##Index --
- run.py : Defines a network according to our design choices performs training.
- network.py : Contains definition of our NN model, Nodes, and Network to perform decentralized training.
- optimizer.py : Contains definitions for our custom optimizers for each optimization scheme
- protecc.py : Contains functions which define our protection schemes against attacks
- model_utils.py : Contains miscellaneous functions required to define ring, torus and quantizers



### Authors--
Moulik Choraria (moulik.choraria@epfl.ch)
Aditya Vardhan Varre (aditya.varre@epfl.ch)
Harshvardhan (harshvardhan.harshvardhan@epfl.ch)