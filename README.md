Artificial Neural Network
===================

Authors:
* Daniel Beckwith ([dbeckwith](http://github.com/dbeckwith))
* Aditya Nivarthi ([SIZMW](http://github.com/sizmw))

Purpose:
* The purpose of this project is to demonstrate a basic neural network that can classify input data.

Execution:
* Make sure you have the required dependencies:
    * `python 3.4`
    * `matplotlib`
    * `numpy`
* Run the ann.py file with the arguments:
```
python ann.py <data_file> [h <hidden_nodes>] [p <holdout_ratio>]
```
* The program will learn on the data in the given file, reporting the training epochs to standard out as it goes. When it finishes, it will report the best validation set error rate it achieved, at what epoch this occurred, and the network weights at this epoch. A `matplotlib` graph will also appear showing the test and validation error rates over time.
