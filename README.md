# Multiple Hypothesis Tracking
This is an implementation of the Multiple Hypothesis Tracking filter,
implemented for educational purposes and for the purpose of the article
''Spatially Indexed Clustering for Scalable Tracking of Remotely Sensed Drift
Ice'' accepted for the IEEE Aerospace 2017 conference, Big Sky, MT.

In particular, this implementation studies the use of Spatial Indexing in the
MHT clustering process.

Plaese be advised that this implementation is educational above efficient, and would require some understanding of the algorithm to tune. 

# Building
The library is built usig the ''[tup](http://gittup.org/tup/)'' buildsystem. Just install tup and execute ''tup'' in the root directory, and a python 3.7 module will be built for python 3.7. If you want to build for a different python version, edit the files in [this commit](https://github.com/jonatanolofsson/mht/commit/c4af9c313c4e44ca23edd418b5281618fc29693d) correspondingly.

# Examples
Have a look at the [test_mht-file](https://github.com/jonatanolofsson/mht/blob/master/tests/test_mht.py) for a usage example

# License
This software is released under the GPLv3 license.
