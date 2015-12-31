Neural Turing Machine in Tensorflow
===================================

Tensorflow implementation of [Neural Turing Machine](http://arxiv.org/abs/1410.5401). This implementation uses an LSTM controller. NTM models with multiple read/write heads are supported.

![alt_tag](etc/NTM.gif)

The referenced torch code can be found [here](https://github.com/kaishengtai/torch-ntm).


Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [Tensorflow](https://www.tensorflow.org/)
- NumPy


Usage
-----

To train a copy task:

    $ python main.py --task copy --is_train True

To test a *quick* copy task:

    $ python main.py --task copy --test_max_length 10


Results
-------

More detailed results can be found [here](ipynb/NTM\ Test.ipynb)

Copy task:

![alt_tag](etc/result_1.png)
![alt_tag](etc/result_2.png)

Recall task:

(in progress)


Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
