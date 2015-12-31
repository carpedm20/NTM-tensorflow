Neural Turing Machine in Tensorflow
===================================

Tensorflow implementation of [Neural Turing Machine](http://arxiv.org/abs/1410.5401). This implementation uses an LSTM controller. NTM models with multiple read/write heads are supported.

![alt_tag](NTM.gif)

The referenced torch code and dataset can be found [here](https://github.com/kaishengtai/torch-ntm).


Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [Tensorflow](https://www.tensorflow.org/)
- NumPy


Usage
-----

To train a copy task:

    $ python main.py --task copy --is_train True

To test a copy task:

    $ python main.py --task copy

To see all training options, run:

    $ python main.py --help

which will print:

    usage: main.py [-h] [--task TASK] [--epoch EPOCH] [--input_dim INPUT_DIM]
                  [--output_dim OUTPUT_DIM] [--min_length MIN_LENGTH]
                  [--max_length MAX_LENGTH] [--checkpoint_dir CHECKPOINT_DIR]
                  [--is_train [IS_TRAIN]] [--nois_train]

    optional arguments:
      -h, --help            show this help message and exit
      --task TASK           Task to run [copy]
      --epoch EPOCH         Epoch to train [100000]
      --input_dim INPUT_DIM
                            Dimension of input [10]
      --output_dim OUTPUT_DIM
                            Dimension of output [10]
      --min_length MIN_LENGTH
                            Minimum length of input sequence [1]
      --max_length MAX_LENGTH
                            Maximum length of output sequence [20]
      --checkpoint_dir CHECKPOINT_DIR
                            Directory name to save the checkpoints [checkpoint]
      --is_train [IS_TRAIN]
                            True for training, False for testing [False]
      --nois_train


Results
-------

Copy task:

![alt_tag](result_15_12_31.png)

Recall task (in progress):



Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
