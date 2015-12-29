from tasks import *

def main(_):
    ntm = copy_train()
    copy(ntm, 20)

if __name__ == '__main__':
    tf.app.run()
