"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf

from data_generator import DataGenerator
from dataset_mini import *
from dataset_tiered import *
from maml_semi1 import MAML
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

# added flags
flags.DEFINE_integer('lr_mode', 0, 'inner lr mode (default 0), 1: all variables share one lr 2: each variable has one lr  3: each variable has one lr with the same shape')
flags.DEFINE_integer('num_unlabel', 0, 'unlabel data each class')
flags.DEFINE_integer('n_distractor', 0, 'distractor class number')

flags.DEFINE_float('reg', 1e-3, 'weight for regularization.') # 0.1 for omniglot

def load_batch_data(loader, n_way, n_shot, n_query, num_unlabel, n_distractor):
    inputa = []
    labela = []
    inputb = []
    labelb = []
    unlabel = []
    if loader.split=="test":
        FLAGS.meta_batch_size = 1
    for b in range(FLAGS.meta_batch_size):
        s, s_labels, q, q_labels, u = loader.next_data(n_way, n_shot, n_query, num_unlabel, n_distractor)
        s = np.reshape(s,(n_way*n_shot,-1))
        q = np.reshape(q,(n_way*n_query,-1))
        s_labels = np.reshape(s_labels, (-1))
        q_labels = np.reshape(q_labels, (-1))
        if num_unlabel>0:
            u = np.reshape(u,((n_way+n_distractor)*num_unlabel,-1))
            unlabel.append(u)
        inputa.append(s)
        inputb.append(q)
        
        s_onehot = np.zeros((n_shot*n_way,n_way))
        s_onehot[np.arange(n_shot*n_way),s_labels] = 1
        labela.append(s_onehot)
        q_onehot = np.zeros((n_query*n_way,n_way))
        q_onehot[np.arange(n_query*n_way),q_labels] = 1
        labelb.append(q_onehot)


    inputa = np.array(inputa)
    labela = np.array(labela)
    inputb = np.array(inputb)
    labelb = np.array(labelb)
    if num_unlabel>0:
        unlabel = np.array(unlabel)

    return inputa, labela, inputb, labelb, unlabel


def train(model, saver, sess, exp_string, resume_itr=0):
    args = {}
    args['x_dim'] = '84,84,3'
    args['ratio'] = 1.0
    args['seed'] = 1000
    n_query = 15
    num_unlabel = FLAGS.num_unlabel
    n_distractor = FLAGS.n_distractor
    if FLAGS.datasource=='miniimagenet':
        loader_train = dataset_mini(600, 100, 'train',args)
        loader_val = dataset_mini(600, 100, 'val',args)
    elif FLAGS.datasource=='tiered':
        loader_train = dataset_tiered(600, 100, 'train',args)
        loader_val = dataset_tiered(600, 100, 'val',args)
    
    loader_train.load_data_pkl()
    loader_val.load_data_pkl()

    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 2000
    if FLAGS.datasource == 'sinusoid':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = FLAGS.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        if True:
            inputa, labela, inputb, labelb, unlabel = load_batch_data(loader_train, num_classes, FLAGS.update_batch_size, n_query, num_unlabel, n_distractor)
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.unlabel:unlabel}

        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource !='sinusoid':
            if True:
                inputa, labela, inputb, labelb, unlabel = load_batch_data(loader_val, num_classes, FLAGS.update_batch_size, n_query, num_unlabel, n_distractor)
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.unlabel:unlabel}
                if model.classification:
                    input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]]
                else:
                    input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]

            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot
NUM_TEST_POINTS = 600

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = FLAGS.num_classes # for classification, 1 otherwise

    np.random.seed(1000)
    random.seed(1000)

    metaval_accuracies = []
    
    args = {}
    args['x_dim'] = '84,84,3'
    args['ratio'] = 1.0
    args['seed'] = 1000
    n_query = 1
    num_unlabel = FLAGS.num_unlabel
    n_distractor = FLAGS.n_distractor
    if FLAGS.datasource=='miniimagenet':
        loader_test = dataset_mini(600, 100, 'test',args)
    elif FLAGS.datasource=='tiered':
        loader_test = dataset_tiered(600, 100, 'test',args)
    
    loader_test.load_data_pkl()
    for _ in range(NUM_TEST_POINTS):
        inputa, labela, inputb, labelb, unlabel = load_batch_data(loader_test, num_classes, FLAGS.update_batch_size, n_query, num_unlabel, n_distractor)
        feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.unlabel:unlabel, model.meta_lr: 0.0}

        if model.classification:
            result = sess.run([model.total_accuracy1] + model.total_accuracies2, feed_dict)
        else:  # this is for sinusoid
            result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'w') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

def main():
    if FLAGS.datasource == 'sinusoid':
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 10
    else:
        if FLAGS.datasource == 'miniimagenet':
            if FLAGS.train == True:
                test_num_updates = 1  # eval on at least one update during training
            else:
                test_num_updates = 10
        else:
            test_num_updates = 10

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    # if FLAGS.datasource == 'sinusoid':
    #     data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    # else:
    #     if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
    #         assert FLAGS.meta_batch_size == 1
    #         assert FLAGS.update_batch_size == 1
    #         data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
    #     else:
    #         if FLAGS.datasource == 'miniimagenet': # TODO - use 15 val examples for imagenet?
    #             if FLAGS.train:
    #                 data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
    #             else:
    #                 data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
    #         else:
    #             data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory


    dim_output = FLAGS.num_classes
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = 84*84*3

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train:
        model.construct_model(input_tensors=None, prefix='metatrain_')
    else:
        model.construct_model(input_tensors=None, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=40)
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print(var.name)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')
    if FLAGS.lr_mode>0:
        exp_string += 'lrmode'+str(FLAGS.lr_mode)
    print(exp_string)

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    #tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, resume_itr)
    else:
        test(model, saver, sess, exp_string, test_num_updates)

if __name__ == "__main__":
    main()
