import datetime
import os
import sys
from collections import deque
import mxnet
from mxnet import gluon, nd, autograd
from sklearn.metrics import confusion_matrix
import numpy as np




def test(metric, ctx, net, val_data, num_views=1, num_class=None, if_caps=False, use_viewpoints=False):
    assert num_views >= 1, "'num_views' should be greater or equal to 1"
    metric.reset()
    iiiii = 1

    # val_data.reset()
    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]
    for data, depth_1,depth_2, label, *rest in val_data:
        position_1=depth_1
        position_2 = depth_2
        if use_viewpoints:
            _, shuffle_idx, _ = rest
        if data.shape[0] == 1:
            Xs = [data.as_in_context(ctx[0])]
            Ys = [label.as_in_context(ctx[0])]
            position_1 = [position_1.as_in_context(ctx[0])]
            position_2 = [position_2.as_in_context(ctx[0])]
            if use_viewpoints:
                IDs = [shuffle_idx.as_in_context(ctx[0])]
        else:
            Xs = gluon.utils.split_and_load(data,
                                            ctx_list=ctx, batch_axis=0, even_split=False)
            Ys = gluon.utils.split_and_load(label,
                                            ctx_list=ctx, batch_axis=0, even_split=False)
            position_1 = gluon.utils.split_and_load(position_1,
                                                  ctx_list=ctx, batch_axis=0, even_split=False)
            position_2 = gluon.utils.split_and_load(position_2,
                                                  ctx_list=ctx, batch_axis=0, even_split=False)
            if use_viewpoints:
                IDs = gluon.utils.split_and_load(shuffle_idx,
                                                 ctx_list=ctx, batch_axis=0, even_split=False)
        if not use_viewpoints:
            if if_caps:
                outputs = [net(X)[1].squeeze(axis=-1) for X in Xs]
            else:
                if num_views > 1:
                    outputs = [net(X).reshape(-1, num_views, num_class).mean(axis=1) for X in Xs]
                else:
                    outputs = []
                    for x, y, z,w in zip(Xs, Ys, position_1,position_2):
                        if iiiii % 10 == 0:
                            print('test batch:', iiiii,datetime.datetime.now())
                        iiiii += 1
                        out = net(x,z,w)
                        # output_per_class(num_class, out, y, iiiii)
                        outputs.append(out)
        else:
            if num_views > 1:
                outputs = [net(X, ID)[0].reshape(-1, num_views, num_class).mean(axis=1) for X, ID in zip(Xs, IDs)]
            else:
                outputs = [net(X, ID)[0] for X, ID in zip(Xs, IDs)]
        metric.update(Ys, outputs)
    return metric.get()



def get_format_time_string(time_interval):
    h, remainder = divmod((time_interval).seconds, 3600)
    m, s = divmod(remainder, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def get_confusion_matrix(net, val_data, ctx, num_views=1, num_class=None):
    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]
    y_preds, y_trues = [], []
    for data, label in val_data:
        if data.shape[0] == 1:
            Xs = [data.as_in_context(ctx[0])]
        else:
            Xs = gluon.utils.split_and_load(data,
                                            ctx_list=ctx, batch_axis=0, even_split=False)

        if num_views > 1:
            outputs = [net(X).reshape(-1, num_views, num_class).mean(axis=1) for X in Xs]
        else:
            outputs = [net(X) for X in Xs]
        output_labels = [out.argmax(axis=1) for out in outputs]
        y_preds.append(nd.concat(*output_labels, dim=0).astype('uint8').asnumpy())
        y_trues.append(label.asnumpy())
    return confusion_matrix(np.concatenate(y_trues, axis=None), np.concatenate(y_preds, axis=None))


def get_view_sequences(num_views):
    s = deque(range(num_views))
    seqs = []
    for i in range(len(s)):
        s.rotate(1)
        seqs.append(list(s))
    s_r = deque(range(num_views - 1, -1, -1))
    for i in range(len(s_r)):
        s_r.rotate(1)
        seqs.append(list(s_r))
    return seqs


def log_string(log_out, out_str):
    log_out.write(out_str + '\n')
    log_out.flush()


def smooth(label, classes, eta=0.1):
    ind = label.astype('int')
    res = nd.zeros((ind.shape[0], classes), ctx=label.context)
    res += eta / classes
    res[nd.arange(ind.shape[0], ctx=label.context), ind] = 1 - eta + eta / classes
    return res


def save_checkpoint(net, current_epoch, checkpoint_prefix):
    net.save_parameters(os.path.join(checkpoint_prefix, 'Epoch%s.params' % current_epoch))


def train(net, train_data, test_data, loss_fun, kvstore, log_out, checkpoint_prefix, train_args):
    trainer_dict = {'learning_rate': train_args.lr, 'wd': train_args.wd}
    if train_args.optimizer == 'sgd':
        trainer_dict['momentum'] = 0.9
    trainer = gluon.Trainer(net.collect_params(), train_args.optimizer, trainer_dict, kvstore=kvstore)
    best_test_acc = 0
    metric = mxnet.metric.Accuracy()

    ctx = [mxnet.gpu(gpu_id) for gpu_id in train_args.gpu]  # 零号GPU卡

    log_string(log_out, str(datetime.datetime.now()))
    log_string(log_out, net.get_info())
    log_string(log_out, str(train_args))
    print('start training on %s' % train_args.dataset_path)

    for epoch in range(train_args.from_epoch, train_args.epoch):
        # epoch
        train_loss = 0.0
        metric.reset()
        tic = datetime.datetime.now()

        # train_acc = 0.0
        if epoch > 0 and epoch % train_args.decay_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * train_args.decay_rate)
        iter_time = 1
        iiiii=1

        for batch in train_data:
            if iiiii%10==0:
                print('batch:',iiiii,datetime.datetime.now())
            iiiii+=1
            data, depth_1,depth_2, label, sample_weights = batch
            if train_args.label_smoothing:
                label_smooth = smooth(label, train_args.num_classes)
            else:
                label_smooth = label

            gpu_data = gluon.utils.split_and_load(data, ctx, even_split=False)
            gpu_label = gluon.utils.split_and_load(label_smooth.astype('float32'), ctx, even_split=False)
            gpu_weights = gluon.utils.split_and_load(sample_weights.astype('float32'), ctx, even_split=False)
            gpu_depth_1 = gluon.utils.split_and_load(depth_1, ctx, even_split=False)
            gpu_depth_2 = gluon.utils.split_and_load(depth_2, ctx, even_split=False)

            outputs = []
            Ls = []
            outputs0 = []
            Ls0 = []
            with autograd.record():
                #for x, y, weight in zip(gpu_data, gpu_label, gpu_weights):
                for x, y, weight,position_1,position_2 in zip(gpu_data, gpu_label, gpu_weights,gpu_depth_1,gpu_depth_2):
                    out = net(x,position_1,position_2)

                    if train_args.multi_output:
                        loss = loss_fun(out, nd.repeat(y, repeats=train_args.num_views, axis=0),
                                        nd.repeat(weight, repeats=train_args.num_views, axis=0))
                    else:
                        if train_args.use_sample_weights:
                            loss = loss_fun(out, y, weight)
                        else:
                            loss = loss_fun(out, y)

                    if 1==0:
                        out0=1
                        if train_args.multi_output:
                            loss0 = loss_fun(out0, nd.repeat(y, repeats=train_args.num_views, axis=0),
                            nd.repeat(weight, repeats=train_args.num_views, axis=0))
                        else:
                            if train_args.use_sample_weights:
                                loss0 = loss_fun(out0, y, weight)
                            else:
                                loss0 = loss_fun(out0, y)
                        outputs0.append(out0)
                        Ls0.append(loss0)

                    outputs.append(out)
                    Ls.append(loss)
            autograd.backward(Ls)


            if iter_time % train_args.batch_update_period == 0:
                trainer.step(train_args.batch_size * train_args.batch_update_period)
                net.collect_params().zero_grad()
            elif iter_time == len(train_data):
                trainer.step(train_args.batch_size * (len(train_data) % train_args.batch_update_period))
                net.collect_params().zero_grad()
            if train_args.label_smoothing:
                origin_label = gluon.utils.split_and_load(label.astype('float32'), ctx, even_split=False)
            else:
                origin_label = gpu_label
            if train_args.multi_output:
                avg_output = [output.reshape((-1, train_args.num_views, train_args.num_classes)).mean(axis=1) for output
                              in outputs]
                metric.update(preds=avg_output, labels=origin_label)
            else:
                metric.update(preds=outputs, labels=origin_label)
            train_loss += nd.sum(loss).asscalar()
            iter_time += 1


        _, train_acc = metric.get()
        if train_args.multi_output:
            _, test_acc = test(metric, ctx, net, test_data, num_views=train_args.num_views,

                               num_class=train_args.num_classes)
        else:
            _, test_acc = test(metric, ctx, net, test_data)
        save_checkpoint(net, '_latest', checkpoint_prefix)
        if test_acc >= best_test_acc:
            save_checkpoint(net, '_best', checkpoint_prefix)
            best_test_acc = test_acc
        toc = datetime.datetime.now()
        time_cost = get_format_time_string(toc - tic)
        epoch_str = "Epoch %d. Loss: %f, Train acc: %f, Valid acc: %f, Best acc: %f, lr: %f, Time: %s" % (
            epoch, train_loss / len(train_data) / train_args.batch_size,
            train_acc, test_acc, best_test_acc, trainer.learning_rate, time_cost)
        print(epoch_str)
        log_string(log_out, epoch_str)
