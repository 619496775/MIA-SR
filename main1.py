if __name__ == '__main__':
    from torch.nn import parameter
    import os
    import time
    import torch
    import argparse

    from model1 import SASRec
    from tqdm import tqdm
    from utils import *
    import numpy as np
    from AutomaticWeightedLoss import AutomaticWeightedLoss

    def str2bool(s):
        if s not in {'false', 'true'}:
            raise ValueError('Not a valid boolean string')
        return s == 'true'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)

    args = parser.parse_args()
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'),
              'w') as f:
        f.write('\n'.join([
            str(k) + ',' + str(v)
            for k, v in sorted(vars(args).items(), key=lambda x: x[0])
        ]))
    f.close()

    # dataset = data_partition('ml-1m')
    dataset = data_partition('ml-1m-new/ml-1m-item-f')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(
        user_train
    ) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    # ------------------------------ 添加category
    dataset1 = data_partition('ml-1m-new/ml-1m-cate-f++')
    # dataset1 = data_partition('ml-1m-new/ml-1m-item-f')
    [user_train1, user_valid1, user_test1, usernum1, itemnum1] = dataset1
    sampler1 = WarpSampler(user_train1,
                           usernum1,
                           itemnum1,
                           batch_size=args.batch_size,
                           maxlen=args.maxlen,
                           n_workers=2)

    # dataset2 = data_partition('user_cate_part++')
    # [user_train2, user_valid2, user_test2, usernum2, itemnum2] = dataset2
    # sampler2 = WarpSampler(user_train2,
    #                        usernum2,
    #                        itemnum2,
    #                        batch_size=args.batch_size,
    #                        maxlen=args.maxlen,
    #                        n_workers=2)

    # dataset3 = data_partition('user_shop_part++')
    # [user_train3, user_valid3, user_test3, usernum3, itemnum3] = dataset3
    # sampler3 = WarpSampler(user_train3,
    #                        usernum3,
    #                        itemnum3,
    #                        batch_size=args.batch_size,
    #                        maxlen=args.maxlen,
    #                        n_workers=2)
    # ------------------------------

    f = open(
        os.path.join(
            args.dataset + '_' + args.train_dir,
            '20220622_ml_maxlen=400.txt'), 'w')

    sampler = WarpSampler(user_train,
                          usernum,
                          itemnum,
                          batch_size=args.batch_size,
                          maxlen=args.maxlen,
                          n_workers=2)
    model = SASRec(usernum, itemnum, args).to(
        args.device)  # no ReLU activation in original SASRec implementation?

    for name, param in model.named_parameters():
        if name != 'item_emb.weight':
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass  # just ignore those failed init layers

    # this fails embedding init 'Embedding' object has no attribute 'dim'

    # model.apply(torch.nn.init.xavier_uniform_)

    model.train()  # enable model training

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(
                torch.load(args.state_dict_path,
                           map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') +
                                        6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print(
                'pdb enabled for your quick check, pls type exit() if you do not need it'
            )
            import pdb
            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()

    # 加入weight loss
    awl = AutomaticWeightedLoss(4)
    # adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    adam_optimizer = torch.optim.Adam([{
        'params': model.parameters()
    }, {
        'params': awl.parameters(),
        'weight_decay': 0
    }],
                                      lr=args.lr,
                                      betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition
        for step in range(
                num_batch
        ):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(
                pos), np.array(neg)
            # ---------------添加cate
            u1, seq1, pos1, neg1 = sampler1.next_batch()  # tuples to ndarray
            u1, seq1, pos1, neg1 = np.array(u1), np.array(seq1), np.array(
                pos1), np.array(neg1)

            # u2, seq2, pos2, neg2 = sampler2.next_batch()  # tuples to ndarray
            # u2, seq2, pos2, neg2 = np.array(u2), np.array(seq2), np.array(
            #     pos2), np.array(neg2)

            # u3, seq3, pos3, neg3 = sampler3.next_batch()  # tuples to ndarray
            # u3, seq3, pos3, neg3 = np.array(u3), np.array(seq3), np.array(
            #     pos3), np.array(neg3)
            # ---------------
            pos_logits, neg_logits, pos_logits1, neg_logits1 = model(
                u, seq, pos, neg, u1, seq1, pos1, neg1)

            pos_labels, neg_labels = torch.ones(
                pos_logits.shape,
                device=args.device), torch.zeros(neg_logits.shape,
                                                 device=args.device)
            pos_labels1, neg_labels1 = torch.ones(
                pos_logits1.shape,
                device=args.device), torch.zeros(neg_logits1.shape,
                                                 device=args.device)

            # pos_labels2, neg_labels2 = torch.ones(
            #     pos_logits2.shape,
            #     device=args.device), torch.zeros(neg_logits2.shape,
            #                                      device=args.device)

            # pos_labels3, neg_labels3 = torch.ones(
            #     pos_logits3.shape,
            #     device=args.device), torch.zeros(neg_logits3.shape,
            #                                      device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0

            indices = np.where(pos != 0)
            indices1 = np.where(pos1 != 0)
            # indices2 = np.where(pos2 != 0)
            # indices3 = np.where(pos3 != 0)

            loss1 = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss1 += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters():
                loss1 += args.l2_emb * torch.norm(param)

            loss2 = bce_criterion(pos_logits1[indices1], pos_labels1[indices1])
            loss2 += bce_criterion(neg_logits1[indices1],
                                   neg_labels1[indices1])
            for param in model.item_emb.parameters():
                loss2 += args.l2_emb * torch.norm(param)

            # loss3 = bce_criterion(pos_logits2[indices2], pos_labels2[indices2])
            # loss3 += bce_criterion(neg_logits2[indices2],
            #                        neg_labels2[indices2])
            # for param in model.item_emb.parameters():
            #     loss3 += args.l2_emb * torch.norm(param)

            # loss4 = bce_criterion(pos_logits3[indices3], pos_labels3[indices3])
            # loss4 += bce_criterion(neg_logits3[indices3],
            #                        neg_labels3[indices3])
            # for param in model.item_emb.parameters():
            #     loss4 += args.l2_emb * torch.norm(param)
            # loss = loss1 + loss2 + loss3 + loss4
            loss = awl(loss1, loss2)
            # loss = loss1+loss2
            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(
                epoch, step,
                loss.item()))  # expected 0.4~0.6 after init few epochs

        if epoch % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print(
                'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f,NDCG@5: %.4f, HR@5: %.4f), test (NDCG@10: %.4f, HR@10, %.4f,NDCG@5: %.4f, HR@5: %.4f)'
                % (epoch, T, t_valid[0], t_valid[1], t_valid[2], t_valid[3],
                   t_test[0], t_test[1], t_test[2], t_test[3]))

            print(
                'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f，NDCG@5: %.4f, HR@5: %.4f), test (NDCG@10: %.4f, HR@10: %.4f,NDCG@5: %.4f, HR@5: %.4f)'
                % (epoch, T, t_valid[0], t_valid[1], t_valid[2], t_valid[3],
                   t_test[0], t_test[1], t_test[2], t_test[3]),file=f)
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks,
                                 args.num_heads, args.hidden_units,
                                 args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")
