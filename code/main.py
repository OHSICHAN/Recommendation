import parse
from parse import args
from dataloader import dataset, MyDataset
from model import Model
import torch

def print_test_result(best_epoch, test_pre, test_recall, test_ndcg):
    print(f'Test Result(at {best_epoch:d} epoch):')
    for i, k in enumerate(parse.topks):
        print(f'ndcg@{k:d} = {test_ndcg[i]:f}, recall@{k:d} = {test_recall[i]:f}, pre@{k:d} = {test_pre[i]:f}.')

def train():
    train_loss = model.train_func()
    if epoch % args.show_loss_interval == 0:
        print(f'epoch {epoch:d}, train_loss = {train_loss:f}')

def valid(epoch):
    global best_valid_ndcg, best_epoch, test_pre, test_recall, test_ndcg
    valid_pre, valid_recall, valid_ndcg = model.valid_func()
    for i, k in enumerate(parse.topks):
        print(f'[{epoch:d}/{args.epochs:d}] Valid Result: ndcg@{k:d} = {valid_ndcg[i]:f}, recall@{k:d} = {valid_recall[i]:f}, pre@{k:d} = {valid_pre[i]:f}.')
    if valid_ndcg[-1] > best_valid_ndcg:
        best_valid_ndcg, best_epoch = valid_ndcg[-1], epoch
        test_pre, test_recall, test_ndcg = model.test_func()
        # best 결과를 여기서 출력하지 않고 변수로만 저장.
        return True
    return False

### MODIFICATION START ###
# 모든 hop에 대한 best 결과를 저장할 리스트(또는 dict)
all_hop_results = []
### MODIFICATION END ###

for hop_val in range(1, 12):
    print('===========================')
    print(f'Run for sample_hop = {hop_val}')
    print('===========================')
    args.sample_hop = hop_val
    # 새로운 dataset, model 로딩
    dataset = MyDataset(parse.train_file, parse.valid_file, parse.test_file, parse.device)
    model = Model(dataset).to(parse.device)

    # 베스트 결과를 위한 변수 초기화
    best_valid_ndcg, best_epoch = 0., 0
    test_pre, test_recall, test_ndcg = torch.zeros(len(args.topks)), torch.zeros(len(args.topks)), torch.zeros(len(args.topks))

    # 초기 validation
    valid(epoch=0)
    stop_flag = False
    for epoch in range(1, args.epochs+1):
        train()
        if epoch % args.valid_interval == 0:
            if not valid(epoch) and epoch - best_epoch >= args.stopping_step*args.valid_interval:
                stop_flag = True
                break
    print('---------------------------')

    # 각 hop마다 best 결과를 all_hop_results에 저장
    all_hop_results.append((hop_val, best_epoch, test_pre.clone(), test_recall.clone(), test_ndcg.clone()))

### MODIFICATION START ###
# 모든 hop(1~11) 종료 후 각 hop의 best 결과 한 번에 출력
print("******** ALL HOP RESULTS ********")
for hop_val, be_epoch, be_pre, be_recall, be_ndcg in all_hop_results:
    print(f'=== sample_hop = {hop_val} ===')
    print_test_result(be_epoch, be_pre, be_recall, be_ndcg)
print("*********************************")
### MODIFICATION END ###