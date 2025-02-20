#Pytorch
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
#import torch_geometric.utils as utils

import numpy as np

#Python Libs
import time
import gc
import argparse
import logging
import math

#Implementations
from models.models import GraphGR
from utils.load_csv import GroupDataset
from eval.evaluate import evaluate

# Logging
logger = logging.getLogger(__name__)
logger.propagate = 0
#('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] >> %(message)s', '%Y-%m-%d %H:%M:%S')

streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)

fileHandler = logging.FileHandler('logs/train.log')
fileHandler.setFormatter(formatter)

logger.addHandler(streamHandler)
logger.addHandler(fileHandler)

logger.setLevel(level=logging.DEBUG)


# Parsing
parser = argparse.ArgumentParser(description='PyTorch GraphGR')
parser.add_argument('--dataset', type=str, default='meetupCA', help='Name of dataset')

# Training settings.
parser.add_argument('--lr', type=float, default=5e-3, help='initial learning rate')
parser.add_argument('--drop_rate', type=float, default=0.2, help='Dropout ratio')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', type=int, default=251, help='maximum # training epochs')
parser.add_argument('--eval_freq', type=int, default=5, help='frequency to evaluate performance on validation set')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay coefficient')
parser.add_argument('--lam1', type=float, default=1.0, help='hyperparameter for reconstruction error')
parser.add_argument('--lam2', type=float, default=1.0, help='hyperparameter for reconstruction error')

# Model settings.
parser.add_argument('--emb_size', type=int, default=128, help='layer size')
parser.add_argument('--gnn_type', type=str, default='SAGE', help='choice of graph neural network',
                    choices=['GATv2', 'SAGE', 'GAT'])

parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--seed', type=int, default=2023, help='random seed for reproducibility')

# Model save file parameters.
parser.add_argument('--save', type=str, default='weights/model.pt', help='path to save the final model')

args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
    if not args.cuda:
        logger.warning("WARNING: You have a CUDA device, so you should probably run with --cuda")

###############################################################################
# Load data
###############################################################################

device = torch.device("cuda" if args.cuda else "cpu")

logger.info('-' * 89)
logger.info(args)
group_dataset = GroupDataset(args.dataset)
data = group_dataset.graphdata
#logger.info('-' * 89)
#logger.info(data)
logger.info('-' * 89)

num_node_info = {'user': group_dataset.n_users,
            'item': group_dataset.n_items,
            'group': group_dataset.n_groups}


model = GraphGR(args.emb_size, data.metadata(), num_node_info, gnn_type=args.gnn_type, device=device, drop_rate=args.drop_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
loss_fn = torch.nn.CrossEntropyLoss()

# Embedding initialize


group_train_loader = NeighborLoader(
    data,
    # Sample 15 neighbors for each node and each edge type for 2 iterations:
    num_neighbors=[20] * 2,
    # Use a batch size of 128 for sampling training nodes of type "paper":
    batch_size=args.batch_size,
    input_nodes=('group', data['group'].is_train), 
)


try:
    best_hits10 = -np.inf
    best_ndcg = -np.inf
    total_loss_list = []
    L_combined_loss_list = []
    for epoch in range(0, args.epochs):
        epoch_start_time = time.time()
        model.train()
        total_examples, total_loss, L_combined_loss = 0, 0, 0


        kd_coef = 0 if epoch < 50 else args.lam1
        kd_coef2 = 0 if epoch < 50 else args.lam2

        # mini-batch
        for batch in group_train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)

            batch_size = batch['group'].batch_size
            out_tea, out_aug, kd_loss, kd_loss2 = model(batch.x_dict, batch.edge_index_dict )

            
            #loss = -torch.mean(torch.sum(F.log_softmax(out_tea['group'][:batch_size], 1) * batch['group'].y[:batch_size], -1))
            loss = loss_fn(out_tea['group'][:batch_size], batch['group'].y[:batch_size])

            #loss += -torch.mean(torch.sum(F.log_softmax(out_tea['user'][:batch_size], 1) * batch['user'].y[:batch_size], -1))
            #loss += -torch.mean(torch.sum(F.log_softmax(out_aug['group'][:batch_size], 1) * batch['group'].y[:batch_size], -1))   
            loss += loss_fn(out_aug['group'][:batch_size], batch['group'].y[:batch_size])
            L_combined_loss += 0.5*(loss.item())         
            #loss += -torch.mean(torch.sum(F.log_softmax(out_aug['user'][:batch_size], 1) * batch['user'].y[:batch_size], -1))
            loss += kd_coef*kd_loss + kd_coef2*kd_loss2

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            total_examples += batch_size
            

            del batch
        gc.collect()

        logger.info(f"\tTrain loss: {total_loss / total_examples:.4f}, Combined loss: {L_combined_loss / total_examples:.4f}")
        total_loss_list.append(total_loss / total_examples)
        L_combined_loss_list.append(L_combined_loss / total_examples)

        if epoch % args.eval_freq == 0:
            # Group evaluation.
            val_loss_group, ndcg10_group, hits10_group = evaluate(model, data, device, 'val')

            logger.info('-' * 89)
            logger.info('\t> End of epoch {:3d} | time: {:4.2f}s | ndcg10 (group) {:5.4f} | hits10 (group) {:5.4f} '
                .format(epoch + 1, time.time() - epoch_start_time, ndcg10_group, hits10_group))
            logger.info('-' * 89)

            # Save the model if the hits10 is the best we've seen so far.
            if (hits10_group > best_hits10) or (hits10_group == best_hits10 and ndcg10_group > best_ndcg):
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                    logger.info('\t> Saved')
                best_hits10 = hits10_group
                best_ndcg = ndcg10_group
    logger.info(f"{total_loss_list}")
    logger.info(f"{L_combined_loss_list}")

except KeyboardInterrupt:
    logger.info('-' * 89)
    logger.warning('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f, map_location=device)
    model = model.to(device)

eval_start_time = time.time()

# Valid data
_, ndcg10_group, hits10_group = evaluate(model, data, device, 'val')
logger.info('-' * 89)
logger.info(args)
logger.info('\t> Valid result: ndcg10 (group) {:5.4f} | hits10 (group) {:5.4f} '
    .format(ndcg10_group, hits10_group))

data = group_dataset.get_test_data()
#logger.info('[Test data]')
#logger.info(data)

'''
# Embedding initialize
unseen_group_idx = torch.unique(data['group'].test_y.nonzero(as_tuple=True)[0])
seen_group_idx = torch.unique(data['group'].y.nonzero(as_tuple=True)[0])
with torch.no_grad():
    avg_vec = torch.mean((model.emb['group'].weight)[seen_group_idx], dim=0)
    for idx in unseen_group_idx:
        (model.emb['group'].weight)[idx] = avg_vec
'''

# Test data
_, ndcg10_group, hits10_group = evaluate(model, data, device, 'test')
logger.info('\t> Test result: ndcg10 (group) {:5.4f} | hits10 (group) {:5.4f} '
    .format(ndcg10_group, hits10_group))
logger.info('-' * 89)
#logger.info('\t> End of eval | time: {:4.2f}s '.format( time.time() - eval_start_time))

