#Pytorch
import torch
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader

#Python Libs
import time
import gc
import argparse
import logging

#Implementations
from models.models import ConvGR
from utils.load_csv import GroupDataset
from eval.evaluate import evaluate

# Logging
logger = logging.getLogger(__name__)
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
parser = argparse.ArgumentParser(description='PyTorch ConvGR: Convolutional Group Recommendation')
parser.add_argument('--dataset', type=str, default='gwl', help='Name of dataset')

# Training settings.
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay coefficient')
parser.add_argument('--drop_ratio', type=float, default=0.4, help='Dropout ratio')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='maximum # training epochs')
parser.add_argument('--eval_freq', type=int, default=5, help='frequency to evaluate performance on validation set')

# Model settings.
parser.add_argument('--emb_size', type=int, default=64, help='layer size')
parser.add_argument('--gnn_type', type=str, default='GAT', help='choice of graph neural network',
                    choices=['GCN', 'SAGE', 'GAT'])

parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--seed', type=int, default=2022, help='random seed for reproducibility')


#args = parser.parse_args()
args_str = '--dataset gwl --epochs 200 --gnn_type GAT --cuda'
args, _ = parser.parse_known_args(args=args_str.split())

torch.manual_seed(args.seed)  # Set the random seed manually for reproducibility.

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
logger.info('-' * 89)
logger.info(data)
logger.info('-' * 89)

num_node_info = {'user': group_dataset.n_users,
            'item': group_dataset.n_items,
            'group': group_dataset.n_groups}


model = ConvGR(args.emb_size, data.metadata(), num_node_info, gnn_type=args.gnn_type, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

'''
with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)
'''

group_train_loader = NeighborLoader(
    data,
    # Sample 15 neighbors for each node and each edge type for 2 iterations:
    num_neighbors=[20] * 2,
    # Use a batch size of 128 for sampling training nodes of type "paper":
    batch_size=args.batch_size,
    input_nodes=('group'),
)

try:
    for epoch in range(0, args.epochs):
        epoch_start_time = time.time()
        model.train()
        total_examples = total_loss = 0

        for batch in group_train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)

            batch_size = batch['group'].batch_size
            out = model(batch.x_dict, batch.edge_index_dict)

            group_loss = -torch.mean(torch.sum(F.log_softmax(out['group'][:batch_size], 1) * batch['group'].y[:batch_size], -1))
            user_loss = -torch.mean(torch.sum(F.log_softmax(out['user'][:batch_size], 1) * batch['user'].y[:batch_size], -1))
            loss = group_loss + user_loss
            loss.backward()
            optimizer.step()

            total_examples += batch_size
            total_loss += float(loss) 

            del batch


        gc.collect()


        logger.info("\tTrain loss: {:.4f}".format(total_loss / total_examples))
        
        if epoch % args.eval_freq == 0:
            # Group evaluation.
            val_loss_group, ndcg10_group, hits10_group = evaluate(model, data, device, 'val')

            logger.info('-' * 89)
            logger.info('\t> End of epoch {:3d} | time: {:4.2f}s | ndcg10 (group) {:5.4f} | hits10 (group) {:5.4f} '
                .format(epoch + 1, time.time() - epoch_start_time, ndcg10_group, hits10_group))
            logger.info('-' * 89)

            # Save the model if the hits10 is the best we've seen so far.


except KeyboardInterrupt:
    logger.info('-' * 89)
    logger.warning('Exiting from training early')

data = group_dataset.get_test_data()
logger.info('[Test data]')
logger.info(data)
test_loss_group, ndcg10_group, hits10_group = evaluate(model, data, device, 'test')

logger.info('-' * 89)
logger.info('\t> Test result: ndcg10 (group) {:5.4f} | hits10 (group) {:5.4f} '
    .format(ndcg10_group, hits10_group))
logger.info('-' * 89)