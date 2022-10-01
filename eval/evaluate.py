import torch
import numpy as np
from eval import metrics
import gc
import copy

def evaluate(model, data, device, mode):
    """ evaluate model on recommending items to groups """
    eval_loss = 0.0
    ndcg10_list, hits10_list = [], []


    model.eval()
    with torch.no_grad():
        data = copy.copy(data).to(device)       
        heldout_data = data['group'].val_y 
        if mode == 'test':
            heldout_data = data['group'].test_y

        eval_group_idx = torch.unique(heldout_data.nonzero(as_tuple=True)[0])

        out = model(data.x_dict, data.edge_index_dict)
        result = out['group'].softmax(1)
        mask = data['group'].y.nonzero()
        result[mask[:,0], mask[:,1]] = -np.inf

        result = result[eval_group_idx]
        heldout_data = heldout_data[eval_group_idx]


        
        hits10 = metrics.hits_at_k_batch_torch(result, heldout_data, 10)
        ndcg10 = metrics.ndcg_binary_at_k_batch_torch(result, heldout_data, 10, device=device)

        ndcg10_list.append(ndcg10)
        hits10_list.append(hits10)

        del data
    gc.collect()

    ndcg10_list = torch.cat(ndcg10_list)
    hits10_list = torch.cat(hits10_list)
    return eval_loss, torch.mean(ndcg10_list), torch.sum(hits10_list)/eval_group_idx.shape[0]

