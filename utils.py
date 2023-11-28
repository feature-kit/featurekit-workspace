import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
from feature_kit.feature_components import Stack, compute_weights_and_pred

def compute_mse_loss_only(doc_ids, acts, feature_idx, token_strings):
    """
    Computes the mean squared error (MSE) loss of a linear model using token_strings as features. 
    
    Parameters:
    doc_ids (torch.Tensor): Tensor of token ids of size (num_docs, num_tokens_per_doc)
    acts (torch.Tensor): Tensor of activations of size (num_features, num_docs, num_tokens_per_doc)
    feature_idx (int): Index of the specific feature to consider.
    token_strings (list of str): List of tokens to consider. If empty, a baseline mse is computed. 
    
    Returns:
    float: The computed MSE loss.
    """
    # assumes out.max() != out.min(), that is, no uniform features.
    feature_acts = acts[feature_idx]/acts[feature_idx].max()
    if token_strings:
        feats_fn = Stack(*token_strings)
        out = feats_fn(doc_ids)
    if not token_strings:
        return (feature_acts ** 2).mean().item()
    else:
        flat_input = out.reshape(-1, out.shape[-1])
        flat_feature_acts = feature_acts.reshape(-1)
        X = out.reshape(-1, out.shape[-1])
        y = feature_acts.reshape(-1)
        weights, pred = compute_weights_and_pred(X, y)
        pred = pred.reshape(feature_acts.shape)
        return ((feature_acts-pred)**2).mean().item()

def correlation(tensor1, tensor2):
    """
    Computes the Pearson correlation coefficient between two 1D tensors.
    
    Parameters:
    tensor1 (torch.Tensor): The first tensor.
    tensor2 (torch.Tensor): The second tensor.
    
    Returns:
    float: The Pearson correlation coefficient.
    """
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()
    
    mean1 = torch.mean(tensor1)
    mean2 = torch.mean(tensor2)
    
    deviation1 = tensor1 - mean1
    deviation2 = tensor2 - mean2
    
    numerator = torch.sum(deviation1 * deviation2)
    denominator = torch.sqrt(torch.sum(deviation1 ** 2) * torch.sum(deviation2 ** 2))
    
    correlation = numerator / denominator
    
    return correlation.item()

def sort_tokens_by_abs_correlation(doc_ids, acts, feature_idx, model_name='roneneldan/TinyStories-33M'):
    """
    Sorts tokens in a document by the absolute value of their correlation with a specified feature.
    
    Parameters:
    doc_ids (torch.Tensor): Tensor of token ids of size (num_docs, num_tokens_per_doc)
    acts (torch.Tensor): Tensor of activations of size (num_features, num_docs, num_tokens_per_doc)
    feature_idx (int): Index of the feature to correlate with.
    model_name (str): Name of the model to use for tokenization.
    
    Returns:
    list: Sorted list of tokens based on the absolute correlation with the specified feature.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    unique_ids = set(doc_ids.flatten().tolist())
    flat_acts = (acts[feature_idx]/acts[feature_idx].max()).flatten()
    corr = dict()
    for tok_id in tqdm(unique_ids, desc='Computing Correlations'):
        token = tokenizer.decode([tok_id])
        flat_token_feature = (doc_ids == tok_id).to(int).flatten()
        corr[token] = correlation(flat_token_feature, flat_acts)
    return sorted(corr.keys(), key=lambda x: -abs(corr[x]))

def get_n_most_correlated_tokens(doc_ids, acts, feature_idx, n=32, model_name='roneneldan/TinyStories-33M'):
    """
    Retrieves the N most correlated tokens with a specific feature.
    
    Parameters:
    doc_ids (torch.Tensor): Tensor of token ids of size (num_docs, num_tokens_per_doc)
    acts (torch.Tensor): Tensor of activations of size (num_features, num_docs, num_tokens_per_doc)
    feature_idx (int): Index of the to correlate with.
    n (int): Number of top correlated tokens to retrieve.
    model_name (str): Name of the model to use for tokenization.
        
    Returns:
    list: List of the top N most correlated tokens with the specified feature.
    """
    return sort_tokens_by_abs_correlation(doc_ids, acts, feature_idx, model_name)[:n]

def plot_mse_loss(doc_ids, acts, feature_idx, max_tokens):
    """
    Plots the MSE loss against the number of tokens, for a specific feature index.
    
    Parameters:
    doc_ids (torch.Tensor): Tensor of token ids of size (num_docs, num_tokens_per_doc)
    acts (torch.Tensor): Tensor of activations of size (num_features, num_docs, num_tokens_per_doc)    
    feature_idx (int): Index of the feature to plot.
    max_tokens (int): Maximum number of tokens to consider for plotting.
    
    Produces:
    A log-log plot of MSE loss versus the number of tokens for the specified feature.
    """
    toks = sort_tokens_by_abs_correlation(doc_ids, acts, feature_idx)
    token_range = list(range(1, max_tokens))
    mse_losses = [compute_mse_loss_only(doc_ids, acts, feature_idx, toks[:n]) for n in tqdm(token_range, desc='Computing MSE Loss')]
    plt.figure(figsize=(10, 6))
    plt.loglog(token_range, mse_losses, marker='o', color='b', label='MSE Loss')
    plt.title(f'Feature {feature_idx} | MSE Loss vs. Number of Tokens')
    plt.xlabel('Number of Tokens')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
