import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer

from sklearn.linear_model import LinearRegression


tokenizer = AutoTokenizer.from_pretrained('roneneldan/TinyStories-33M')

def enc(*s):
    if isinstance(s, str):
        s = [s]
    toks = [tokenizer.encode(s) for s in s]
    for t in toks:
        assert len(t) == 1, [tokenizer.decode([_t]) for _t in t]
    toks = torch.tensor([t[0] for t in toks])
    return toks

def preprocess_feats(feats, allow_nones=False):
    return [Match(feat) if isinstance(feat, str) else feat for feat in feats if (feat is not None or allow_nones)]


class Match(nn.Module):
    def __init__(self, *match_toks):
        super().__init__()
        if isinstance(match_toks[0], str):
            match_toks = enc(*match_toks)
        else:
            assert isinstance(match_toks[0], int) or isinstance(match_toks[0], np.int_), print(type(match_toks[0]))
        self.match_toks = torch.tensor(list(set(match_toks)))
        self.weight = nn.Parameter(torch.tensor(torch.ones(1,)))
    def get_feature(self, doc_ids):
        return torch.isin(doc_ids, self.match_toks).int()
    def __add__(self, match):
        return Match(*torch.cat([self.match_toks, match.match_toks]).numpy().tolist())
    def __contains__(self, tok):
        return enc(tok)[0] in self.match_toks
    def forward(self, doc_ids):
        matches = self.get_feature(doc_ids)
        return matches*self.weight
    

class And(nn.Module):
    def __init__(self, *feats):
        super().__init__()
        self.feats = preprocess_feats(feats)
        self.weight = nn.Parameter(torch.tensor(torch.ones(1,)))
    def get_feature(self, doc_ids):
        # feats = self.stacked(doc_ids)
        feats = torch.stack([feat.get_feature(doc_ids) for feat in self.feats], dim=-1)
        res = torch.prod(feats, dim=-1)
        return res
    def forward(self, doc_ids):
        feat_out = self.get_feature(doc_ids)
        return feat_out*self.weight
    

class Prod(nn.Module):
    def __init__(self, *feats):
        super().__init__()
        self.feats = preprocess_feats(feats)
    def forward(self, doc_ids):
        feats = torch.stack([feat(doc_ids) for feat in self.feats], dim=-1)
        res = torch.prod(feats, dim=-1)
        return res


class Sum(nn.Module):
    def __init__(self, *feats):
        super().__init__()
        self.feats = preprocess_feats(feats)
    def forward(self, doc_ids):
        feats = torch.stack([feat(doc_ids) for feat in self.feats], dim=-1)
        res = torch.sum(feats, dim=-1)
        return res


class Or(nn.Module):
    def __init__(self, *feats):
        super().__init__()
        feats = preprocess_feats(feats)
        matches = [feat for feat in feats if isinstance(feat, Match)]
        non_matches = [feat for feat in feats if not isinstance(feat, Match)]

        feats = non_matches
        if matches:
            total_match = matches[0]
            for match in matches[1:]:
                total_match += match
            feats += [total_match]
        
        self.feats = feats
        self.weight = nn.Parameter(torch.tensor(torch.ones(1,)))

    def get_feature(self, doc_ids):
        feats = torch.stack([feat.get_feature(doc_ids) for feat in self.feats], dim=-1)
        res = torch.max(feats, dim=-1).values
        return res
    def forward(self, doc_ids):
        return self.get_feature(doc_ids)*self.weight

# def ElasticInterval(k):
#     return Seq(
#         *[
#             Optional(Anything()) for _ in range(k)
#         ]
#     )

# def FixedInterval(k):
#     assert k > 0
#     return Seq(
#         *[
#             Anything() for _ in range(k)
#         ]
#     )


# def CasedInterval(k):
#     if k == 0:
#         return None
#     else:
#         return Cases(
#             *(
#                 [None]
#                +[FixedInterval(i) for i in range(1,k+1)]
#             )
#         )


class Anything(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1,))
    def get_feature(self, doc_ids):
        return torch.ones(doc_ids.shape).long()
    def forward(self, doc_ids):
        return self.get_feature(doc_ids)



# Cases[x, y, z] ...Distances[1,2,3]... Cases[a, b, c]

# Relu(Cases[x, y, z]*Distances[1,2,3]*Match[a,b,c]


def find_duplicates(l):
    dupes = []
    while l:
        head, tail = l[0], l[1:]
        if head in tail:
            dupes.append(head)
        l = tail
    return dupes

class Cases:
    def __init__(self, *feats):
        assert len(set(feats)) == len(feats), f'Each case entry should be unique. Duplicates: {find_duplicates(feats)}'
        self.feats = preprocess_feats(feats, allow_nones=True)
    def __call__(self, *args, **kwargs):
        assert False, 'Cases object should not be called directly. It should be an argument for a Stack or Sequence object.'


class Optional:
    def __init__(self, feature_fn):
        self.feature_fn = feature_fn if not isinstance(feature_fn, str) else Match(feature_fn)
    def __call__(self, doc_ids):
        return self.feature_fn(doc_ids)
    
def a_then_b(feat_a, feat_b, optional_b=False):
    assert len(feat_a.shape) == 2
    assert len(feat_b.shape) == 2
    # indices that match a then b
    feat_ab = feat_a[:,:-1]*feat_b[:,1:]
    # fill in the beginning because the resultant tensor is a little too short
    feat_ab = torch.cat((torch.zeros(feat_a.shape[0], 1).int(), res), dim=-1)
    # if optional_b, return OR(ab, a)
    if optional_b:
        res = (feat_ab + feat_a).clamp(max=1)
    else:
        res = feat_ab
    return res

class Seq(nn.Module):
    def __init__(self, *feats):
        super().__init__()
        feats = preprocess_feats(feats)
        final_feats = []
        for feat in feats:
            if isinstance(feat, Seq):
                final_feats.extend(feat.feats)
            else:
                final_feats.append(feat)
        self.feats = final_feats
        self.weight = nn.Parameter(torch.ones(1,))
    
    def __new__(cls, *feats, **kwargs):
        feats = preprocess_feats(feats)
        for i, feat in enumerate(feats):
            if isinstance(feat, Cases):
                case_feats = feat.feats
                return Sum(
                    *[
                        Seq(*(feats[:i] + [case_feat] + feats[i+1:])) for case_feat in case_feats 
                    ]
                )
        instance = super().__new__(cls)
        return instance
    
    def get_feature(self, doc_ids):
        if isinstance(self.feats[0], Optional):
            assert False, 'An Optional feature cannot be the first part of a sequence.'
        assert len(doc_ids.shape) == 2
        feats = torch.stack([feat(doc_ids) for feat in self.feats], dim=-1)

        # in parallel: does this token match the first token of the sequence
        valid_start = feats[:,:,0]
         
        for i in range(1, feats.shape[-1]):
            # calculate which tokens match the i+1 feature start subsequence of the sequence
            is_optional = isinstance(self.feats[i], Optional)
            valid_start = a_then_b(valid_start, feats[:,:,i], optional_b=is_optional)
                
        return valid_start
    def forward(self, doc_ids):
        return self.get_feature(doc_ids)*self.weight


class Not:
    def __init__(self, feat):
        self.not_feat = preprocess_feats([feat])[0]
        self.weight = nn.Parameter(torch.ones(1,))
    def get_feature(self, doc_ids):
        feat = self.not_feat.get_feature(doc_ids)
        return torch.ones_like(doc_ids).int() - feat
    def forward(self, doc_ids):
        return self.get_feature(doc_ids)*self.weight
    # def __repr__(self, ..)
        # child.__repr__

# Not: 1.0
#     some_feature: 0.3

# Not: 0.3
#     some_feature



# class Stack:
#     def __init__(self, *input_feats):
#         input_feats = preprocess_feats(input_feats)
#         feats = []
#         for feat in input_feats:
#             if isinstance(feat, Stack):
#                 feats.extend(feat.feats)
#             else:
#                 feats.append(feat)
#         self.feats = feats

#     def __call__(self, doc_ids):
#         feat_outputs = [feat(doc_ids) for feat in self.feats]
#         for i, feat in enumerate(feat_outputs):
#             if isinstance(feat, np.ndarray):
#                 feat_outputs[i] = torch.tensor(feat)
#             if len(feat.shape) == 2:
#                 feat_outputs[i] = (feat_outputs[i])[:,:,None]
#         return torch.cat(feat_outputs, dim=-1)


def pred_feature(feature_acts, feats_fn, doc_ids):
    '''
    feature_acts: activations of feature inside the transformer (from sparse mlp/autoencoder)
    feats_fn: featurekit Stack fn
    doc_ids: list of lists of token ids (list of docs)

    Returns
        reg_weights: weights of linear regression on feats_fn output
        reg_bias: bias of linear regression on feats_fn output
        pred: predictions of feats_fn combined with linear regression on doc_ids
    '''
    def get_weights(feats_fn, doc_ids):
        features = feats_fn(doc_ids)
        return features
    feature_acts = feature_acts/feature_acts.max()
    out = get_weights(feats_fn, doc_ids)
    flat_input = out.reshape(-1, out.shape[-1])
    flat_feature_acts = feature_acts.reshape(-1)
    reg = LinearRegression().fit(flat_input, flat_feature_acts)
    reg_weights = reg.coef_
    reg_bias = reg.intercept_
    pred = torch.tensor(reg.predict(flat_input).reshape(feature_acts.shape))
    return reg_weights, reg_bias, pred


def compute_weights_and_pred(X, y):
    """
    Computes the weights of a linear regression model and predictions based on input features.

    Parameters:
    X (torch.Tensor): The matrix of input features.
    y (torch.Tensor): The vector of output (target) values.

    The function adds a bias term to the input features, computes the weights using
    the least squares method, and then calculates predictions based on these weights.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: A tuple containing two elements:
        - weights (torch.Tensor): The weights of the linear regression model, including the bias term.
        - pred (torch.Tensor): The predictions made by the model.
    """
    X_with_bias = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)
    weights = torch.inverse(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    pred = (weights @ X_with_bias.T)
    return weights, pred

def pred_feature_fast(feature_acts, feats_fn, doc_ids):
    """
    Performs feature extraction and linear regression to predict feature activations.

    Parameters:
    feature_acts (torch.Tensor): Activations of features inside a transformer, such as from a sparse MLP or autoencoder.
    feats_fn (Callable): A feature extraction function, typically a 'featurekit Stack' function.
    doc_ids (List[List[int]]): A list of lists containing token IDs, representing a list of documents.

    The function normalizes feature activations, applies the feature extraction function,
    and uses linear regression to predict the feature activations based on the extracted features.

    Returns:
    Tuple[np.ndarray, float, torch.Tensor]: A tuple containing three elements:
        - reg_weights (np.ndarray): The weights of the linear regression model, excluding the bias term.
        - reg_bias (float): The bias term of the linear regression model.
        - pred (torch.Tensor): The predictions of feature activations.
    """
    feature_acts = feature_acts / feature_acts.max()
    out = feats_fn(doc_ids)
    if out.max() == out.min(): # Anything() mask
        reg_weights = np.array([0.])
        reg_bias = feature_acts.mean().item()
        pred = torch.full(feature_acts.shape, reg_bias)
    else:
        flat_input = out.reshape(-1, out.shape[-1])
        flat_feature_acts = feature_acts.reshape(-1)
        X = out.reshape(-1, out.shape[-1])
        y = feature_acts.reshape(-1)
        weights, pred = compute_weights_and_pred(X, y)
        reg_weights = np.array(weights[1:])
        reg_bias = float(weights[0])
        pred = pred.reshape(feature_acts.shape).to(dtype=torch.float64)
    return reg_weights, reg_bias, pred

