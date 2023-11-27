import torch
import numpy as np
import torch

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

class Match:
    def __init__(self, *match_toks):
        if isinstance(match_toks[0], str):
            match_toks = enc(*match_toks)
        else:
            assert isinstance(match_toks[0], int) or isinstance(match_toks[0], np.int_), print(type(match_toks[0]))
        self.match_toks = torch.tensor(list(set(match_toks)))
    def get_matches(self, doc_ids):        
        return torch.isin(doc_ids, self.match_toks).int()
    def __call__(self, doc_ids):
        return self.get_matches(doc_ids)
    def __add__(self, match):
        return Match(*torch.cat([self.match_toks, match.match_toks]).numpy().tolist())
    def __contains__(self, tok):
        return enc(tok)[0] in self.match_toks

class And:
    def __init__(self, *feats):
        self.feats = preprocess_feats(feats)
        self.stacked = Stack(*feats)
    def __call__(self, doc_ids):
        feats = self.stacked(doc_ids)
        res = torch.prod(feats, dim=-1)
        return res

class Or:
    def __init__(self, *feats):
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

        self.stacked = Stack(*feats)
    def __call__(self, doc_ids):
        feats = self.stacked(doc_ids)
        res = torch.sum(feats, dim=-1).clamp(0, 1)
        return res

# class KBehind:
#     def __init__(self, match, k=4):
#         self.match = Match(match)
#         self.k = k
#     def __call__(self, doc_ids):
#         match_vals = self.match(doc_ids)
#         max_pool = F.max_pool1d(match_vals[None].float(), kernel_size=self.k, stride=1, padding=0)[0]//1
#         res = torch.cat([torch.zeros(self.k), max_pool[:-1]])
#         # zeros(0, k+1) join max_pool[:-1] join 
#         return res

def ElasticInterval(k):
    return Seq(
        *[
            Optional(Anything()) for _ in range(k)
        ]
    )

def FixedInterval(k):
    assert k > 0
    return Seq(
        *[
            Anything() for _ in range(k)
        ]
    )

# class Null:
#     def __init__(self):
#         pass
#     def __call__(self, *args, **kwargs):
#         assert False, 'Null feature must be used as argument to Sequence. It cannot be called.'

def CasedInterval(k):
    if k == 0:
        return None
    else:
        return Cases(
            *(
                [None]
               +[FixedInterval(i) for i in range(1,k+1)]
            )
        )


class Anything:
    def __init__(self):
        pass
    def __call__(self, doc_ids):
        return torch.ones(doc_ids.shape).long()

class Optional:
    def __init__(self, feature_fn):
        self.feature_fn = feature_fn if not isinstance(feature_fn, str) else Match(feature_fn)
    def __call__(self, doc_ids):
        return self.feature_fn(doc_ids)
    
def a_then_b(feat_a, feat_b, optional_b=False):
    assert len(feat_a.shape) == 2
    assert len(feat_b.shape) == 2
    res = feat_b[:,1:]*feat_a[:,:-1]
    res = torch.cat((torch.zeros(feat_a.shape[0], 1).int(), res), dim=-1)
    if optional_b:
        res = (res + feat_a).clamp(max=1)
    return res

class Seq:
    def __init__(self, *feats):
        feats = preprocess_feats(feats)
        final_feats = []
        for feat in feats:
            if isinstance(feat, Seq):
                final_feats.extend(feat.feats)
            else:
                final_feats.append(feat)
        self.feats = final_feats
        
        self.stacked = Stack(*self.feats)
    
    def __new__(cls, *feats, **kwargs):
        feats = preprocess_feats(feats)
        for i, feat in enumerate(feats):
            if isinstance(feat, Cases):
                case_feats = feat.feats
                return Stack(
                    *[
                        Seq(*(feats[:i] + [case_feat] + feats[i+1:])) for case_feat in case_feats 
                    ]
                )
        instance = super().__new__(cls)
        return instance


    def __call__(self, doc_ids):
        if isinstance(self.feats[0], Optional):
            assert False, 'An Optional feature cannot be the first part of a sequence.'
        assert len(doc_ids.shape) == 2
        feats = self.stacked(doc_ids)
        valid_start = feats[:,:,0]
        for i in range(1, feats.shape[-1]):
            is_optional = isinstance(self.stacked.feats[i], Optional)
            valid_start = a_then_b(valid_start, feats[:,:,i], optional_b=is_optional)
                
        return valid_start

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


class Not:
    def __init__(self, *feats):
        self.not_feat = Or(
            *preprocess_feats(feats)
        )
    def __call__(self, doc_ids):
        feat = self.not_feat(doc_ids)
        return torch.ones_like(doc_ids).int() - feat


class Stack:
    def __init__(self, *input_feats):
        input_feats = preprocess_feats(input_feats)
        feats = []
        for feat in input_feats:
            if isinstance(feat, Stack):
                feats.extend(feat.feats)
            else:
                feats.append(feat)
        self.feats = feats

    def __call__(self, doc_ids):
        feat_outputs = [feat(doc_ids) for feat in self.feats]
        for i, feat in enumerate(feat_outputs):
            if isinstance(feat, np.ndarray):
                feat_outputs[i] = torch.tensor(feat)
            if len(feat.shape) == 2:
                feat_outputs[i] = (feat_outputs[i])[:,:,None]
        return torch.cat(feat_outputs, dim=-1)


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

