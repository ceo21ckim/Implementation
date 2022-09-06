import numpy as np
import math 
import torch 
import torch.nn as nn 

from torch.nn import CrossEntropyLoss

def logsumexp(tensor, dim=-1, keepdim=False):
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()

def viterbi_decode(tag_sequence, transition_matrix,
                   tag_observations=None, allowed_start_transitions=None,
                   allowed_end_transitions=None):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.
    Parameters
    ----------
    tag_sequence : torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    tag_observations : Optional[List[int]], optional, (default = None)
        A list of length ``sequence_length`` containing the class ids of observed
        elements in the sequence, with unobserved elements being set to -1. Note that
        it is possible to provide evidence which results in degenerate labelings if
        the sequences of tags you provide as evidence cannot transition between each
        other, or those transitions are extremely unlikely. In this situation we log a
        warning, but the responsibility for providing self-consistent evidence ultimately
        lies with the user.
    allowed_start_transitions : torch.Tensor, optional, (default = None)
        An optional tensor of shape (num_tags,) describing which tags the START token
        may transition *to*. If provided, additional transition constraints will be used for
        determining the start element of the sequence.
    allowed_end_transitions : torch.Tensor, optional, (default = None)
        An optional tensor of shape (num_tags,) describing which tags may transition *to* the
        end tag. If provided, additional transition constraints will be used for determining
        the end element of the sequence.
    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : torch.Tensor
        The score of the viterbi path.
    """
    sequence_length, num_tags = list(tag_sequence.size())

    has_start_end_restrictions = allowed_end_transitions is not None or allowed_start_transitions is not None

    if has_start_end_restrictions:

        if allowed_end_transitions is None:
            allowed_end_transitions = torch.zeros(num_tags)
        if allowed_start_transitions is None:
            allowed_start_transitions = torch.zeros(num_tags)

        num_tags = num_tags + 2
        new_transition_matrix = torch.zeros(num_tags, num_tags)
        new_transition_matrix[:-2, :-2] = transition_matrix

        # Start and end transitions are fully defined, but cannot transition between each other.
        # pylint: disable=not-callable
        allowed_start_transitions = torch.cat([allowed_start_transitions, torch.tensor([-math.inf, -math.inf])])
        allowed_end_transitions = torch.cat([allowed_end_transitions, torch.tensor([-math.inf, -math.inf])])
        # pylint: enable=not-callable

        # First define how we may transition FROM the start and end tags.
        new_transition_matrix[-2, :] = allowed_start_transitions
        # We cannot transition from the end tag to any tag.
        new_transition_matrix[-1, :] = -math.inf

        new_transition_matrix[:, -1] = allowed_end_transitions
        # We cannot transition to the start tag from any tag.
        new_transition_matrix[:, -2] = -math.inf

        transition_matrix = new_transition_matrix

    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise Exception("Observations were provided, but they were not the same length "
                                     "as the sequence. Found sequence of length: {} and evidence: {}"
                                     .format(sequence_length, tag_observations))
    else:
        tag_observations = [-1 for _ in range(sequence_length)]


    if has_start_end_restrictions:
        tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
        zero_sentinel = torch.zeros(1, num_tags)
        extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
        tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
        tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
        sequence_length = tag_sequence.size(0)

    path_scores = []
    path_indices = []

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.
        path_scores.append(one_hot)
    else:
        path_scores.append(tag_sequence[0, :])

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transition_matrix
        scores, paths = torch.max(summed_potentials, 0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.

        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.
            path_scores.append(one_hot)
        else:
            path_scores.append(tag_sequence[timestep, :] + scores.squeeze())
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    viterbi_score, best_path = torch.max(path_scores[-1], 0)
    viterbi_path = [int(best_path.numpy())]
    for backward_timestep in reversed(path_indices):
        viterbi_path.append(int(backward_timestep[viterbi_path[-1]]))
    # Reverse the backward path.
    viterbi_path.reverse()

    if has_start_end_restrictions:
        viterbi_path = viterbi_path[1:-1]
    #return viterbi_path, viterbi_score
    return np.array(viterbi_path, dtype=np.int32)


class TaggerConfig:
    def __init__(self):
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.n_rnn_layers = 1
        self.bidirectional = True 
    
class SAN(nn.Module):
    '''
    SAN: Self-Attention + residual connection
    '''
    def __init__(self, d_model, nhead, dropout=0.1):
        super(SAN, self).__init__()
        self.d_model = d_model 
        self.nhead = nhead 
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm(src)
        return src 
    

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(GRU, self).__init__()

        self.input_size = input_size 
        if bidirectional:
            self.hidden_size = hidden_size // 2 
        
        else:
            self.hidden_size = hidden_size 
        
        self.bidirectional = bidirectional
        self.Wxrz = nn.Linear(in_features=self.input_size, out_features=2*self.hidden_size, bias=True)
        self.Whrz = nn.Linear(in_features=self.hidden_size, out_features=2*self.hidden_size, bias=True)
        self.Wxn = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=True)
        self.Whn = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=True)

        self.LNx1 = nn.LayerNorm(2*self.hidden_size)
        self.LNh1 = nn.LayerNorm(2*self.hidden_size)
        self.LNx2 = nn.LayerNorm(self.hidden_size)
        self.LNh2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        '''
        x: input tensor, shape: (batch_size, seq_len, input_size)
        '''

        def recurrence(xt, htm1):
            '''
            param xt: current input
            param htm1: previous hidden state 
            '''

            gates_rz = torch.sigmoid(self.LNx1(self.Wxrz(xt)) + self.LNh1(self.Whrz(htm1)))
            rt, zt = gates_rz.chunk(2, 1)
            nt = torch.tanh(self.LNx2(self.Wxn(xt)) + rt * self.LNh2(self.Whn(htm1)))
            ht = (1.0 - zt) * nt + zt * htm1 
            return ht 

        steps = range(x.size(1))
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)
        input = x.transpose(0, 1)
        output = []
        for t in steps:
            hidden = recurrence(input[t], hidden)
            output.append(hidden)
        
        output = torch.stack(output, 0 ).transpose(0, 1)

        if self.hidirectional:
            output_b = []
            hidden_b = self.init_hidden(batch_size)
            for t in steps[::-1]:
                hidden_b = recurrence(input[t], hidden_b)
                output_b.append(hidden_b)

            output_b = output_b[::-1]
            output_b = torch.stack(output_b, 0).transpose(0, 1)
            output = torch.cat([output, output_b], dim=-1)
        return output, None 

    def init_hidden(self, bs):
        if torch.cuda.is_available() :
            h_0 = torch.zeros(bs, self.hidden_size).cuda()
        else:
            h_0 = torch.zeros(bs, self.hidden_size).cpu()

        return h_0 

class CRF(nn.Module):
    # borrow the code from 
    # https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
    def __init__(self, num_tags, constraints=None, include_start_end_trainsitions=None):
        super(CRF, self).__init__()
        self.num_tags = num_tags 
        self.include_start_end_trainsitions = include_start_end_trainsitions
        self.transitions = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))
        constraint_mask = torch.Tensor(self.num_tag + 2, self.num_tags + 2).fill_(1.)
        if include_start_end_trainsitions:
            self.start_transitions = nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = nn.Parameter(torch.Tensor(num_tags))
        
        self.constraint_mask = nn.Parameter(constraint_mask, requires_grad=True)
        self._init_weight()

    def forward(self, inputs, tags, mask=None):
        if mask in None:
            mask = torch.ones(*tags.size(), dtype=torch.long)
        
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)
        return torch.sum(log_numerator - log_denominator)

    def _init_weight(self):
        '''
        initialize the parameters in CRF
        '''
        nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_trainsitions:
            nn.init.normal_(self.start_transitions)
            nn.init.normal_(self.end_transitions)
    
    def _init_likelihood(self, logits, mask):
        '''
        logits: emission score calculated bt a linear layer, shape: (batch_size, seq_len, num_tags)

        '''
        bsz, seq_len, num_tags = logits.size()
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose().contiguous()

        if self.include_start_end_trainsitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        
        else:
            alpha = logits[0]

        for t in range(1, seq_len):
            emit_scores = logits[t].view(bsz, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(bsz, num_tags, 1)

            # calculate the likelihood
            inner = broadcast_alpha + emit_scores + transition_scores

            # mask the padded token when me the padded token, retain the previous alpha 
            alpha = (logsumexp(inner, 1) * mask[t].view(bsz, 1) + alpha * (1 - mask[t]).view(bsz, 1))

        if self.include_start_end_trainsitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        
        else:
            stops = alpha 

        return logsumexp(stops)
    
    def _joint_likelihood(self, logits, tags, mask):
        bsz, seq_len, _ = logits.size()

        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        if self.include_start_end_trainsitions:
            score = self.start_transitions.index_select(0, tags[0])
        
        else:
            score = 0.0 

        for t in range(seq_len - 1):
            current_tag, next_tags = tags[t], tags[t+1]
            
            # The scores for transitioning from current_tag to next_tag
            transitions_score = self.transitions[current_tag.view(-1), next_tags.view(-1)]

            # The score for using current_tag 
            emit_score = logits[t].gather(1, current_tag.view(bsz, 1)).squeeze(1)

            score = score + transitions_score * mask[t+1] + emit_score * mask[t]

        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, bsz)).squeeze(0)

        if self.include_start_end_trainsitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        
        else:
            last_transition_score = 0.0 

        last_inputs = logits[-1] # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))
        last_input_score = last_input_score.squeeze()

        score = score + last_transition_score + last_input_score * mask[-1]

        return score 
    
    def viter_tags(self, logits, mask):
        _, max_seq_len, num_tags = logits.size()

        logits, mask = logits.data, mask.data 

        start_tag = num_tags 
        end_tag = num_tags + 1 
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        constrained_transition = (
            self.transitions * self.constraint_mask[:num_tags, :num_tags] + 
            -10000.0 * (1-self.constraint_mask[:num_tags, :num_tags])
        )

        transitions[:num_tags, :num_tags] = constrained_transition.data 

        if self.include_start_end_trainsitions:
            transitions[start_tag, :num_tags] = (
                self.start_transitions.detach() * self.constraint_mask[start_tag, :num_tags].data + 
                -10000.0 * (1-self.constraint_mask[start_tag, :num_tags].detach())
            )

            transitions[:num_tags, end_tag] = (
                self.end_transitions.detach() * self.constraint_mask[:num_tags, end_tag].data + 
                -10000.0 * (1-self.constraint_mask[:num_tags, end_tag].detach())
            )

        else:
            transitions[start_tag, :num_tags] = (-10000.0 * (1-self.constraint_mask[start_tag, :num_tags].detach()))
            transitions[:num_tags, end_tag] = (-10000.0 * (1-self.constraint_mask[:num_tags, end_tag].detach()))
        
        best_paths = []
        tag_sequence = torch.Tensor(max_seq_len + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            # perform viterbi decoding sample by sample
            seq_len = torch.sum(prediction_mask)
            # Start with everything totally unikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG 
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence length we just use the incoming prediction 
            tag_sequence[1:(seq_len + 1), :num_tags] = prediction[:seq_len]
            # And at the last timestep we must have the END_TAG
            tag_sequence[seq_len + 1, end_tag] = 0.
            viterbi_path = viterbi_decode(tag_sequence[:(seq_len + 2)], transitions)
            viterbi_path = viterbi_path[1:-1]
            best_paths.append(viterbi_path)
        
        return best_paths 