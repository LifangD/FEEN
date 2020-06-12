import torch
import torch.nn as nn
import torch.nn.functional as F


def to_scalar(var):
    return var.view (-1).detach ().tolist ()[0]


def argmax(vec):
    _, idx = torch.max (vec, 1)
    return to_scalar (idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax (vec)]
    max_score_broadcast = max_score.view (1, -1).expand (1, vec.size ()[1])
    return max_score + torch.log (torch.sum (torch.exp (vec - max_score_broadcast)))


def argmax_batch(vecs):
    _, idx = torch.max (vecs, 1)
    return idx


def log_sum_exp_batch(vecs):  # bs*ts
    maxi = torch.max (vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat (1, vecs.shape[1])
    recti_ = torch.log (torch.sum (torch.exp (vecs - maxi_bc), 1))
    return maxi + recti_








class CRF (nn.Module):
    def __init__(self, tagset_size, tag_dictionary, device, is_bert=None):
        super (CRF, self).__init__ ()

        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        if is_bert:
            # 这里是不对的，CRF 中的start 和 stop 是虚拟的， 但是 BERT中的[CLS] 和[SEP]是实际的
            # 实际在预测时，[CLS] 和[SEP] 位置被学成是O 但是不太影响其他指标的学习. 或者 这里的SEP,CLS就应该被预测成O
            # 因为在assign transition时 被到[CLS]的概率 非常大 但是给到的标签是[CLS] 所以会有矛盾，导致crf的loss非常大；
            # fix: 不应该额外考虑BERT的情况 而是一视同仁，首尾是cls 或者 sep 应该被学习出来
            self.START_TAG = "[CLS]"
            self.STOP_TAG = "[SEP]"
        self.tag_dictionary = tag_dictionary
        self.tagset_size = tagset_size
        self.device = device
        self.transitions = torch.randn (tagset_size, tagset_size)
        # self.transitions = torch.zeros(tagset_size, tagset_size)
        self.transitions.detach ()[self.tag_dictionary[self.START_TAG], :] = -10000  #终点是start的概率
        self.transitions.detach ()[:,self.tag_dictionary[self.STOP_TAG]] = -10000  # ts  起点是ende的概率 和zhihu上的刚好相反  X_{ij} 表示从 j到i
        self.transitions = self.transitions.to (device)
        self.transitions = nn.Parameter (self.transitions)

    def _viterbi_decode(self, feats):  # sample-wise
        backpointers = []
        backscores = []
        scores = []
        init_vvars = (torch.FloatTensor (1, self.tagset_size).to (self.device).fill_ (-10000.0))
        init_vvars[0][self.tag_dictionary[self.START_TAG]] = 0
        forward_var = init_vvars

        for i in range(1,feats.shape[0]): # exclude the first
            feat = feats[i,:] #emission
            next_tag_var = (
                    forward_var.view (1, -1).expand (self.tagset_size, self.tagset_size)
                    + self.transitions
            )
            _, bptrs_t = torch.max (next_tag_var, dim=1)  # 到tag n 的最大概率，index-value 中包含了前置的链接。如果需要取TOP-N的时候，则取最后时刻的前N个
            viterbivars_t = next_tag_var[range (len (bptrs_t)), bptrs_t]  # 每次只保留tag个数的值
            forward_var = viterbivars_t + feat  # 1*ts
            backscores.append (forward_var)
            backpointers.append (bptrs_t)  # 前序节点的index

        # terminal_var = (
        #         forward_var
        #         + self.transitions[self.tag_dictionary[self.STOP_TAG]]
        # )  # ts
        # terminal_var.detach ()[self.tag_dictionary[self.STOP_TAG]] = -10000.0
        # terminal_var.detach ()[self.tag_dictionary[self.START_TAG]] = -10000.0

        terminal_var = forward_var
        best_tag_id = argmax (terminal_var.unsqueeze (0))
        best_path = [best_tag_id]
        for bptrs_t in reversed (backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append (best_tag_id.item ())  # [id_n,id_{n-1},...,id_0 ]
        best_scores = []
        for backscore in backscores:
            softmax_score = F.softmax (backscore, dim=0)  # softmax on tag_size
            _, idx = torch.max (backscore, 0)
            prediction = idx.item ()
            best_scores.append (softmax_score[prediction].item ())
            scores.append ([elem.item () for elem in softmax_score.flatten ()])
            # each tag's prob on each place # 每一步都选最大并不是最好的？ 每一步的最大是end 但是end 不能作为起点

        #swap_best_path, swap_max_score = (best_path[0], scores[-1].index (max (scores[-1])))



        # best_path[0] 是路径分数最高的最后一个label, scores[-1].index (max (scores[-1])) 是最后local 位置上分数最大的
        # 作者代码没删干净，这里原来只是为check 代码的正确性而做assertionhttps://github.com/flairNLP/flair/pull/782； best_scores 以及scores 不影响模型的输出
        #scores[-1][swap_best_path], scores[-1][swap_max_score] = (scores[-1][swap_max_score], scores[-1][swap_best_path])   # 为啥要交换？，也没用上
        start = best_path[-1]
        assert start == self.tag_dictionary[self.START_TAG]
        best_path.reverse () # 没有额外增加start 和 stop

        return best_scores, best_path, scores




    def _forward_alg(self, feats, lens_):

        # 注意存在[CLS] 和[SEP]是否被重复计算的问题

        init_alphas = feats[:, 0, :].clone()
        init_alphas[:,:self.tag_dictionary[self.START_TAG]] = -10000.0
        init_alphas[:, self.tag_dictionary[self.START_TAG]+1:]  = -10000.0
        forward_var = torch.zeros (
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=self.device,
        )  # bs*(seq_len+1)*tag_size
        forward_var[:, 1, :] = init_alphas
        transitions = self.transitions.view (1, self.transitions.shape[0], self.transitions.shape[1]).repeat (
            feats.shape[0], 1, 1)  # bs*tag_size*tag_size
        for i in range (1,feats.shape[1]): #
            emit_score = feats[:, i, :]
            tag_var = (
                    emit_score[:, :, None].repeat (1, 1, transitions.shape[2])  # the current
                    + transitions
                    + forward_var[:, i, :][:, :, None].repeat (1, 1, transitions.shape[2]).transpose (2, 1)
            # the previous
            )  # bs * ts * ts
            max_tag_var, _ = torch.max (tag_var,
                                        dim=2)  # log(e^{x_{max}}*e^{x_1-x_{max}}, in case of the large number triggering the overflow.
            tag_var = tag_var - max_tag_var[:, :, None].repeat (1, 1, transitions.shape[2])
            agg_ = torch.log (torch.sum (torch.exp (tag_var), dim=2))
            cloned = forward_var.clone ()
            cloned[:, i + 1, :] = max_tag_var + agg_
            forward_var = cloned
            forward_var[:, i + 1, :] = max_tag_var + agg_  #one of the variables needed for gradient computation has been modified by an inplace operation


        terminal_var = forward_var[range (forward_var.shape[0]), lens_, :]
        #terminal_var = forward_var + self.transitions[self.tag_dictionary[self.STOP_TAG]][None, :].repeat (forward_var.shape[0], 1)  # bs*ts -10000
        alpha = log_sum_exp_batch (terminal_var)
        return alpha

    def partial_loss(self, feats, lens_, one_hot_labels):

        #init_alphas = torch.FloatTensor (self.tagset_size).fill_ (-10000.0)
        #init_alphas[self.tag_dictionary[self.START_TAG]] = 0.0  # [0,...,-10000]
        init_alphas = feats[:,0,:].clone().masked_fill_(one_hot_labels[:,0].byte(),-10000.0)

        #gold_scores = self._score_sentence(feats,tags,lens_)
        forward_var = torch.zeros (
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=self.device,
        )  # bs*(seq_len+1)*tag_size

        forward_var[:, 1, :] = init_alphas  # bs*tag_size, where the logit value of the start is [0,...,-10000]
        partial_forward_var =   forward_var = torch.zeros (
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=self.device,
        )
        partial_forward_var[:, 1, :] = init_alphas  # bs*tag_size, where the logit value of the start is [0,...,-10000]

        #

        transitions = self.transitions.view (1, self.transitions.shape[0], self.transitions.shape[1]).repeat (feats.shape[0], 1, 1)  # bs*tag_size*tag_size
        for i in range (1,feats.shape[1]):
            emit_score = feats[:, i, :]
            tag_var = (
                    emit_score[:, :, None].repeat (1, 1, transitions.shape[2])  # the current
                    + transitions
                    + forward_var[:, i, :][:, :, None].repeat (1, 1, transitions.shape[2]).transpose (2, 1)
                # the previous
            )  # bs * ts * ts

            max_tag_var, _ = torch.max (tag_var, dim=2)  # log(e^{x_{max}}*e^{x_1-x_{max}}, in case of the large number triggering the overflow.
            tag_var = tag_var - max_tag_var[:, :, None].repeat (1, 1, transitions.shape[2])
            agg_ = torch.log (torch.sum (torch.exp (tag_var), dim=2))
            cloned = forward_var.clone ()
            cloned[:, i + 1, :] = max_tag_var + agg_  # why clone?
            forward_var = cloned

            partial_tag_var = (emit_score[:, :, None].repeat (1, 1, transitions.shape[2])  # the current
                               + transitions
                               + partial_forward_var[:, i, :][:, :, None].repeat (1, 1,transitions.shape[2]).transpose (2,1))
            max_partial_tag_var, _ = torch.max (partial_tag_var, dim=2)
            partial_tag_var = partial_tag_var - max_partial_tag_var[:, :, None].repeat (1, 1, transitions.shape[2])
            partial_agg_ = torch.log (torch.sum (torch.exp (partial_tag_var), dim=2))
            partial_cloned = partial_forward_var.clone ()
            partial_cloned[:, i + 1, :] = max_partial_tag_var + partial_agg_
            partial_cloned[:, i + 1, :].masked_fill_ (one_hot_labels[:, i].byte (), -10000)
            partial_forward_var = partial_cloned

        #terminal_idx = list(map((lambda x:x-1),lens_))
        terminal_var = forward_var[range (forward_var.shape[0]), lens_, :]
        # terminal_var = forward_var + self.transitions[self.tag_dictionary[self.STOP_TAG]][None, :].repeat (
        #     forward_var.shape[0], 1)  # bs*ts -10000

        partial_terminal_var = partial_forward_var[range (partial_forward_var.shape[0]), lens_, :]
        # partial_terminal_var = partial_forward_var + self.transitions[self.tag_dictionary[self.STOP_TAG]][None,
        #                                              :].repeat (partial_forward_var.shape[0], 1)
        partial_alpha = log_sum_exp_batch (partial_terminal_var)
        alpha = log_sum_exp_batch (terminal_var)
        loss = (alpha - partial_alpha).mean ()
        return loss

    def _score_sentence(self, feats, tags, lens_): # the initial implementation is wrong for bert-->fixed
        '''
        start = torch.LongTensor ([self.tag_dictionary[self.START_TAG]]).to (self.device)
        start = start[None, :].repeat (tags.shape[0], 1)
        stop = torch.LongTensor ([self.tag_dictionary[self.STOP_TAG]]).to (self.device)
        stop = stop[None, :].repeat (tags.shape[0], 1)
        pad_start_tags = torch.cat ([start, tags], 1)
        pad_stop_tags = torch.cat ([tags, stop], 1)
        '''
        pad_stop_tags = tags[:,1:] # 不计算额外的start 和 stop; 只实际的标签
        pad_start_tags = tags[:,:-1]

        for i in range (len (lens_)):
            pad_stop_tags[i, lens_[i]-1:] = self.tag_dictionary[self.STOP_TAG]  # lens: the real seq_len
        score = torch.FloatTensor (feats.shape[0]).to (self.device)
        for i in range (feats.shape[0]):
            r = torch.LongTensor (range (lens_[i])).to (self.device)
            score[i] = torch.sum (self.transitions[pad_stop_tags[i, : lens_[i]-1], pad_start_tags[i, : lens_[i]-1]])\
                       + torch.sum (feats[i, r, tags[i, : lens_[i]]])# emission score!
        return score


        '''
        start = torch.LongTensor([self.tag_dictionary[self.START_TAG]]).to(self.device)
        start = start[None, :].repeat(tags.shape[0], 1)
        stop = torch.LongTensor([self.tag_dictionary[self.STOP_TAG]]).to(self.device)
        stop = stop[None, :].repeat(tags.shape[0], 1)
        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)
        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i] :] = self.tag_dictionary[self.STOP_TAG]
        score = torch.FloatTensor(feats.shape[0]).to(self.device)
        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(self.device)
            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]
                ]
            ) + torch.sum(feats[i, r, tags[i, : lens_[i]]])
        return score
        '''






    def _obtain_labels(self, feature, id2label, input_lens):
        tags = []
        tag_ids = []
        for feats, length in zip (feature, input_lens):
            confidences, tag_seq, scores = self._viterbi_decode (feats[:length])
            tags.append ([id2label[tag] for tag in tag_seq])
            tag_ids.append(tag_seq)
            #下面语句的等价写法
            # mat = []
            # for score_dist in scores:
            #     sen = []
            #     for score_id,score in enumerate(score_dist):
            #         sen.append(id2label[score_id])
            #     mat.append(sen)
            # all_tags.append(mat)

            #all_tags.append ([[id2label[score_id] for score_id, score in enumerate (score_dist)] for score_dist in scores])  # a the path for TOP-N?ll
        return tags, tag_ids

    def calculate_loss(self, scores, tag_list, lengths):
        return self._calculate_loss_old (scores, lengths, tag_list)

    def _calculate_loss_old(self, features, lengths, tags):
        forward_score = self._forward_alg (features, lengths)
        gold_score = self._score_sentence (features, tags, lengths)
        score = forward_score - gold_score
        return score.mean ()

    def beam_search_decode(self, feats,k):  # sample-wise
        backpointers = []
        backscores = []
        scores = []
        init_vvars = (torch.FloatTensor(1, self.tagset_size).to(self.device).fill_(-10000.0))
        init_vvars[0][self.tag_dictionary[self.START_TAG]] = 0
        forward_var = init_vvars

        for feat in feats:
            next_tag_var = (
                    forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size)
                    + self.transitions
            )
            _, bptrs_t = torch.max(next_tag_var, dim=1)  # 到tag n 的最大概率，index-value 中包含了前置的链接。如果需要取TOP-N的时候，则取最后时刻的前N个
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]  # 每次只保留tag个数的值
            forward_var = viterbivars_t + feat  # 1*ts, 对K之外的mask掉 而不是直接削减 不然无法做矩阵运算
            backscores.append(forward_var)
            backpointers.append(bptrs_t)  # 前序节点的index

        terminal_var = (
                forward_var
                + self.transitions[self.tag_dictionary[self.STOP_TAG]]
        )  # ts
        terminal_var.detach()[self.tag_dictionary[self.STOP_TAG]] = -10000.0
        terminal_var.detach()[self.tag_dictionary[self.START_TAG]] = -10000.0
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id.item())  # [id_n,id_{n-1},...,id_0 ]
        best_scores = []
        for backscore in backscores:
            softmax_score = F.softmax(backscore, dim=0)  # softmax on tag_size
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax_score[prediction].item())
            scores.append([elem.item() for elem in softmax_score.flatten()])
            # each tag's prob on each place # 每一步都选最大并不是最好的？ 每一步的最大是end 但是end 不能作为起点

        swap_best_path, swap_max_score = (best_path[0], scores[-1].index(max(scores[-1])))

        # best_path[0] 是路径分数最高的最后一个label, scores[-1].index (max (scores[-1])) 是最后local 位置上分数最大的
        # 作者代码没删干净，这里原来只是为check 代码的正确性而做assertionhttps://github.com/flairNLP/flair/pull/782； best_scores 以及scores 不影响模型的输出
        # scores[-1][swap_best_path], scores[-1][swap_max_score] = (scores[-1][swap_max_score], scores[-1][swap_best_path])   # 为啥要交换？，也没用上
        start = best_path.pop()
        assert start == self.tag_dictionary[self.START_TAG]
        best_path.reverse()

        return best_scores, best_path, scores

