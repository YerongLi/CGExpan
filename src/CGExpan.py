import os
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
import random
import math
from collections import defaultdict as ddict
import queue
import copy
import nltk
import inflect
import pickle
from utils import *
import time

import logging
GENERATION_SAMPLE_SIZE = 6
EXPANSION_SAMPLE_SIZE = 3
POS_CNAME_THRES = 5./6


class CGExpan(object):

    def __init__(self, args, device, model_name='bert-base-uncased', dim=768):

        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case = False)

        self.maskedLM = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        self.maskedLM.to(device)
        self.maskedLM.eval()

        self.k = args.k
        self.gen_thres = args.gen_thres
        self.eid2name, self.keywords, self.eid2idx = load_vocab(os.path.join(args.dataset, args.vocab))

        self.entity_pos = pickle.load(open(os.path.join(args.dataset, args.entity_pos_out), 'rb'))

        self.pretrained_emb = np.memmap(os.path.join(args.dataset, args.emb_file), dtype='float32', mode='r', shape=(self.entity_pos[-1], dim))

        self.means = np.array([np.mean(emb, axis=0) for emb in self.get_emb_iter()])

        self.inflect = inflect.engine()

        mask_token = self.tokenizer.mask_token
        self.generation_templates = [
            [mask_token, ' such as {} , {} , and {} .', 1],
            ['such ' + mask_token, ' as {} , {} , and {} .', 1],
            ['{} , {} , {} or other ' + mask_token, ' .', 0],
            ['{} , {} , {} and other ' + mask_token, ' .', 0],
            [mask_token, ' including {} , {} , and {} .', 1],
            [mask_token, ' , especially {} , {} , and {} .', 1],
        ]

        self.ranking_templates = [
            '{} such as ' + mask_token + ' .',
            'such {} as ' + mask_token + ' .',
            mask_token + ' or other {} .',
            mask_token + ' and other {} .',
            '{} including ' + mask_token + ' .',
            '{} especially ' + mask_token + ' .',
        ]

        self.expansion_templates = [
            ('', ' such as {} , {} , {} , and {} .'),
            ('such ', ' as {} , {} , {} , and {} .'),
            ('{} , {} , {} , {} or other ', ' .'),
            ('{} , {} , {} , {} and other ', ' .'),
            ('', ' including {} , {} , {} , and {} .'),
            ('', ' , especially {} , {} , {} , and {} .'),
        ]

        self.calculated_cname_rep = {}

    def rand_idx(self, l):
        for _ in range(10000):
            for i in np.random.permutation(l):
                yield i

    def get_emb(self, i):
        return self.pretrained_emb[self.entity_pos[i]:self.entity_pos[i+1]]

    def get_emb_iter(self):
        for i in range(len(self.keywords)):
            yield self.pretrained_emb[self.entity_pos[i]:self.entity_pos[i+1]]

    def expand(self, query_set, target_size, m=2, gt=None):

        logging.info('start expanding: ' + str([self.eid2name[eid] for eid in query_set]))

        start_time = time.time()
        
        expanded_set = []
        
        prev_cn = set()
        
        neg_set = set()
        neg_cnames = set()
        decrease_count = 0
        margin = m
        
        while len(expanded_set) < target_size:
            
            logging.info(f'num of expanded entities: {len(expanded_set)}, time: {int((time.time() - start_time)/60)} min {int(time.time() - start_time)%60} sec')
            if gt is not None:
                logging.info(f'map10: {apk(gt, expanded_set, 10)}, map20: {apk(gt, expanded_set, 20)}, map50: {apk(gt, expanded_set, 50)}')


            # generate class names
            
            set_text = [self.eid2name[q].lower() for q in query_set + expanded_set]

            cname2count = self.class_name_generation(set_text)
            # logging.info('cname2count')
            # logging.info(cname2count)
            # INFO:root:cname2count
            # INFO:root:defaultdict(<class 'int'>, {'areas': 17, 'countries': 6, 'states': 36, 'small places': 4, 'southern states': 25, 'small areas': 6, 'small states': 6, 'large areas': 4, 'places': 12, 'european countries': 4, 'eu countries': 2, 'similar locations': 3, 'developed countries': 3, 'locations': 6, 'geographical locations': 2, 'coastal areas': 7, 'similar places': 6, 'northern states': 2, 'geographic areas': 2, 'european places': 2, 'eastern states': 4, 'northern areas': 5, 'europes': 2, 'many usas': 2, 'usas': 6, 'exotic places': 3, 'coastal states': 7, 'species': 2, 'plant species': 2, 'southern locations': 4, 'native species': 2, 'coastal locations': 3, 'desert areas': 1, 'regional locations': 2, 'exotic locations': 1, 'see examples': 1, 'examples': 1, 'foreign countries': 1, 'mountainous areas': 1, 'souths': 1, 'deep souths': 1, 'coas
            # tal regions': 1, 'regions': 2, 'border areas': 2, 'illinois': 1, 'several states': 2, 'border regions': 1, 'commonwealth countries': 1, 'metropolitan areas': 1, 'asian countries': 1, 'eastern texa': 2, 'western states': 1, 'southern texa': 2, 'texa': 2, 'geographic locations': 1, 'so
            # utheastern states': 1, 'northern locations': 1})
            # INFO:root:cname2count

            # logging.info(len(cname2count))
            # INFO:root:start expanding: ['texas', 'florida', 'california']
            # INFO:root:cname2count
            # INFO:root:28
            # INFO:root:cname2count
            # INFO:root:49
            # INFO:root:start expanding: ['iowa', 'illinois', 'new jersey']
            # INFO:root:cname2count
            # INFO:root:24

            pos_cname, neg_cnames = self.class_name_ranking(cname2count, query_set, expanded_set, neg_cnames, prev_cn, margin)
            prev_cn.add(pos_cname)

            # expansion

            new_entities = self.class_guided_expansion(pos_cname, query_set + expanded_set, set_text, neg_set)

            # filter

            current_expanded_size = len(expanded_set)
            expanded_set.extend(new_entities)
            expanded_set, filter_out = self.class_guided_filter(query_set, expanded_set, pos_cname, neg_cnames, cname2count)
            neg_set = neg_set | filter_out
            if len(expanded_set) <= current_expanded_size:
                decrease_count += 1
                if decrease_count >= 2:
                    margin += 1
                    decrease_count = 0
                    neg_set = set()
                    neg_cnames = set()
                    if margin >= 10:
                        break
            else:
                decrease_count = 0

        return expanded_set

    def class_name_generation(self, set_text):
        cname2count = ddict(int)
        idx_generator = self.rand_idx(len(set_text))
        for _ in range(GENERATION_SAMPLE_SIZE):
            for template in self.generation_templates:
                candidate = set()
                q = queue.Queue()
                q.put([])
                
                template = copy.deepcopy(template)
                indices = []
                for n in idx_generator:
                    if n not in indices:
                        indices.append(n)
                        if len(indices) == 3:
                            break
                template[template[2]] = template[template[2]].format(*[set_text[i] for i in indices])
                    
                while not q.empty():
                    c = q.get()
                    if len(c) >= 2:
                        continue
                    text = template[0] + (' ' if len(c) > 0 else '') + ' '.join(c) + template[1]
                    ids = torch.tensor([self.tokenizer.encode(text, max_length=512)]).long()
                    mask_pos = (ids == self.tokenizer.mask_token_id).nonzero()[0, 1]
                    ids = ids.cuda()
                    with torch.no_grad():
                        predictions = self.maskedLM(ids)[0]
                    _, predicted_index = torch.topk(predictions[0, mask_pos], k=3)
                    predicted_token = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in predicted_index]
                    for t in predicted_token:
                        tag = nltk.pos_tag([t] + c)
                        tag = tag[0][1]
                        if tag in set(['JJ', 'NNS', 'NN']) and t not in set(c)\
                            and t not in set([self.inflect.plural(cc) for cc in c]) and t not in ['other', 'such', 'others']:
                            if len(c) == 0 and tag == 'JJ':
                                continue
                            if len(c) == 0 and tag == 'NN':
                                t = self.inflect.plural(t)
                            new = [t] + c
                            candidate.add(tuple(new))
                            q.put(new)
                for c in candidate:
                    cname2count[' '.join(c)] += 1
        return cname2count

    def class_name_ranking(self, cname2count, query_set, expanded_set, neg_cnames, prev_cn, margin):
        current_set = query_set + expanded_set
        ids = []
        cnames = [cname for cname in cname2count if cname2count[cname] >= self.gen_thres]
        cnames += [cn for cn in prev_cn if cn not in cnames]
        cname2idx = {cname:i for i, cname in enumerate(cnames)}
        cnames_rep = np.vstack([self.get_cname_rep(cname) for cname in cnames])
        scores = np.zeros((len(current_set), len(cnames)))
        for i, eid in enumerate(current_set):
            emb = self.get_emb(self.eid2idx[eid])
            if len(emb) < self.k:
                continue
            sims = cos(cnames_rep, emb)
            for j in range(len(cnames)):
                scores[i, j] = np.mean(np.partition(np.amax(sims[j*6:(j+1)*6], axis=0), -self.k)[-self.k:])
        cname2mrr=ddict(float)
        for eid, score in zip(current_set, scores):
            r = 0.
            for i in np.argsort(-score):
                cname = cnames[i]
                if cname2count[cname] < min(GENERATION_SAMPLE_SIZE*len(self.generation_templates)*POS_CNAME_THRES, max(cname2count.values())) and cname not in prev_cn:
                    continue
                r += 1
                cname2mrr[cname] += 1 / r
        pos_cname = sorted(cname2mrr.keys(), key=lambda x: cname2mrr[x], reverse=True)[0]

        # find negative entities
        uni_cnames = [cname for cname in cnames if len(cname.split(' ')) == 1 and not pos_cname.endswith(cname)]
        this_neg_cnames = set(uni_cnames)
        for eid, score in zip(query_set, scores):
            ranked_uni_cnames = sorted([pos_cname]+uni_cnames, key=lambda x: score[cname2idx[x]], reverse=True)
            for i, cname in enumerate(ranked_uni_cnames):
                if cname == pos_cname:
                    break
            this_neg_cnames = this_neg_cnames & set(ranked_uni_cnames[i+1+margin:])
        return pos_cname, neg_cnames | this_neg_cnames


    def class_guided_expansion(self, pos_cname, current_set, set_text, neg_set):
        global_idx_generator = self.rand_idx(len(current_set))
        local_idx_generator = self.rand_idx(len(current_set))
        global_scores = cos(self.means[[self.eid2idx[eid] for eid in current_set]], self.means)

        ids = []
        for _ in range(EXPANSION_SAMPLE_SIZE):
            for template in self.expansion_templates:
                indices = []
                for n in local_idx_generator:
                    if n not in indices:
                        indices.append(n)
                        if len(indices) == 3:
                            break
                fill_in = [self.tokenizer.mask_token] + [set_text[i] for i in indices]
                # logging.info('fill_in')
                # logging.info(fill_in)
                # INFO:root:fill_in
                # INFO:root:['[MASK]', 'california', 'missouri', 'florida']
                # INFO:root:fill_in
                # INFO:root:['[MASK]', 'georgia', 'arizona', 'new york']
                # INFO:root:fill_in
                # INFO:root:['[MASK]', 'delaware', 'texas', 'nevada']
                # INFO:root:fill_in

                fill_in = np.random.permutation(fill_in)
                text = template[0] + pos_cname + template[1]
                text = text.format(*fill_in)
                # logging.info('text')
                # logging.info(text)
                # INFO:root:text
                # INFO:root:[MASK] , florida , texas , california or other states .
                # INFO:root:text
                # INFO:root:[MASK] , california , florida , texas and other states .
                # INFO:root:text
                # INFO:root:states including texas , [MASK] , florida , and california .
                # INFO:root:text
                # INFO:root:states , especially florida , california , texas , and [MASK] .
                # INFO:root:text
                # INFO:root:states such as [MASK] , texas , florida , and california .
                ids.append(self.tokenizer.encode(text, max_length=512))
        mask_rep = self.get_mask_rep(ids)
        # logging.info('length')
        # logging.info(len(ids))
        # INFO:root:length
        # INFO:root:18

        # logging.info(mask_rep)
        # INFO:root:mask_rep
        # INFO:root:[[-0.02163155  0.08432811 -0.11579558 ... -0.15900703 -0.05456618
        #   -0.12115346]
        #  [ 0.07413231  0.19879022 -0.11666524 ... -0.15382662  0.03588961
        #   -0.12604576]
        #  [ 0.02818201  0.06998995 -0.28485173 ... -0.03825528  0.00897471
        #   -0.0417298 ]
        #  ...
        #  [ 0.05698122  0.21749909 -0.07018307 ... -0.2447705   0.00188999
        #   -0.07645971]
        #  [ 0.02078669  0.08932656 -0.1914087  ... -0.13100626  0.02239787
        #   -0.24592625]
        #  [-0.23963006  0.18225236  0.08653576 ... -0.19262496  0.1368391
        #   -0.03403295]]


        eid2mrr = ddict(float)
        for local_rep in mask_rep:
            indices = []
            for n in global_idx_generator:
                if n not in indices:
                    indices.append(n)
                    if len(indices) == 3:
                        break
            this_global_score = np.mean(global_scores[indices], axis=0)
            this_global_score_ranking = np.argsort(-this_global_score)

            this_keywords = [self.keywords[i] for i in this_global_score_ranking[:500]]
            # logging.info('this_keywords')
            # logging.info(this_keywords)
            # INFO:root:this_keywords
            # INFO:root:[12400, 22982, 19129, 79092, 1371, 39817, 1807, 50103, 259, 1978, 19214, 12380, 73219, 55857, 48157, 79187, 73148, 48156, 2479, 34957, 2109, 66993, 80218, 2544, 31525, 68920, 19641, 63093, 17742, 78468, 3725, 63436, 74107, 28016, 12343, 22952, 85440, 19120, 9950, 3264, 90991, 5806, 73150, 63281, 3500, 19386, 75549, 20301, 12224, 8245, 17098, 10888, 73205, 19864, 16710, 18172, 8607, 12777, 3711, 8355, 8287, 12430, 30169, 16579, 27451, 8252, 8920, 8660, 1858, 60035, 3555, 41610, 51677, 78138, 33235, 3472, 20227, 31858, 49686, 1407, 26743, 16641, 13341, 1600, 58559, 16599, 18198, 56219, 37038, 8075, 38092, 18201, 41575, 66013, 50323, 15767, 71325, 74678, 8852, 66986, 67645, 75465, 78609, 20310, 16585, 63686, 17301, 20293, 35972, 73852, 65882, 33889, 17904, 49871, 9776, 25860, 60972, 30697, 12723, 11900, 63226, 7074, 51785, 73951, 9744, 7899, 17316, 29828, 26742, 92306, 4733, 24947, 32056, 54488, 56793, 14423, 34334, 68839, 38127, 25636, 20692, 51543, 58530, 14454, 83257, 37338, 16781, 17354, 75776, 51756, 39556, 55818, 3899, 6576, 66589, 28573, 58138, 41387, 5058, 17242, 18197, 2312, 58893, 2937, 39116, 20037, 26274, 24862, 66963, 8678, 28058, 31849, 83262, 12683, 59171, 75259, 15973, 14874, 80194, 34821, 16650, 61210, 24315, 17745, 17291, 37986, 83567, 19999, 43481, 21322, 76332, 59122, 3783, 21279, 3125, 66894, 93429, 27370, 31166, 54490, 70201, 62752, 8099, 48040, 93434, 8277, 82953, 51191, 63873, 31787, 56596, 65908, 4320, 75286, 80033, 82173, 93421, 15401, 4698, 38736, 21280, 47259, 21728, 5950, 7222, 16452, 39387, 80004, 16447, 39563, 73436, 70221, 17759, 17154, 50110, 63315, 16932, 25687, 29136, 74334, 16665, 8831, 36216, 10793, 2366, 48749, 10639, 68742, 63295, 25814, 74113, 13983, 83597, 60310, 8192, 82970, 19736, 62485, 14670, 55047, 4463, 34790, 72889, 58542, 38082, 92233, 34486, 3596, 58544, 66715, 35608, 4397, 29966, 23413, 25035, 15291, 27323, 42029, 35855, 24705, 52163, 8365, 73853, 1359, 92318, 45609, 11574, 52393, 77836, 8613, 43681, 77179, 77543, 2803, 6415, 38086, 48006, 50697, 83545, 19993, 35112, 14826, 67032, 29959, 18113, 40371, 3106, 34150, 19290, 41398, 20018, 34828, 52012, 92284, 41208, 44460, 35917, 18758, 75807, 25918, 18472, 45858, 34332, 46585, 25988, 7457, 49279, 483, 21646, 29365, 82139, 8744, 5057, 54160, 15938, 48350, 24427, 33468, 54991, 65912, 43090, 51032, 37898, 8172, 42847, 6694, 83465, 82133, 39728, 50727, 60621, 77402, 30808, 40502, 56857, 3787, 8235, 33196, 61320, 75548, 79055, 57974, 11367, 75610, 85487, 66901, 10826, 12478, 63253, 39641, 48974, 37875, 4586, 81849, 18247, 9530, 55248, 84996, 41603, 18351, 8962, 136, 28220, 69772, 34951, 35900, 33938, 9175, 5974, 70322, 24691, 82795, 4833, 5062, 19295, 3964, 15749, 5841, 75561, 45016, 29602, 19994, 81222, 75832, 53498, 33939, 33819, 128, 20984, 7102, 48085, 43883, 7046, 10360, 31315, 22763, 34352, 14745, 83863, 37873, 28017, 4816, 22611, 45389, 79000, 14553, 29849, 449, 59457, 51315, 75123, 4016, 61189, 16161, 8390, 34439, 964, 78222, 7630, 28418, 21266, 61143, 75714, 23555, 23803, 27021, 14235, 69896, 56270, 92249, 74776, 7125, 6311, 73528, 83250, 44939, 8319, 17805, 62552, 81349, 6277, 85493, 49693, 61176, 29615, 8424, 31768, 82131, 67566, 79073, 7349, 20041, 33458, 5738, 66895, 18535, 8477, 76965, 20046, 9770, 32477, 80433, 43650, 24386, 19387, 61242, 63097, 22950, 51403, 31288, 16999, 34489, 17275, 24976, 84378, 21230, 5223, 32703, 9772, 33820]
            # INFO:root:this_keywords
            # INFO:root:[12400, 22982, 19129, 79092, 1371, 39817, 1807, 50103, 259, 1978, 19214, 12380, 73219, 55857, 48157, 79187, 73148, 48156, 2479, 34957, 2109, 66993, 80218, 254
            
            logging.info('this_keywords')
            logging.info(self.tokenizer.decode(this_keywords))
            this_global_score = [this_global_score[i] for i in this_global_score_ranking[:500]]
            this_embs = [self.get_emb(i) for i in [self.eid2idx[eid] for eid in this_keywords]]
            this_entity_pos = [0] + list(np.cumsum([len(emb) for emb in this_embs]))
            this_embs = np.vstack(this_embs)
            
            raw_local_scores = cos(local_rep[np.newaxis, :], this_embs)[0]

            local_scores = np.zeros((500,))
            for i in range(500):
                start_pos = this_entity_pos[i]
                end_pos = this_entity_pos[i+1]
                if end_pos - start_pos < self.k:
                    local_scores[i] = 1e-8
                else:
                    local_scores[i] = np.mean(np.partition(raw_local_scores[start_pos:end_pos], -self.k)[-self.k:])

            scores = 5*np.log(local_scores) + np.log(this_global_score)

            r = 0.
            for i in np.argsort(-scores):
                eid = this_keywords[i]
                if eid not in set(current_set) and eid not in neg_set:
                    r += 1
                    eid2mrr[eid] += 1 / r
                if r >= 20:
                    break
                            
        eid_rank = sorted(eid2mrr, key=lambda x: eid2mrr[x], reverse=True)
        for i, eid in enumerate(eid_rank):
            if eid2mrr[eid] < EXPANSION_SAMPLE_SIZE * len(self.expansion_templates) * 0.2:
                break
        return eid_rank[:max(5, i)]

    def class_guided_filter(self, query_set, expanded_set, pos_cname, neg_cnames, cname2count):

        cnames = [pos_cname] + list(neg_cnames)
        cname2idx = {cname:i for i, cname in enumerate(cnames)}
        cnames_rep = np.vstack([self.get_cname_rep(cname) for cname in cnames])

        filter_out = set()
        for eid in expanded_set:
            emb = self.get_emb(self.eid2idx[eid])
            sims = cos(cnames_rep, emb)
            cnt = 0
            for i in range(len(self.ranking_templates)):
                scores = np.mean(np.partition(sims[[j*6+i for j in range(len(cnames))]], -self.k, axis=1)[:, -self.k:], axis=1)
                if np.argmax(scores) != cname2idx[pos_cname]:
                    cnt += 1
            if cnt > 2:
                filter_out.add(eid)
        temp = set([cn for cn in cname2count if cname2count[cn] >= GENERATION_SAMPLE_SIZE * len(self.generation_templates) / 6.])
        temp.update([self.inflect.plural(cn) for cn in temp])
        filter_out.update([eid for eid in expanded_set if self.eid2name[eid].lower() in temp])
        return [eid for eid in expanded_set if eid not in filter_out], filter_out

    def get_cname_rep(self, cname):
        if cname not in self.calculated_cname_rep:
            ids = []
            for template in self.ranking_templates:
                text = copy.deepcopy(template).format(cname)
                ids.append(self.tokenizer.encode(text, max_length=512))
            self.calculated_cname_rep[cname] = self.get_mask_rep(ids)
        return self.calculated_cname_rep[cname]

    def get_mask_rep(self, batch_ids):
        batch_max_length = max(len(ids) for ids in batch_ids)
        ids = torch.tensor([ids + [0 for _ in range(batch_max_length - len(ids))] for ids in batch_ids]).long()
        masks = (ids != 0).long()
        temp = (ids == self.tokenizer.mask_token_id).nonzero()
        mask_pos = []
        for ti, t in enumerate(temp):
            assert t[0].item() == ti
            mask_pos.append(t[1].item())
        ids = ids.to('cuda')
        masks = masks.to('cuda')
        with torch.no_grad():
            batch_final_layer = self.maskedLM(ids, masks)[1][-1]
        return np.array([final_layer[idx].cpu().numpy() for final_layer, idx in zip(batch_final_layer, mask_pos)])
