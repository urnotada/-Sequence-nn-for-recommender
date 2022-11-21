#!xiao gu
# 2021.10 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import wget

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
         # x shape: (batch_size, channel, seq_len)
        return F.max_pool1d(x, kernel_size=x.shape[2]) # shape: (batch_size, channel, 1)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_channels,out_dim):
        super(TextCNN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  # embedding.shape: torch.Size([200, 8, 300])
#         self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), out_dim)
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = embedding_dim,
                                        out_channels = c,
                                        kernel_size = k))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.permute(0, 2, 1)
        encoding = torch.cat([self.pool(F.relu(conv(embeds))).squeeze(-1) for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs

class LinearGRU(nn.Module):
    def __init__(self, n_users, n_items, title_vocab, genres_vocab, vab_sex_len=None, vab_age_len=None,
                 vab_work_len=None, emb_size=None, hidden_units=1000, dropout=0.8, user_dropout=0.5):
        super(self.__class__, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_units = hidden_units
        if emb_size == None:
            emb_size = hidden_units
        self.emb_size = emb_size

        #
        embedding_dim, kernel_sizes, num_channels = 100, [3, 4, 5], [100, 100, 100]
        # inputshape：batch,seqlen => batch,emb_size
        self.txt_cnn = TextCNN(len(title_vocab), embedding_dim, kernel_sizes, num_channels, emb_size)
        self.genre_emb = nn.Embedding(len(genres_vocab), emb_size)
        self.genre_fc = nn.Linear(emb_size, emb_size)
        self.sex_emb = nn.Embedding(vab_sex_len, emb_size)
        self.age_emb = nn.Embedding(vab_age_len, emb_size)
        self.work_emb = nn.Embedding(vab_work_len, emb_size)
        #
        self.user_emb = nn.Embedding(n_users, emb_size)
        self.item_emb = nn.Embedding(n_items, emb_size)

        self.grucell = nn.GRUCell(input_size=emb_size * 7, hidden_size=hidden_units)
        self.linear = nn.Linear(hidden_units, n_items)
        self.dropout = nn.Dropout(dropout)
        self.user_dropout = nn.Dropout(user_dropout)

    def forward(self, user_vectors, item_vectors, title_ids, genres_ids, sex, age, work):
        batch_size, sequence_size = user_vectors.size()
        user_vectors = user_vectors
        item_vectors = item_vectors
        # batch_size * sequence_size * emb_size
        users = self.user_dropout(self.user_emb(user_vectors))
        items = self.item_emb(item_vectors)
        # title
        title_ids_list = []
        for title_item in title_ids:
            # batch,20,seq_len,=>20,seq_len
            title_ids_list.append(self.txt_cnn(title_item).unsqueeze(0))
        title_embs = torch.cat(title_ids_list, 0)

        sex = self.sex_emb(sex)
        age = self.age_emb(age)
        work = self.work_emb(work)
        # batch,20,emb_size
        # genres
        # shape:batch,20,6
        genres_embs = self.genre_emb(genres_ids)
        # batch,20,6,emb_size
        genres_embs = torch.sum(genres_embs, dim=-2, keepdim=False)
        h = torch.zeros(batch_size, self.hidden_units).to(device)
        #  1 * batch_size * hidden_units
        h_t = h.unsqueeze(0)
        for i in range(sequence_size):
            # 1 * batch_size * (2*emb_size)
            gru_input = torch.cat(
                [users[:, i, :], items[:, i, :], title_embs[:, i, :], genres_embs[:, i, :], sex[:, i, :], age[:, i, :],
                 work[:, i, :]], dim=-1)
            # 1 * batch_size * hidden_units
            h = self.grucell(gru_input, h)
            # (1+i) * batch_size * hidden_units
            h_t = torch.cat([h_t, h.unsqueeze(0)], dim=0)
        # batch_size * sequence_size * hidden_units
        ln_input = self.dropout(h_t[1:].transpose(0, 1))
        output_ln = self.linear(ln_input)
        output = F.log_softmax(output_ln, dim=-1)
        return output


from torch.nn.parameter import Parameter


class RectifiedLinearGRU(nn.Module):

    #     def __init__(self, n_users,n_items, emb_size=None, hidden_units=1000,dropout = 0.8,user_dropout = 0.5):
    def __init__(self, n_users, n_items, title_vocab, genres_vocab, vab_sex_len=None, vab_age_len=None,
                 vab_work_len=None, emb_size=None, hidden_units=1000, dropout=0.8, user_dropout=0.5):

        super(self.__class__, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_units = hidden_units
        if emb_size == None:
            emb_size = hidden_units
        self.emb_size = emb_size
        ## todo why embeding?
        embedding_dim, kernel_sizes, num_channels = 100, [3, 4, 5], [100, 100, 100]
        # inputshape：batch,seqlen => batch,emb_size
        self.txt_cnn = TextCNN(len(title_vocab), embedding_dim, kernel_sizes, num_channels, emb_size)
        self.genre_emb = nn.Embedding(len(genres_vocab), emb_size)
        self.genre_fc = nn.Linear(emb_size, emb_size)
        self.sex_emb = nn.Embedding(vab_sex_len, emb_size)
        self.age_emb = nn.Embedding(vab_age_len, emb_size)
        self.work_emb = nn.Embedding(vab_work_len, emb_size)
        #
        self.user_emb = nn.Embedding(n_users, emb_size)
        self.item_emb = nn.Embedding(n_items, emb_size)
        self.k1 = nn.Linear(hidden_units + 7 * emb_size, emb_size)
        self.k2 = nn.Linear(hidden_units + 7 * emb_size, emb_size)

        self.k3 = nn.Linear(hidden_units + 7 * emb_size, emb_size)
        self.k4 = nn.Linear(hidden_units + 7 * emb_size, emb_size)

        self.k5 = nn.Linear(hidden_units + 7 * emb_size, emb_size)
        self.k6 = nn.Linear(hidden_units + 7 * emb_size, emb_size)
        self.k7 = nn.Linear(hidden_units + 7 * emb_size, emb_size)

        self.grucell = nn.GRUCell(input_size=emb_size * 2, hidden_size=hidden_units)
        self.linear = nn.Linear(hidden_units, n_items)
        self.dropout = nn.Dropout(dropout)
        self.user_dropout = nn.Dropout(user_dropout)

    def forward(self, user_vectors, item_vectors, title_ids, genres_ids, sex, age, work):
        batch_size, _ = user_vectors.size()
        user_vectors = user_vectors
        item_vectors = item_vectors
        sequence_size = user_vectors.size()[1]

        users = self.user_dropout(self.user_emb(user_vectors))
        items = self.item_emb(item_vectors)

        # title
        title_ids_list = []
        for title_item in title_ids:
            # batch,20,seq_len,=>20,seq_len
            title_ids_list.append(self.txt_cnn(title_item).unsqueeze(0))
        title_embs = torch.cat(title_ids_list, 0)
        # batch,20,emb_size
        # genres
        # shape:batch,20,6
        genres_embs = self.genre_emb(genres_ids)
        # batch,20,6,emb_size
        genres_embs = torch.sum(genres_embs, dim=-2, keepdim=False)
        sex = self.sex_emb(sex)
        age = self.age_emb(age)
        work = self.work_emb(work)

        h = torch.zeros(batch_size, self.hidden_units).to(device)
        h_t = h.unsqueeze(0)
        for i in range(sequence_size):
            # rectified linear user integration -> impede the user component when necessary
            rect_users = rectified_users(self, users[:, i, :], items[:, i, :], title_embs[:, i, :],
                                         genres_embs[:, i, :], sex[:, i, :], age[:, i, :], work[:, i, :], h)
            gru_input = torch.cat([rect_users, items[:, i, :]], dim=-1)
            h = self.grucell(gru_input, h)
            h_t = torch.cat([h_t, h.unsqueeze(0)], dim=0)
        ln_input = self.dropout(h_t[1:].transpose(0, 1))
        output_ln = self.linear(ln_input)

        output = F.log_softmax(output_ln, dim=-1)
        return output


def rectified_users(self, users, items, title_embs, genres_embs, sex, age, work, h):
    # Transform
    k1 = self.k1(torch.cat([users, items, title_embs, genres_embs, sex, age, work, h], dim=-1))
    k2 = self.k2(torch.cat([users, items, title_embs, genres_embs, sex, age, work, h], dim=-1))
    k3 = self.k3(torch.cat([users, items, title_embs, genres_embs, sex, age, work, h], dim=-1))
    k4 = self.k4(torch.cat([users, items, title_embs, genres_embs, sex, age, work, h], dim=-1))
    k5 = self.k5(torch.cat([users, items, title_embs, genres_embs, sex, age, work, h], dim=-1))
    k6 = self.k6(torch.cat([users, items, title_embs, genres_embs, sex, age, work, h], dim=-1))
    k7 = self.k7(torch.cat([users, items, title_embs, genres_embs, sex, age, work, h], dim=-1))
    rect_users = users

    # Leaky
    rect_users[users < k7] = rect_users[users < k7] * 0.2
    rect_users[users < k6] = rect_users[users < k6] * 0.2
    rect_users[users < k5] = rect_users[users < k5] * 0.2
    rect_users[users < k4] = rect_users[users < k4] * 0.2
    rect_users[users < k3] = rect_users[users < k3] * 0.2

    rect_users[users < k2] = rect_users[users < k2] * 0.2
    rect_users[users < k1] = 0
    return rect_users


# Libraries and provided functions
import pandas as pd
import zipfile
from io import StringIO
import numpy as np
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
from tqdm import \
    tqdm  # Very useful library to see progress bar during range iterations: just type `for i in tqdm(range(10)):`
from matplotlib import pyplot as plt


from collections import namedtuple
import sys


def drop_unused_items(train, val, test):
    train_items = train.itemid.unique()
    val_items = val.itemid.unique()
    test_items = test.itemid.unique()

    droped_items = list((set(val_items) | set(test_items)) - set(train_items))
    val_mask = val.userid == droped_items[0]
    test_mask = test.userid == droped_items[0]
    for droped_item in droped_items:
        val_mask += val.itemid == droped_item
        test_mask += test.itemid == droped_item
    val = val[~val_mask]
    test = test[~test_mask]
    return val, test


def move_timestamps_to_end(x, max_order):
    new_order = x.groupby('timestamp', sort=True).grouper.group_info[0]
    x["timestamp"] = (max_order - new_order.max()) + new_order
    return x


def normalize_timestamp(x):
    x["timestamp"] = x.groupby(['timestamp', 'itemid'], sort=True).grouper.group_info[0]
    return x


def set_timestamp_length(x):
    x['length'] = len(x)
    return x


def to_coo(data):
    user_idx, item_idx, feedback = data['userid'], data['itemid'], data['rating']
    return user_idx, item_idx, feedback


def to_matrices(data):
    data = split_by_groups(data)

    data_max_order = data['timestamp'].max()
    data = data.groupby("index").apply(move_timestamps_to_end, data_max_order)

    data_shape = data[['index', 'timestamp']].max() + 1
    data_matrix = sp.sparse.csr_matrix((data['itemid'],
                                        (data['index'], data['timestamp'])),
                                       shape=data_shape, dtype=np.float64).todense()
    mask_matrix = sp.sparse.csr_matrix((np.ones(len(data)),
                                        (data['index'], data['timestamp'])),
                                       shape=data_shape, dtype=np.float64).todense()

    data_users = data.drop_duplicates(['index'])
    user_data_shape = data_users['index'].max() + 1
    user_vector = sp.sparse.csr_matrix((data_users['userid'],
                                        (data_users['index'], np.zeros(user_data_shape))),
                                       shape=(user_data_shape, 1), dtype=np.float64).todense()
    user_matrix = np.tile(user_vector, (1, data_shape[1]))
    return data_matrix, mask_matrix, user_matrix


def train_val_test_split(data, frac):
    data = data.groupby("userid").apply(set_timestamp_length)
    max_time_stamp = data['length'] * frac
    timestamp = data['timestamp']
    data_train = data[timestamp < max_time_stamp * 0.9].groupby("userid").apply(normalize_timestamp)
    data_val = data[(0.9 * max_time_stamp <= timestamp) & (timestamp < 0.95 * max_time_stamp)]
    data_test = data[(0.95 * max_time_stamp <= timestamp) & (timestamp <= max_time_stamp)]

    data_val, data_test = drop_unused_items(data_train, data_val, data_test)
    data_val, data_test = data_val.groupby("userid").apply(normalize_timestamp), \
                          data_test.groupby("userid").apply(normalize_timestamp)

    return data_train, data_val, data_test


def split_by_groups(data, group_length=20):
    data["group"] = data['timestamp'] // group_length
    data["timestamp"] = data['timestamp'] % group_length
    data["index"] = data.groupby(['userid', 'group'], sort=False).grouper.group_info[0]
    return data


def get_prepared_data(data, frac=1):
    print("Normalizing indices to avoid gaps")
    # normalize indices to avoid gaps
    data['itemid'] = data.groupby('itemid', sort=False).grouper.group_info[0]
    data['userid'] = data.groupby('userid', sort=False).grouper.group_info[0]
    data = data.groupby("userid").apply(normalize_timestamp)

    # build sparse user-movie matrix
    print("Splitting into train, validation and test parts")

    data_train, data_val, data_test = train_val_test_split(data, frac)

    user_idx, item_idx, feedback = to_coo(data_train.copy())

    train_items, train_mask, train_users = to_matrices(data_train.copy())
    val_items, val_mask, val_users = to_matrices(data_val.copy())
    test_items, test_mask, test_users = to_matrices(data_test.copy())

    print('Done.')
    return (train_items, train_mask, train_users), \
           (val_items, val_mask, val_users), \
           (test_items, test_mask, test_users), \
           (user_idx, item_idx, feedback)


def get_movielens_data():
    '''Downloads movielens data, normalizes users, timesteps and movies ids,
    returns data in sparse CSR format.
    '''
    print('Loading data into memory...')
    with zipfile.ZipFile("ml-1m.zip") as zfile:
        zdata = zfile.read('ml-1m/ratings.dat').decode()
        delimiter = ';'
        zdata = zdata.replace('::', delimiter)  # makes data compatible with pandas c-engine
        ml_data = pd.read_csv(StringIO(zdata), sep=delimiter, header=None, engine='c',
                              names=['userid', 'movieid', 'rating', 'timestamp'],
                              usecols=['userid', 'movieid', 'rating', 'timestamp'])
        ml_data['itemid'] = ml_data['movieid']
    return ml_data

# Only need to run for the first time
ml_data = get_movielens_data()

(ml_train_items, ml_train_mask,ml_train_users),\
(ml_val_items, ml_val_mask,ml_val_users),\
(ml_test_items, ml_test_mask,ml_test_users),\
(ml_train_user_idx, ml_train_item_idx, ml_train_feedback) = get_prepared_data(ml_data)

# Save for future load
np.save("ml_train_items",ml_train_items)
np.save('ml_train_mask',ml_train_mask)
np.save('ml_train_users',ml_train_users)
np.save('ml_val_items',ml_val_items)
np.save('ml_val_mask',ml_val_mask)
np.save('ml_val_users',ml_val_users)
np.save('ml_test_items',ml_test_items)
np.save('ml_test_mask',ml_test_mask)
np.save('ml_test_users',ml_test_users)
np.save('ml_train_user_idx',ml_train_user_idx)
np.save('ml_train_item_idx',ml_train_item_idx)
np.save('ml_train_feedback',ml_train_feedback)

ml_train_items = np.load("ml_train_items.npy")
ml_train_mask = np.load("ml_train_mask.npy")
ml_train_users = np.load("ml_train_users.npy")
ml_val_items = np.load("ml_val_items.npy")
ml_val_mask = np.load("ml_val_mask.npy")
ml_val_users = np.load("ml_val_users.npy")
ml_test_items = np.load("ml_test_items.npy")
ml_test_mask = np.load("ml_test_mask.npy")
ml_test_users = np.load("ml_test_users.npy")

import gc
torch.cuda.empty_cache()
gc.collect()


def extract_movie_feature_ids_and_user(train_movie_items, test_movie_items, valid_movie_items, ml_train_users,
                                       ml_test_users, ml_val_users):
    movie_id = []
    movie_title = ['']
    movie_title_vocab = []
    movie_genres = ['']
    each_gen_list = []
    movie_feature_dict = {0: {'title': '', 'genres': ''}}

    # user -s
    dict_sex = {}
    dict_age = {}
    dict_work = {}
    #       1:  "Under 18"
    # 	* 18:  "18-24"
    # 	* 25:  "25-34"
    # 	* 35:  "35-44"
    # 	* 45:  "45-49"
    # 	* 50:  "50-55"
    # 	* 56:  "56+"
    #       	*  0:  "other" or not specified
    # 	*  1:  "academic/educator"
    # 	*  2:  "artist"
    # 	*  3:  "clerical/admin"
    # 	*  4:  "college/grad student"
    # 	*  5:  "customer service"
    # 	*  6:  "doctor/health care"
    # 	*  7:  "executive/managerial"
    # 	*  8:  "farmer"
    # 	*  9:  "homemaker"
    # 	* 10:  "K-12 student"
    # 	* 11:  "lawyer"
    # 	* 12:  "programmer"
    # 	* 13:  "retired"
    # 	* 14:  "sales/marketing"
    # 	* 15:  "scientist"
    # 	* 16:  "self-employed"
    # 	* 17:  "technician/engineer"
    # 	* 18:  "tradesman/craftsman"
    # 	* 19:  "unemployed"
    # 	* 20:  "writer"
    sex_label = {-1: 0, 0: 1, 1: 2}
    age_label = {-1: 0, 1: 1, 18: 2, 25: 3, 35: 4, 45: 5, 50: 6, 56: 7}
    file_ = open('ml-1m/users.dat')
    lines = file_.readlines()
    for line in lines:
        line = line.strip('\n').split('::')

        user_id = int(line[0].strip())
        sex = 0
        if (line[1] == 'M'):
            sex = 1
        dict_sex[user_id] = sex
        dict_age[user_id] = line[2]
        dict_work[user_id] = line[3]

    # 计算训练集词表长度
    vab_sex = []
    vab_age = []
    vab_work = []
    # train
    m, n = ml_train_users.shape
    ml_train_sex = np.zeros((m, n))
    ml_train_age = np.zeros((m, n))
    ml_train_work = np.zeros((m, n))

    for i in range(m):
        L = [int(i) for i in ml_train_users[i, :]]
        sex_L = []
        age_L = []
        work_L = []
        for x in L:
            sex = dict_sex.get(x)
            age = dict_age.get(x)
            work = dict_work.get(x)
            if (sex == None):
                sex = 0
            else:
                sex = sex_label.get(int(sex))
            if (age == None):
                age = 0
            else:
                age = age_label.get(int(age))
            if (work == None):
                work = -1

            if (int(sex) not in vab_sex):
                vab_sex.append(int(sex))
            if (int(age) not in vab_age):
                vab_age.append(int(age))
            if (int(work) not in vab_work):
                vab_work.append(int(work))

            sex_L.append(int(sex))
            age_L.append(int(age))
            work_L.append(int(work) + 1)
        ml_train_sex[i, :] = sex_L
        ml_train_age[i, :] = age_L
        ml_train_work[i, :] = work_L

    # test
    m, n = ml_test_users.shape
    ml_test_sex = np.zeros((m, n))
    ml_test_age = np.zeros((m, n))
    ml_test_work = np.zeros((m, n))
    for i in range(m):
        L = [int(i) for i in ml_test_users[i, :]]
        sex_L = []
        age_L = []
        work_L = []
        for x in L:
            sex = dict_sex.get(x)
            age = dict_age.get(x)
            work = dict_work.get(x)
            if (sex == None):
                sex = 0
            else:
                sex = sex_label.get(int(sex))
            if (age == None):
                age = 0
            else:
                age = age_label.get(int(age))
            if (work == None):
                work = -1
            sex_L.append(int(sex))
            age_L.append(int(age))
            work_L.append(int(work) + 1)
        ml_test_sex[i, :] = sex_L
        ml_test_age[i, :] = age_L
        ml_test_work[i, :] = work_L

    # val
    m, n = ml_val_users.shape
    ml_val_sex = np.zeros((m, n))
    ml_val_age = np.zeros((m, n))
    ml_val_work = np.zeros((m, n))
    for i in range(m):
        L = [int(i) for i in ml_val_users[i, :]]
        sex_L = []
        age_L = []
        work_L = []
        for x in L:
            sex = dict_sex.get(x)
            age = dict_age.get(x)
            work = dict_work.get(x)
            if (sex == None):
                sex = 0
            else:
                sex = sex_label.get(int(sex))
            if (age == None):
                age = 0
            else:
                age = age_label.get(int(age))
            if (work == None):
                work = -1
            sex_L.append(int(sex))
            age_L.append(int(age))
            work_L.append(int(work) + 1)
        ml_val_sex[i, :] = sex_L
        ml_val_age[i, :] = age_L
        ml_val_work[i, :] = work_L

    # user -e

    with open(r'ml-1m/movies.dat', encoding='utf-8') as f:
        for line in f.readlines():
            s = line.strip().split('::')
            movie_feature_dict[int(s[0])] = {'title': s[1], 'genres': s[-1].split('|')}
            movie_id += [int(s[0])]
            movie_title += [s[1]]
            movie_title_vocab.extend(s[1].split(' '))
            movie_genres += s[-1].split('|')
            each_gen_list.append(len(s[-1].split('|')))

    movie_genres_label2id = {j: i for i, j in enumerate(list(set(movie_genres)))}
    movie_title_label2id = {j: i for i, j in enumerate(list(set(movie_title)))}
    title_vocab_dict = {j: i for i, j in enumerate([''] + list(set(movie_title_vocab)))}

    # movie_title_label2id
    def title_genre_f(movie_items):
        title_ids_list = []
        genres_ids_list = []
        len_genre_list = []
        len_title_list = []
        for mids in movie_items:
            title_mids = []
            genres_ids = []
            for mid in mids:
                # 通过id找电影名 并把单词转换id
                mid = int(mid)
                temp_title_ids = [title_vocab_dict.get(i, 0) for i in
                                  movie_feature_dict.get(mid, movie_feature_dict[0])['title'].split(' ')]
                title_mids.append(temp_title_ids)
                len_title_list.append(len(temp_title_ids))
                # 通过id找到电影类型
                temp_genres_ids = [movie_genres_label2id.get(i, 0) for i in
                                   movie_feature_dict.get(mid, movie_feature_dict[0])['genres']]
                len_genre_list.append(len(temp_genres_ids))
                #             temp_genres_ids+=[0]*(max_genres_num-len(temp_genres_ids))
                genres_ids.append(temp_genres_ids)
            genres_ids_list.append(genres_ids)
            title_ids_list.append(title_mids)
        return title_ids_list, genres_ids_list, len_genre_list, len_title_list

    train_title_ids_list, train_genres_ids_list, len_genre_list, len_title_list = title_genre_f(train_movie_items)
    valid_title_ids_list, valid_genres_ids_list, *_t = title_genre_f(valid_movie_items)
    test_title_ids_list, test_genres_ids_list, *_ = title_genre_f(test_movie_items)
    # 标题、类别数量统一长度
    len_title = max(len_title_list)
    len_genre = max(len_genre_list)

    #
    def len_process(title_ids_list, genres_ids_list, len_title, len_genre):
        new_title_ids_list = []
        for title_ids in title_ids_list:  # genres_ids_list
            new_title_ids = []
            for ids in title_ids:
                if len_title - len(ids) >= 0:
                    ids += [0] * (len_title - len(ids))
                else:
                    ids = ids[:len_title]
                new_title_ids.append(ids)
            new_title_ids_list.append(new_title_ids)

        new_genres_ids_list = []
        for genres_ids in genres_ids_list:  # genres_ids_list
            new_genres_ids = []
            for ids in genres_ids:
                if len_genre - len(ids) >= 0:
                    ids += [0] * (len_genre - len(ids))
                else:
                    ids = ids[:len_genre]
                new_genres_ids.append(ids)
            new_genres_ids_list.append(new_genres_ids)
        return new_title_ids_list, new_genres_ids_list

    train_title_ids_list, train_genres_ids_list = len_process(train_title_ids_list, train_genres_ids_list, len_title,
                                                              len_genre)
    valid_title_ids_list, valid_genres_ids_list = len_process(valid_title_ids_list, valid_genres_ids_list, len_title,
                                                              len_genre)
    test_title_ids_list, test_genres_ids_list = len_process(test_title_ids_list, test_genres_ids_list, len_title,
                                                            len_genre)
    # test and valid
    #     print(len(vab_sex),len(vab_age),len(vab_work))
    return (train_title_ids_list, train_genres_ids_list, ml_train_sex, ml_train_age, ml_train_work,
            valid_title_ids_list, valid_genres_ids_list, ml_val_sex, ml_val_age, ml_val_work,
            test_title_ids_list, test_genres_ids_list, ml_test_sex, ml_test_age, ml_test_work,
            title_vocab_dict, movie_genres_label2id, len(vab_sex), len(vab_age), len(vab_work))


# ml_train_items = np.load("ml_train_items.npy")
# ml_train_mask = np.load("ml_train_mask.npy")
# ml_train_users = np.load("ml_train_users.npy")
# ml_val_items = np.load("ml_val_items.npy")
# ml_val_mask = np.load("ml_val_mask.npy")
# ml_val_users = np.load("ml_val_users.npy")
# ml_test_items = np.load("ml_test_items.npy")
# ml_test_mask = np.load("ml_test_mask.npy")
# ml_test_users = np.load("ml_test_users.npy")

(train_title_ids_list,train_genres_ids_list, ml_train_sex,ml_train_age,ml_train_work,
            valid_title_ids_list,valid_genres_ids_list, ml_val_sex,ml_val_age,ml_val_work,
            test_title_ids_list,test_genres_ids_list, ml_test_sex,ml_test_age,ml_test_work,
           title_vocab_dict,movie_genres_label2id,vab_sex_len,vab_age_len,vab_work_len) =extract_movie_feature_ids_and_user(ml_train_items,ml_test_items,ml_val_items,ml_train_users,ml_test_users,ml_val_users)

print(vab_sex_len,vab_age_len,vab_work_len)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_users = int(ml_test_users.max()+1)
n_items = int(np.max([ml_train_items.max()+1,ml_val_items.max()+1,ml_test_items.max()])+1)
# RectifiedLinearGRU   LinearGRU  AttentionalLinearGRU
network = RectifiedLinearGRU(n_users=n_users,n_items=n_items,title_vocab=title_vocab_dict, genres_vocab=movie_genres_label2id,vab_sex_len=vab_sex_len, vab_age_len=vab_age_len,vab_work_len=vab_work_len).to(device)

# n_users,n_items,title_vocab, genres_vocab,emb_size=None,vab_sex_len=None,vab_age_len=None,vab_work_len=None

import torch.utils.data

opt = torch.optim.Adam(network.parameters(), lr=0.001)

history = []

# train_loader = torch.utils.data.DataLoader(\
#             torch.utils.data.TensorDataset(\
#             *(torch.LongTensor(ml_train_users),torch.LongTensor(ml_train_items),torch.FloatTensor(ml_train_mask))),\
#             batch_size=800,shuffle=True)


train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        *(torch.LongTensor(ml_train_users),
          torch.LongTensor(ml_train_items),
          torch.FloatTensor(ml_train_mask),
          torch.LongTensor(train_title_ids_list),
          torch.LongTensor(train_genres_ids_list),

          torch.LongTensor(ml_train_sex),
          torch.LongTensor(ml_train_age),
          torch.LongTensor(ml_train_work)

          )),
    batch_size=800, shuffle=True)

val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        *(torch.LongTensor(ml_val_users),
          torch.LongTensor(ml_val_items),
          torch.FloatTensor(ml_val_mask),
          torch.LongTensor(valid_title_ids_list),
          torch.LongTensor(valid_genres_ids_list),

          torch.LongTensor(ml_val_sex),
          torch.LongTensor(ml_val_age),
          torch.LongTensor(ml_val_work)
          )),
    batch_size=800, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        *(torch.LongTensor(ml_test_users),
          torch.LongTensor(ml_test_items),
          torch.FloatTensor(ml_test_mask),
          torch.LongTensor(test_title_ids_list),
          torch.LongTensor(test_genres_ids_list),

          torch.LongTensor(ml_test_sex),
          torch.LongTensor(ml_test_age),
          torch.LongTensor(ml_test_work)
          )),
    batch_size=800, shuffle=True)

import torch.utils.data
from IPython.display import clear_output
import gc


def validate_lce(network, val_loader):
    torch.cuda.empty_cache()
    gc.collect()
    network.eval()
    losses = []
    with torch.no_grad():
        for user_batch_ix, item_batch_ix, mask_batch_ix, title_ids, genres_ids, sex, age, work in val_loader:
            user_batch_ix = Variable(user_batch_ix).to(device)
            item_batch_ix = Variable(item_batch_ix).to(device)
            mask_batch_ix = Variable(mask_batch_ix).to(device)
            title_ids_ix = Variable(title_ids).to(device)
            genres_ids_ix = Variable(genres_ids).to(device)

            sex = Variable(sex).to(device)
            age = Variable(age).to(device)
            work = Variable(work).to(device)

            logp_seq = network(user_batch_ix, item_batch_ix, title_ids_ix, genres_ids_ix, sex, age, work)
            # compute loss
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
            actual_next_tokens = item_batch_ix[:, 1:]

            logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, None])
            loss = -logp_next.sum() / mask_batch_ix[:, :-1].sum()
            losses.append(loss.cpu().data.numpy())
    torch.cuda.empty_cache()
    gc.collect()
    return np.mean(losses)


def train_network(network, train_loader, val_loader, num_epoch=10):
    for epoch in range(num_epoch):
        print('epoch:', epoch)
        i = 0
        for user_batch_ix, item_batch_ix, mask_batch_ix, title_ids, genres_ids, sex, age, work in train_loader:
            network.train()
            user_batch_ix = Variable(user_batch_ix).to(device)
            item_batch_ix = Variable(item_batch_ix).to(device)
            mask_batch_ix = Variable(mask_batch_ix).to(device)
            title_ids_ix = Variable(title_ids).to(device)
            genres_ids_ix = Variable(genres_ids).to(device)
            sex = Variable(sex).to(device)
            age = Variable(age).to(device)
            work = Variable(work).to(device)
            logp_seq = network(user_batch_ix, item_batch_ix, title_ids_ix, genres_ids_ix, sex, age, work)
            # compute loss
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
            actual_next_tokens = item_batch_ix[:, 1:]

            logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, None])
            loss = -logp_next.sum() / mask_batch_ix[:, :-1].sum()

            # train with backprop
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5)
            opt.step()

            if (i + 1) % 50 == 0:
                val_loss = validate_lce(network, val_loader)
                history.append(val_loss)

                clear_output(True)
                plt.title("Validation error")
                plt.plot(history)
                plt.ylabel('Cross-entropy Error')
                plt.xlabel('#iter')
                plt.show()
            i += 1

    val_loss = validate_lce(network, val_loader)
    history.append(val_loss)

    clear_output(True)
    plt.plot(history, label='val loss')
    plt.legend()
    plt.show()
import gc
torch.cuda.empty_cache()
gc.collect()
train_network(network,train_loader,val_loader)
torch.save(network,"network_linear_ml.p")

import numpy as np, scipy.stats as st
import numpy as np
import scipy as sp
import scipy.stats


# calculate the 95% confidence interval of the mean of data
def mean_confidence_interval(data, confidence=0.95, num_parts=5):
    part_len = len(data) // num_parts
    estimations = []
    for i in range(num_parts):
        est = np.mean(data[part_len * i:part_len * (i + 1)])
        estimations.append(est)
    a = 1.0 * np.array(estimations)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


# MRR@𝑘 (Mean Reciprocal Rank) is defined as the average of the reciprocal ranks of the desired items.
# The rank is set to zero if it is above 𝑘.
def validate_mrr(network, k, test_loader):
    network.eval()
    losses = []
    with torch.no_grad():
        for user_batch_ix, item_batch_ix, mask_batch_ix, title_ids, genres_ids, sex, age, work in test_loader:
            user_batch_ix = Variable(user_batch_ix).to(device)
            item_batch_ix = Variable(item_batch_ix).to(device)
            mask_batch_ix = Variable(mask_batch_ix).to(device)
            title_ids_ix = Variable(title_ids).to(device)
            genres_ids_ix = Variable(genres_ids).to(device)
            sex = Variable(sex).to(device)
            age = Variable(age).to(device)
            work = Variable(work).to(device)
            logp_seq = network(user_batch_ix, item_batch_ix, title_ids_ix, genres_ids_ix, sex, age, work)

            # logp_seq = network(user_batch_ix, item_batch_ix)
            # compute loss
            predictions_logp = logp_seq[:, -2]
            _, ind = torch.topk(predictions_logp, k, dim=-1)
            mrr = torch.zeros(predictions_logp.size())
            mrr.scatter_(-1, ind.cpu(), 1 / torch.range(1, k).repeat(*ind.size()[:-1], 1).type(torch.FloatTensor).cpu())
            actual_next_tokens = item_batch_ix[:, -1]

            logp_next = torch.gather(mrr.to(device) * mask_batch_ix[:, -2, None], dim=1,
                                     index=actual_next_tokens[:, None])
            loss = logp_next.sum() / mask_batch_ix[:, -2].sum()
            losses.append(loss.cpu().data.numpy())
            torch.cuda.empty_cache()
            gc.collect()
    m, h = mean_confidence_interval(losses)
    return m, h


# Recall@𝑘 is defined as the fraction of cases where the item actually consumed in the next event
# among the top 𝑘 items recommended
def validate_recall(network, k, test_loader):
    torch.cuda.empty_cache()
    gc.collect()

    network.eval()
    losses = []
    with torch.no_grad():
        for user_batch_ix, item_batch_ix, mask_batch_ix, title_ids, genres_ids, sex, age, work in test_loader:
            user_batch_ix = Variable(user_batch_ix).to(device)
            item_batch_ix = Variable(item_batch_ix).to(device)
            mask_batch_ix = Variable(mask_batch_ix).to(device)
            title_ids_ix = Variable(title_ids).to(device)
            genres_ids_ix = Variable(genres_ids).to(device)
            sex = Variable(sex).to(device)
            age = Variable(age).to(device)
            work = Variable(work).to(device)
            logp_seq = network(user_batch_ix, item_batch_ix, title_ids_ix, genres_ids_ix, sex, age, work)
            # compute loss
            predictions_logp = logp_seq[:, -2]
            minus_kth_biggest_logp, _ = torch.kthvalue(-predictions_logp.cpu(), k, dim=-1, keepdim=True)
            prediicted_kth_biggest = (predictions_logp > (-minus_kth_biggest_logp.to(device))).type(
                torch.FloatTensor).to(device)
            actual_next_tokens = item_batch_ix[:, -1]

            logp_next = torch.gather(prediicted_kth_biggest * mask_batch_ix[:, -2, None], dim=1,
                                     index=actual_next_tokens[:, None])
            loss = logp_next.sum() / mask_batch_ix[:, -2].sum()
            losses.append(loss.cpu().data.numpy())
    torch.cuda.empty_cache()
    gc.collect()
    m, h = mean_confidence_interval(losses)
    return m, h


def print_scores(model, name):
    network = torch.load(model).to(device)
    mrr_score, h = validate_mrr(network, 20, test_loader)
    print("MRR@20 score for ", name, ": ", mrr_score, "±", h)
    recall_score, h = validate_recall(network, 20, test_loader)
    print("Recall@20 score for " + name + ": ", recall_score, "±", h)

print_scores("network_linear_ml.p",'Linear User-based GRU on MovieLens')


