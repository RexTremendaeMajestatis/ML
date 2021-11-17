from os import path

from numpy import inf

cur_dir = path.dirname(__file__)
dataset_dir = path.join(cur_dir, 'dataset')
dataset_name = 'combined_data_1.txt'
dataset_path = path.join(dataset_dir, dataset_name)

csr_path = path.join(cur_dir, 'answers', 'csr_{0}'.format(dataset_name))

u_count = 2649429
m_count = 17770
k = 2
row_count = inf