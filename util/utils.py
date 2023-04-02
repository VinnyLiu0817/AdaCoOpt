import numpy as np
import pickle, struct, socket, math

def get_random_data_arrive(cli_num, round_num, total_k, data_total):
    arrive_round = []
    arrive_num = []
    for i in range(cli_num):
        round = list(np.random.randint(1, total_k, round_num))
        round.sort()
        arrive_round.append(round)

        m = list(np.random.randint(1,data_total,round_num))
        sum_m = sum(m)
        if sum(m) >= data_total:
            for i in range(len(m)):
                m[i] = int(m[i] / sum_m * data_total)

        while sum(m) < data_total:
            idx = int(np.random.randint(0,len(m),1))
            m[idx] += 1

        arrive_num.append(m)

    print("Arrive round:", arrive_round)
    print("Arrive data_num:", arrive_num)

    return [arrive_round, arrive_num]


def get_even_odd_from_one_hot_label(label):
    for i in range(0, len(label)):
        if label[i] == 1:
            c = i % 2
            if c == 0:
                c = 1
            elif c == 1:
                c = -1
            return c


def get_index_from_one_hot_label(label):
    for i in range(0, len(label)):
        if label[i] == 1:
            return [i]


def get_one_hot_from_label_index(label, number_of_labels=10):
    one_hot = np.zeros(number_of_labels)
    one_hot[label] = 1
    return one_hot


def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername())


def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    print(msg[0], 'received from', sock.getpeername())

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg


def moving_average(param_mvavr, param_new, movingAverageHoldingParam):
    if param_mvavr is None or np.isnan(param_mvavr):
        param_mvavr = param_new
    else:
        if not np.isnan(param_new):
            param_mvavr = movingAverageHoldingParam * param_mvavr + (1 - movingAverageHoldingParam) * param_new
    return param_mvavr


def get_indices_each_node_case(n_nodes, maxCase, label_list):
    indices_each_node_case = []
    for i in range(0, maxCase):
        indices_each_node_case.append([])

    for i in range(0, n_nodes):
        for j in range(0, maxCase):
            indices_each_node_case[j].append([])

    # indices_each_node_case is a big list that contains N-number of sublists. Sublist n contains the indices that should be assigned to node n

    min_label = min(label_list)
    max_label = max(label_list)
    num_labels = max_label - min_label + 1
    order_label_list = [[] for i in range(10)]

    for i in range(0, len(label_list)):
        # case 1
        indices_each_node_case[0][(i % n_nodes)].append(i)

        # case 2
        tmp_target_node = int((label_list[i] - min_label) % n_nodes)
        if n_nodes > num_labels:
            tmp_min_index = 0
            tmp_min_val = math.inf
            for n in range(0, n_nodes):
                if n % num_labels == tmp_target_node and len(indices_each_node_case[1][n]) < tmp_min_val:
                    tmp_min_val = len(indices_each_node_case[1][n])
                    tmp_min_index = n
            tmp_target_node = tmp_min_index
        indices_each_node_case[1][tmp_target_node].append(i)

        # case 3
        for n in range(0, n_nodes):
            indices_each_node_case[2][n].append(i)

        # case 4
        tmp = int(np.ceil(min(n_nodes, num_labels) / 2))
        if label_list[i] < (min_label + max_label) / 2:
            tmp_target_node = i % tmp
        elif n_nodes > 1:
            tmp_target_node = int(((label_list[i] - min_label) % (min(n_nodes, num_labels) - tmp)) + tmp)

        if n_nodes > num_labels:
            tmp_min_index = 0
            tmp_min_val = math.inf
            for n in range(0, n_nodes):
                if n % num_labels == tmp_target_node and len(indices_each_node_case[3][n]) < tmp_min_val:
                    tmp_min_val = len(indices_each_node_case[3][n])
                    tmp_min_index = n
            tmp_target_node = tmp_min_index

        indices_each_node_case[3][tmp_target_node].append(i)

        # case 5: 20 clients distribution for CIFAR
        if i < 5000:
            for n in range(0, n_nodes):
                indices_each_node_case[4][n].append(i)
        elif i < 20000:
            for n in range(0, n_nodes - 6):
                indices_each_node_case[4][n].append(i)
        else:
            for n in range(0, n_nodes - 13):
                indices_each_node_case[4][n].append(i)

        # case 6: 20 clients distribution for CIFAR (Extreme number distribution)
        # if i < 46000:
        #     client = i % 3
        #     indices_each_node_case[5][client].append(i)
        # else:
        #     client = i % 16 + 3
        #     indices_each_node_case[5][client].append(i)

        # case 6: 20 clients distribution for CIFAR (Extreme Class distribuiton 1)

        if label_list[i] <=5:
            indices_each_node_case[5][0].append(i)
            indices_each_node_case[5][1].append(i)
            indices_each_node_case[5][2].append(i)
        else:
            indices_each_node_case[5][(i % n_nodes)].append(i)

        # case 6: 20 clients distribution for CIFAR (Extreme Class distribuiton 2 bad)

        # indices_each_node_case[5][0].append(i)
        # indices_each_node_case[5][1].append(i)
        # indices_each_node_case[5][2].append(i)
        # if label_list[i] <= 2:
        #     for n in range(3, n_nodes):
        #         indices_each_node_case[5][n].append(i)

        # case 6: 20 clients distribution for CIFAR (Extreme Class distribuiton 3)
        # indices_each_node_case[5][0].append(i)
        # indices_each_node_case[5][label_list[i] + 1].append(i)
        # indices_each_node_case[5][label_list[i] + 9].append(i)

        # case 6: 20 clients distribution for CIFAR (Extreme Class distribuiton 4)
        # indices_each_node_case[5][0].append(i)
        # indices_each_node_case[5][1].append(i)
        # indices_each_node_case[5][2].append(i)
        # indices_each_node_case[5][3].append(i)
        # if label_list[i] <= 4:
        #     indices_each_node_case[5][label_list[i] + 4].append(i)
        #     indices_each_node_case[5][label_list[i] + 9].append(i)
        #     indices_each_node_case[5][label_list[i] + 14].append(i)

        # case 6: 20 clients distribution for CIFAR (Extreme Class distribuiton 5)
        # indices_each_node_case[5][0].append(i)
        # if i < 18000:
        #     indices_each_node_case[5][i % 18 + 1].append(i)

        # case 7.1: Clients with data samples in the order of classes

        order_label_list[int(label_list[i])].append(i)
        if i == (len(label_list) - 1):
            for n in range(0, n_nodes):
                order_list = []
                # np.random.RandomState(seed= n+1).shuffle(order_label_list)
                for j in range(len(order_label_list)):
                    order_list = order_list + order_label_list[j]
                indices_each_node_case[6][n] = order_list

    # 对 case3的全部数据进行打乱
    for n in range(0, n_nodes):
        np.random.RandomState(seed=n + 1).shuffle(indices_each_node_case[2][n])

    return indices_each_node_case
