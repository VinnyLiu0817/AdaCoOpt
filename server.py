import socket
import time
from config import *
import numpy as np
if use_dynamic_tau is False:
    from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauServer
else:
    from control_algorithm.default_adaptive_tau import ControlAlgAdaptiveTauServer
from data_reader.data_reader import get_data
from models.get_model import get_model
from statistic.collect_stat import CollectStatistics
from util.utils import send_msg, recv_msg, get_indices_each_node_case

# Configurations are in a separate config.py file



model = get_model(model_name)
if hasattr(model, 'create_graph'):
    model.create_graph(learning_rate=step_size)

if time_gen is not None:
    use_fixed_averaging_slots = True
else:
    use_fixed_averaging_slots = False

if batch_size < total_data:   # Read all data once when using stochastic gradient descent
    train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data, dataset_file_path)

    # This function takes a long time to complete,
    # putting it outside of the sim loop because there is no randomness in the current way of computing the indices
    indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# listening_sock.bind(('192.168.1.41', 49999))
listening_sock.bind((SERVER_ADDR, SERVER_PORT))
#listening_sock.bind(('10.0.0.21', 51000))
client_sock_all=[]

# Establish connections to each client, up to n_nodes clients
while len(client_sock_all) < n_nodes:
    listening_sock.listen(5)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept()
    print('Got connection from ', (ip,port))
    print(client_sock)

    client_sock_all.append(client_sock)

if single_run:
    stat = CollectStatistics(results_file_name=single_run_results_file_path, is_single_run=True)
else:
    stat = CollectStatistics(results_file_name=multi_run_results_file_path, is_single_run=False)

for sim in sim_runs:

    if batch_size >= total_data:  # Read data again for different sim. round
        train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data, dataset_file_path, sim_round=sim)

        # This function takes a long time to complete
        indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)

    for case in case_range:

        for tau_setup in tau_setup_all:

            stat.init_stat_new_global_round()

            dim_w = model.get_weight_dimension(train_image, train_label)
            w_global_init = model.get_init_weight(dim_w, rand_seed=sim)
            w_global = w_global_init

            w_global_min_loss = None
            loss_min = np.inf
            prev_loss_is_min = False

            if tau_setup < 0:
                is_adapt_local = True
                tau_config = 1
            else:
                is_adapt_local = False
                tau_config = tau_setup

            if is_adapt_local or estimate_beta_delta_in_all_runs:

                if tau_setup == -1:
                    control_alg = ControlAlgAdaptiveTauServer(is_adapt_local, dim_w, client_sock_all, n_nodes,
                                                              control_param_phi, moving_average_holding_param)
                else:
                    raise Exception('Invalid setup of tau.')
            else:
                control_alg = None

            # Message server to client
            datanum = []
            lr_list = [step_size] * n_nodes
            for n in range(0, n_nodes):
                datanum.append(len(indices_each_node_case[case][n]))
            print('Data distribution:',datanum)
            total_data = sum(datanum)
            print('Total data:', total_data)

            # Caluculate step_size
            if bs_dis is True:
                for n in range(0, n_nodes):
                    lr_list[n] = (step_size * datanum[n] / (total_data / n_nodes))

            # if no_strag is True:
                # lr_list = [0.002,0.002,0.002,0.002,0.002,0.002,0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.008,0.008,0.008,0.008,0.008,0.008]

            print('Step size:', lr_list)

            for n in range(0, n_nodes):
                indices_this_node = indices_each_node_case[case][n]
                msg = ['MSG_INIT_SERVER_TO_CLIENT', model_name, dataset,
                       num_iterations_with_same_minibatch_for_tau_equals_one, lr_list[n],
                       batch_size, total_data, control_alg, indices_this_node, read_all_data_for_stochastic,
                       use_min_loss, sim, init_buff[n], buff_size[n]]
                send_msg(client_sock_all[n], msg)

            print('All clients connected')

            # Wait until all clients complete data preparation and sends a message back to the server
            for n in range(0, n_nodes):
                recv_msg(client_sock_all[n], 'MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER')

            print('Start learning')

            time_global_aggregation_all = None

            total_time = 0      # Actual total time, where use_fixed_averaging_slots has no effect
            total_time_recomputed = 0  # Recomputed total time using estimated time for each local and global update,
                                        # using predefined values when use_fixed_averaging_slots = true
            it_each_local = None
            it_each_global = None

            is_last_round = False
            is_eval_only = False

            tau_new_resume = None
            current_K = 0  # 表示第几轮
            first_local_loss = 0
            first_global_loss = 0
            bs_list = [batch_size] * n_nodes  # Initialize the bs-list

            # Loop for multiple rounds of local iterations + global aggregation
            while True:
                current_K += 1

                print('---------------------------------------------------------------------------')
                print('Round ', current_K)
                print('current tau config:', tau_config)
                print('current bs distribution',bs_list)
                print('current lr schedule:', lr_list)
                time_total_all_start = time.time()

                for n in range(0, n_nodes):
                    if stream_mode is True and len(arrive_round[n]) != 0 and current_K == arrive_round[n][0]:
                        msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau_config, is_last_round, prev_loss_is_min, bs_list[n], arrive_num[n][0]]
                        arrive_round[n].pop(0)
                        arrive_num[n].pop(0)
                    else:
                        msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau_config, is_last_round, prev_loss_is_min, bs_list[n], 0]
                    send_msg(client_sock_all[n], msg)

                w_global_prev = w_global

                print('Waiting for local iteration at client')

                w_global = np.zeros(dim_w)
                loss_last_global = 0.0
                loss_w_prev_min_loss = 0.0
                received_loss_local_w_prev_min_loss = False
                data_size_total = 0
                time_all_local_all = 0
                data_size_local_all = []
                p_all = []  # Capacity list
                tau_actual = 0

                for n in range(0, n_nodes):
                    msg = recv_msg(client_sock_all[n], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')
                    # ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w, time_all_local, tau_actual, data_size_local,
                    # loss_last_global, loss_w_prev_min_loss]
                    w_local = msg[1]
                    time_all_local = msg[2]
                    tau_actual = max(tau_actual, msg[3])  # Take max of tau because we wait for the slowest node
                    data_size_local = msg[4]
                    loss_local_last_global = msg[5]
                    loss_local_w_prev_min_loss = msg[6]
                    p = msg[7]

                    w_global += w_local * data_size_local
                    data_size_local_all.append(data_size_local)
                    data_size_total += data_size_local
                    time_all_local_all = max(time_all_local_all, time_all_local)   #Use max. time to take into account the slowest node
                    p_all.append(p)

                    if no_strag is True:
                        p_total = sum(p_all)
                        tot = batch_size * n_nodes
                        bs_list = [int(tot * i / p_total) for i in p_all]

                    if use_min_loss:
                        loss_last_global += loss_local_last_global * data_size_local
                        if loss_local_w_prev_min_loss is not None:
                            loss_w_prev_min_loss += loss_local_w_prev_min_loss * data_size_local
                            received_loss_local_w_prev_min_loss = True

                    else:
                        first_local_loss += loss_local_last_global * data_size_local

                print("Data_size_local_all:", data_size_local_all)
                first_global_loss = first_local_loss / sum(data_size_local_all)
                first_local_loss = 0
                print('Current global training loss is: ', first_global_loss)
                w_global /= data_size_total

                # Adaptive adjust step_size
                if bs_dis is True and adaptive_lr is True:
                    for n in range(0, n_nodes):
                        lr_list[n] = step_size * n_nodes * data_size_local_all[n] / sum(data_size_local_all)

                if True in np.isnan(w_global):
                    print('*** w_global is NaN, using previous value')
                    w_global = w_global_prev   # If current w_global contains NaN value, use previous w_global
                    use_w_global_prev_due_to_nan = True
                else:
                    use_w_global_prev_due_to_nan = False

                if use_min_loss:
                    loss_last_global /= data_size_total

                    if received_loss_local_w_prev_min_loss:
                        loss_w_prev_min_loss /= data_size_total
                        loss_min = loss_w_prev_min_loss

                    if loss_last_global < loss_min:
                        loss_min = loss_last_global
                        w_global_min_loss = w_global_prev
                        prev_loss_is_min = True
                    else:
                        prev_loss_is_min = False

                    print("Loss of previous global value: " + str(loss_last_global))
                    print("Minimum loss: " + str(loss_min))

                # If use_w_global_prev_due_to_nan, then use tau = 1 for next round
                if not use_w_global_prev_due_to_nan:
                    if control_alg is not None:
                        if use_dynamic_tau is False:
                            # Only update tau if use_w_global_prev_due_to_nan is False
                            [tau_new, bs_list] = control_alg.compute_new_tau(data_size_local_all, data_size_total,
                                                                  it_each_local, it_each_global, max_time,
                                                                  step_size, tau_config, use_min_loss, first_global_loss, p_all, bs_list)
                        else:
                            tau_new = control_alg.compute_new_tau(data_size_local_all, data_size_total,
                                                                  it_each_local, it_each_global, max_time,
                                                                  step_size, tau_config, use_min_loss)
                            bs_list = [int(ts/tau_new)] * n_nodes
                    else:
                        if bs_dis is True:  # No adaptive alg, but still our proposed bs-dis strategy
                            s_tot = batch_size * n_nodes
                            bs_list = [int(s_tot * i / total_data + 1) for i in data_size_local_all]
                        if tau_new_resume is not None:
                            tau_new = tau_new_resume
                            tau_new_resume = None
                        else:
                            tau_new = tau_config
                else:
                    if tau_new_resume is None:
                        tau_new_resume = tau_config
                    tau_new = 1

                # Calculate time
                time_total_all_end = time.time()
                time_total_all = time_total_all_end - time_total_all_start
                time_global_aggregation_all = max(0.0, time_total_all - time_all_local_all)

                print('Time for one global round:', time_all_local_all)
                print('Time for global averaging:', time_global_aggregation_all)

                if use_fixed_averaging_slots: # Not using real time
                    if isinstance(time_gen, (list,)):
                        t_g = time_gen[case]
                    else:
                        t_g = time_gen
                    it_each_local = max(0.00000001, np.sum(t_g.get_local(tau_actual)) / tau_actual)
                    it_each_global = t_g.get_global(1)[0]
                else: # Using the real time
                    it_each_local = max(0.00000001, time_all_local_all)
                    it_each_global = time_global_aggregation_all

                #Compute number of iterations is current slot
                total_time_recomputed += it_each_local # Only computing time
                #total_time_recomputed += it_each_local + it_each_global

                #Compute time in current slot
                total_time += time_total_all
                stat.collect_stat_end_local_round(case, tau_actual, it_each_local, it_each_global, control_alg, model,
                                                      train_image, train_label, test_image, test_label, w_global,
                                                      total_time_recomputed, first_global_loss, current_K)

                # Check remaining resource budget (use a smaller tau if the remaining time is not sufficient)
                # is_last_round_tmp = False
                #
                # if use_min_loss:
                #     tmp_time_for_executing_remaining = total_time_recomputed + it_each_local * (tau_new + 1) + it_each_global * 2
                # else:
                #     tmp_time_for_executing_remaining = total_time_recomputed + it_each_local * tau_new + it_each_global
                #
                # if tmp_time_for_executing_remaining < max_time:
                #     tau_config = tau_new
                # else:
                #     if use_min_loss:  # Take into account the additional communication round in the end
                #         tau_config = int((max_time - total_time_recomputed - 2 * it_each_global - it_each_local) / it_each_local)
                #     else:
                #         tau_config = int((max_time - total_time_recomputed - it_each_global) / it_each_local)
                #
                #     if tau_config < 1:
                #         tau_config = 1
                #     elif tau_config > tau_new:
                #         tau_config = tau_new
                #
                #     is_last_round_tmp = True

                if current_K == total_k:
                    break
                else:
                    tau_config = tau_new

                if is_eval_only:
                    tau_config = 1
                    is_last_round = True

                # if is_last_round_tmp:
                #     if use_min_loss:
                #         is_eval_only = True
                #     else:
                #         is_last_round = True

            if use_min_loss:
                w_eval = w_global_min_loss
            else:
                w_eval = w_global

            stat.collect_stat_end_global_round(sim, case, tau_setup, total_time, model, train_image, train_label,
                                               test_image, test_label, w_eval, total_time_recomputed)
