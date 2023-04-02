import math
import numpy as np
from numpy import linalg
from util.utils import recv_msg, send_msg, moving_average
from config import tau_max


class ControlAlgAdaptiveTauServer:
    def __init__(self, is_adapt_local, dim_w, client_sock_all, n_nodes, control_param_phi,
                 moving_average_holding_param):

        self.is_adapt_local = is_adapt_local
        self.dim_w = dim_w
        self.client_sock_all = client_sock_all
        self.n_nodes = n_nodes
        self.control_param_phi = control_param_phi
        self.moving_average_holding_param = moving_average_holding_param

        self.beta_adapt_mvaverage = None
        self.delta_adapt_mvaverage = None
        self.rho_adapt_mvaverage = None
        self.c_adapt_mvaverage = None

    def compute_new_tau(self, data_size_local_all, data_size_total, it_each_local, it_each_global, max_time,
                        step_size, tau, use_min_loss, first_global_loss):

        beta_adapt = 0
        delta_adapt = 0
        rho_adapt = 0
        c_adapt = 0
        global_grad_global_weight = np.zeros(self.dim_w)

        local_grad_global_weight_all = []

        control_param_computed = False
        for n in range(0, self.n_nodes):
            msg = recv_msg(self.client_sock_all[n], 'MSG_CONTROL_PARAM_COMPUTED_CLIENT_TO_SERVER')
            # ['MSG_CONTROL_PARAM_COMPUTED_CLIENT_TO_SERVER', control_param_computed]
            control_param_computed_this_client = msg[1]  # Boolean parameter specifying whether parameters for
                                                         # control algorithm follows this message

            control_param_computed = control_param_computed or control_param_computed_this_client

            # Receive additional message for control algorithm if it has been computed
            if control_param_computed_this_client:
                msg = recv_msg(self.client_sock_all[n], 'MSG_BETA_RHO_GRAD_CLIENT_TO_SERVER')
                # ['MSG_BETA_RHO_GRAD_CLIENT_TO_SERVER', betaAdapt, rhoAdapt, localGradGlobalWeight]

                beta_adapt_local = msg[1]
                rho_adapt_local = msg[2]
                local_grad_global_weight = msg[3]
                c_adapt_local = msg[4]

                local_grad_global_weight_all.append(local_grad_global_weight)

                beta_adapt += data_size_local_all[n] * beta_adapt_local
                rho_adapt += data_size_local_all[n] * rho_adapt_local
                global_grad_global_weight += data_size_local_all[n] * local_grad_global_weight
                c_adapt += data_size_local_all[n] * c_adapt_local

        global_grad_global_weight /= data_size_total

        if control_param_computed and (it_each_local is not None) and (it_each_global is not None):
            # finalize beta and delta computation when using control algorithm
            beta_adapt /= data_size_total
            rho_adapt /= data_size_total
            c_adapt /= data_size_total

            for i in range(0, self.n_nodes):
                delta_adapt += data_size_local_all[i] * linalg.norm(local_grad_global_weight_all[i]
                                                                    - global_grad_global_weight)
            delta_adapt /= data_size_total

            # compute moving averages
            self.beta_adapt_mvaverage = moving_average(self.beta_adapt_mvaverage, beta_adapt, self.moving_average_holding_param)
            self.delta_adapt_mvaverage = moving_average(self.delta_adapt_mvaverage, delta_adapt, self.moving_average_holding_param)
            self.rho_adapt_mvaverage = moving_average(self.rho_adapt_mvaverage, rho_adapt, self.moving_average_holding_param)
            self.c_adapt_mvaverage = moving_average(self.c_adapt_mvaverage, c_adapt, self.moving_average_holding_param)

            print('betaAdapt_mvaverage =', self.beta_adapt_mvaverage)
            print('deltaAdapt_mvaverage =', self.delta_adapt_mvaverage)
            print('rhoAdapt_mvaverage =', self.rho_adapt_mvaverage)
            print('cAdapt_mvaverage =', self.c_adapt_mvaverage)

            # Find tau if using control algorithm
            if self.is_adapt_local:

                # Find new optimal tau
                min_tau_new_tmp = 1
                min_val = float('inf')

                for tau_new_tmp in range(1, 11):
                    h_tau_tmp = max(0.0, (self.delta_adapt_mvaverage / self.beta_adapt_mvaverage) * (
                    np.power(step_size * self.beta_adapt_mvaverage + 1,
                             tau_new_tmp) - 1) - self.delta_adapt_mvaverage * step_size * tau_new_tmp)

                    # The below lines are the new expression, with betaAdapt and rhoAdapt, and additional term of ht

                    q = 1 - step_size * self.c_adapt_mvaverage
                    K = 1000
                    T = K * tau_new_tmp
                    ts = 120
                    s = int(ts/tau_new_tmp)

                    tmp_gap = (q ** T) * first_global_loss + (1 - q ** T)/(1 - q ** tau_new_tmp)*(
                            (1 - q ** tau_new_tmp) * self.beta_adapt_mvaverage * (step_size ** 2)*sum(
                        [i**2 for i in data_size_local_all])/((1-q**tau_new_tmp)*(1-q)*2*data_size_total**2*s)+
                            self.rho_adapt_mvaverage * h_tau_tmp)

                    if tmp_gap < min_val:
                        min_val = tmp_gap
                        min_tau_new_tmp = tau_new_tmp

                tau_new = min_tau_new_tmp
            else:
                tau_new = tau
        else:
            tau_new = tau

        return min(tau_new, tau_max)

    def __getstate__(self):
        # To remove socket from pickle
        state = self.__dict__.copy()
        del state['client_sock_all']
        return state


class ControlAlgAdaptiveTauClient:
    def __init__(self):
        self.w_last_local_last_round = None
        self.grad_last_local_last_round = None
        self.loss_last_local_last_round = None

    def init_new_round(self, w):
        self.control_param_computed = False
        self.beta_adapt = None
        self.rho_adapt = None
        self.grad_last_global = None
        self.c_adapt = None

    def update_after_each_local(self, iteration_index, w, grad, total_iterations):
        if iteration_index == 0:
            self.grad_last_global = grad

        return False

    def update_after_all_local(self, model, train_image, train_label, train_indices,
                               w, w_last_global, loss_last_global):

        # Only compute beta and rho locally, delta can only be computed globally
        if (self.w_last_local_last_round is not None) and (self.grad_last_local_last_round is not None) and \
                (self.loss_last_local_last_round is not None):

            # compute beta
            c = self.grad_last_local_last_round - self.grad_last_global
            tmp_norm = linalg.norm(self.w_last_local_last_round - w_last_global)
            if tmp_norm > 1e-10:
                self.beta_adapt = linalg.norm(c) / tmp_norm
            else:
                self.beta_adapt = 0

            # Compute rho
            if tmp_norm > 1e-10:
                self.rho_adapt = linalg.norm(self.loss_last_local_last_round - loss_last_global) / tmp_norm
            else:
                self.rho_adapt = 0

            if self.beta_adapt < 1e-5 or np.isnan(self.beta_adapt):
                self.beta_adapt = 1e-5

            if np.isnan(self.rho_adapt):
                self.rho_adapt = 0

            print('betaAdapt =', self.beta_adapt)

            # Compute convexity constant c

            self.c_adapt = linalg.norm(self.grad_last_local_last_round)**2/self.loss_last_local_last_round

            print('Convexity Constant c= ', self.c_adapt)


            self.control_param_computed = True

        self.grad_last_local_last_round = model.gradient(train_image, train_label, w, train_indices)

        try:
            self.loss_last_local_last_round = model.loss_from_prev_gradient_computation()
        except:  # Will get an exception if the model does not support computing loss from previous gradient computation
            self.loss_last_local_last_round = model.loss(train_image, train_label, w, train_indices)

        self.w_last_local_last_round = w

    def send_to_server(self, sock):

        msg = ['MSG_CONTROL_PARAM_COMPUTED_CLIENT_TO_SERVER', self.control_param_computed]
        send_msg(sock, msg)

        if self.control_param_computed:
            msg = ['MSG_BETA_RHO_GRAD_CLIENT_TO_SERVER', self.beta_adapt, self.rho_adapt, self.grad_last_global, self.c_adapt]
            send_msg(sock, msg)

