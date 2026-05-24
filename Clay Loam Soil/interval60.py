#!/usr/bin/env python
# coding: utf-8

# uninstall tensorflow 2.x on Google Colab
# %tensorflow_version 2.x
# !pip uninstall -y tensorflow
# !pip install tensorflow-gpu==1.14.0

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
import random
import pickle



# 加载配置
with open('config.json', 'r') as f:
    VG_PARAMS = json.load(f)

def calc_nef(y_true, y_pred):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

def calc_rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def get_theoretical_curves(psi_positive):
    """基于吸力计算理论 Theta 和 K"""
    tr, ts, a, n2, Ks, l = VG_PARAMS['theta_r'], VG_PARAMS['theta_s'], VG_PARAMS['alpha'], VG_PARAMS['n2'], VG_PARAMS['Ks'], VG_PARAMS['l']
    m = 1 - 1/n2
    actual_theta = tr + (ts - tr) / (1 + (a * psi_positive)**n2)**m
    Se = (actual_theta - tr) / (ts - tr)
    actual_K = Ks * Se**l * (1 - (1 - Se**(1/m))**m)**2
    return actual_theta, actual_K

# 预生成验证网格 (0.01 - 100 cm)
psi_eval_pos = np.arange(0.01, 100.01, 0.01)
psi_eval_neg = -psi_eval_pos.reshape(-1, 1)
theta_true_eval, K_true_eval = get_theoretical_curves(psi_eval_pos)



# 确保使用的是 TensorFlow 1.x
# tf.__version__ # tensorflow 1.x

# --- 辅助函数：初始化 MNN (Monotonic Neural Network) ---
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev = xavier_stddev), dtype = tf.float32)

def initialize_MNN(layers, init_func): 
    """Monotonic neural network initialization function."""
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0, num_layers-1):
        W = init_func(size = [layers[l],layers[l+1]])
        W2 = W**2 # Squared weights for monotonicity
        b = tf.Variable(tf.zeros([1, layers[l+1]], dtype = tf.float32), dtype = tf.float32)
        weights.append(W2)
        biases.append(b)
    return weights, biases

def initialize_NN(layers, init_func): # stndard neural network
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0, num_layers-1):
        W = init_func(size = [layers[l],layers[l+1]])
        b = tf.Variable(tf.zeros([1, layers[l+1]], dtype = tf.float32), dtype = tf.float32)
        weights.append(W)
        biases.append(b)
    return weights, biases

# --- WRC Network Function ---
def net_theta(X, weights, biases): 
    num_layers = len(weights) + 1
    H = X
    for l in range(0, num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    # theta is constrained between 0 and 0.5 (as a typical example, adjust 0.5 if max theta is different)
    theta = 0.5 * tf.sigmoid(tf.add(tf.matmul(H, W), b)) 
    return theta




# ====================================================================
# STEP 1: WRC 训练类 (仅数据损失和梯度损失)
# ====================================================================
class WRC_NN:
    def __init__(self, theta, psi, layers_theta):
        self.loss_records = []

        # 在 WRC_NN 的 __init__ 中：
        self.theta_true_eval = theta_true_eval # 接收真实答案用于 callback 对比

        

        self.theta = theta
        self.psi = psi
        self.layers_theta = layers_theta

        self.weights_theta, self.biases_theta = initialize_MNN(layers_theta, xavier_init)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

        self.psi_tf = tf.placeholder(tf.float32, shape=[None, self.psi.shape[1]])
        self.theta_tf = tf.placeholder(tf.float32, shape=[None, self.theta.shape[1]])

        # WRC Prediction
        tf_log_h = tf.math.log(-self.psi_tf + 1e-6)
        self.theta_pred = net_theta(-tf_log_h, self.weights_theta, self.biases_theta)
        
        # Loss 1: Data Fitting Loss (WRC)
        self.theta_loss = tf.reduce_mean(tf.square(self.theta_tf - self.theta_pred))

        # --- Loss 2: Gradient Loss (Non-uniform sampling) ---
        log10_start = np.log10(0.001).astype(np.float32)
        log10_end = np.log10(0.1).astype(np.float32)
        log10_h_uniform = tf.linspace(log10_start, log10_end, 60)
        h_extra = tf.pow(10.0, log10_h_uniform)
        h_extra = tf.reshape(h_extra, [-1, 1])

        # Input to net_theta is log(h), which is -log(-psi)
        theta_extra = net_theta(-tf.math.log(h_extra), self.weights_theta, self.biases_theta)
        grad_theta_extra = tf.gradients(theta_extra, h_extra)[0]
        
        # 增加对 grad_theta_extra 为 None 的检查
        if grad_theta_extra is None:
             self.grad_loss = tf.constant(0.0) 
        else:
             self.grad_loss = tf.reduce_mean(tf.square(grad_theta_extra))


        # Total loss for WRC
        self.loss = self.theta_loss * 500 + self.grad_loss * 10 # Example weights

        # 【新增】验证节点
        self.psi_eval_tf = tf.placeholder(tf.float32, shape=[None, 1])
        tf_log_h_eval = tf.math.log(-self.psi_eval_tf + 1e-6)
        self.theta_eval_pred = net_theta(-tf_log_h_eval, self.weights_theta, self.biases_theta)

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', options = {'maxiter': 50000, 'maxfun': 50000, 'maxcor': 50, 'maxls': 50, 'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        self.epohNum = 0
        self.sess.run(tf.global_variables_initializer())

    # 保持原方法名 train，用于 Adam 优化
    def train(self, N_iter):
        tf_dict = {self.psi_tf: self.psi, self.theta_tf: self.theta, self.psi_eval_tf: psi_eval_neg}
        start_time = time.time()
        for it in range(N_iter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # --- 修复: 在 Adam 训练过程中记录损失 ---
            if it % 100 == 0:
                theta_loss_val, grad_loss_val, total_loss_val = self.sess.run([self.theta_loss, self.grad_loss, self.loss], tf_dict)
                elapsed = time.time() - start_time
                print('It: %d, T_Loss: %.3e, $\\theta$ Loss: %.3e, Grad Loss: %.3e, Time: %.2f' %(it, total_loss_val, theta_loss_val, grad_loss_val, elapsed))
                start_time = time.time()

                loss_val, pred_theta = self.sess.run([self.loss, self.theta_eval_pred], feed_dict=tf_dict)
                # 【记录 NEF】
                current_nef = calc_nef(theta_true_eval, pred_theta)
                current_rmse = calc_rmse(self.theta_true_eval, pred_theta)
                self.loss_records.append({
                    'Stage': 'Adam', 
                    'Epoch': it, 
                    'Theta_NEF': current_nef, 
                    'Theta_RMSE': current_rmse,
                    'Theta Loss': theta_loss_val, 
                    'Grad Loss': grad_loss_val, 
                    'Total Loss': total_loss_val
                })
    
    # 新增方法，用于启动 L-BFGS-B 优化
    # 【修改 1】：增加 psi_eval_negative 参数，用于接收验证集
    def train_lbfgsb(self):
        # 【修改 2】：把验证集占位符加入 feed_dict
        tf_dict = {
            self.psi_tf: self.psi, 
            self.theta_tf: self.theta,
            self.psi_eval_tf: psi_eval_neg  # <--- 喂入验证集数据
        }
        print('Starting L-BFGS-B Optimization...')
        self.epohNum = 0 # 重置或继续 L-BFGS-B 的 epoch 计数
        
        # 【修改 3】：在 fetches 列表中加入预测节点 self.theta_eval_pred
        self.optimizer.minimize(
            self.sess, 
            feed_dict=tf_dict, 
            fetches=[self.loss, self.theta_loss, self.grad_loss, self.theta_eval_pred], # <--- 让图顺便把预测值算出来
            loss_callback=self.callback
        )

    # L-BFGS-B 优化器在每次迭代时调用的回调函数
    # 【修改 4】：参数列表最后面加上 pred_theta，接收刚刚 fetches 传出来的结果
    def callback(self, total_loss, theta_loss, grad_loss, pred_theta):
        self.epohNum += 1
        
        # 【修改 5】：调用外部函数计算 NEF 
        # (注意：前提是你已经在 __init__ 里把标准的 theta 存为 self.theta_true_eval，如果是全局变量直接写 theta_true_eval 也可以)
        nef_theta = calc_nef(self.theta_true_eval, pred_theta)
        current_rmse = calc_rmse(self.theta_true_eval, pred_theta)
        
        # --- 修复: 确保记录了 L-BFGS-B 的损失，Stage 标记为 L-BFGS-B ---
        self.loss_records.append({
            'Stage': 'L-BFGS-B', 
            'Epoch': self.epohNum, 
            'Theta Loss': theta_loss, 
            'Grad Loss': grad_loss, 
            'Total Loss': total_loss,
            'Theta_NEF': nef_theta,      # <--- 【修改 6】：把误差存进字典
            'Theta_RMSE': current_rmse
        })
        
        if self.epohNum % 500 == 0:
            # 顺便在控制台打印的时候也加上 NEF 方便观察
            print('L-BFGS-B Epoch: %d, T_Loss: %.3e, $\\theta$ Loss: %.3e, Grad Loss: %.3e, Theta_NEF: %.4f' % 
                  (self.epohNum, total_loss, theta_loss, grad_loss, nef_theta))

    def WRC(self, psi_star):
        # 保持不变
        tf_log_h = tf.math.log(-psi_star + 1e-6)
        tf_dict = {self.psi_tf: psi_star}
        theta = self.sess.run(self.theta_pred, tf_dict)
        return theta
    
    def get_params(self):
        # 保持不变
        return self.sess.run(self.weights_theta), self.sess.run(self.biases_theta)
    
    def save_losses_to_csv(self, path):
        # 保持不变
        df = pd.DataFrame(self.loss_records)
        df.to_csv(path, index=False)


# ====================================================================
# STEP 2: HCF 训练类 (WRC 参数固定, HCF 依赖 Richards 方程损失)
# ====================================================================
class HCF_NN:
    def __init__(self, t, z, theta, psi, WRC_weights, WRC_biases, layers_K, layers_psi, lb, ub):
        
        self.loss_records = []
        
        # Training data for system identification

        self.t= t
        self.z = z
        self.theta = theta
        self.psi = psi


        # 【新增】接收用于验证的理论真实值
        self.theta_true_eval = theta_true_eval
        self.K_true_eval = K_true_eval

        # --- WRC Network (Fixed/Non-trainable) ---  self.WRC_weights, self.WRC_biases
        self.WRC_weights = [tf.constant(w, dtype=tf.float32) for w in WRC_weights]
        self.WRC_biases = [tf.constant(b, dtype=tf.float32) for b in WRC_biases]
        
        # --- K Network (Trainable) ---
        self.layers_K = layers_K
        self.weights_K, self.biases_K = initialize_MNN(layers_K, xavier_init)
        self.weights_psi, self.biases_psi = initialize_NN(layers_psi, xavier_init)


        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        

        self.z_tf = tf.placeholder(tf.float32, shape = [None, self.z.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape = [None, self.t.shape[1]])
        self.theta_tf = tf.placeholder(tf.float32, shape = [None, self.theta.shape[1]])
        self.psi_tf = tf.placeholder(tf.float32, shape = [None, self.theta.shape[1]])  # this is for lookup table


        # ---------------------------------Change 2
        self.x_values = np.arange(lb, ub + 0.5, 1.0)
        # self.x_values = np.arange(0, 100.5, 5.0)

        print(self.x_values)


        

        # Step 2: t from 0 to 1.5 (inclusive), step 0.1
        # self.t_values = np.arange(0.1, 100.001, 0.1)
        N = 500

        start, end = 0.1, 200.0

        # log/几何分布：在 log10 空间等间距
        self.t_values = np.geomspace(start, end, N)

        # Step 3: Create Cartesian product of x and t
        self.X2, self.T2 = np.meshgrid(self.x_values, self.t_values)  # shape (16, 18)
        self.result2 = np.column_stack([self.X2.ravel(), self.T2.ravel()])  # shape (288, 2)


        self.result2.flatten()

        # Extract x_vector as a column vector
        self.x_vector = self.result2[:, 0].reshape(-1, 1)
        

        # Extract t_vector similarly if needed
        self.t_vector = self.result2[:, 1].reshape(-1, 1)

        self.z_tf2 = tf.placeholder(tf.float32, shape = [None, self.x_vector.shape[1]])
        self.t_tf2 = tf.placeholder(tf.float32, shape = [None, self.t_vector.shape[1]])

        # 【新增】验证集占位符
        self.psi_eval_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # prediction from PINNs
        self.theta_pred, self.psi_pred, self.K_pred,self.f_pred, self.theta_t_pred, self.psi_z_pred, self.psi_zz_pred, self.K_z_pred = self.net(self.t_tf, self.z_tf)

        self.f_pred2 = self.f_model(self.t_tf2, self.z_tf2)        

        self.theta_loss = tf.reduce_mean(tf.square(self.theta_tf - self.theta_pred))
        self.psi_loss = tf.reduce_mean(tf.square(self.psi_tf - self.psi_pred))

        # loss for identification
        # --------------------------------------Change-4
        self.loss_data = self.theta_loss * 150 + 1e-2 * self.psi_loss


        self.loss_phys = tf.reduce_mean(tf.square(self.f_pred2))                     # PDE 残差损失



        self.loss = self.loss_data + self.loss_phys * 150


        # 【新增】验证集预测节点
        tf_log_h_eval = tf.math.log(-self.psi_eval_tf + 1e-6)
        self.theta_eval_pred = net_theta(-tf_log_h_eval, self.WRC_weights, self.WRC_biases)
        self.K_eval_pred = self.net_K(-tf_log_h_eval, self.weights_K, self.biases_K)

        # Optimizer for identification
        # L-BFGS-B method
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                        method = 'L-BFGS-B',
                                                        options = {'maxiter': 50000,
                                                                   'maxfun': 50000,
                                                                   'maxcor': 50,
                                                                   'maxls': 50,
                                                                   'ftol' : 1.0 * np.finfo(float).eps})

        # Adam method
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        self.epohNum = 0
        

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # tf.saver
        self.saver = tf.train.Saver()        
        

    def train(self, N_iter):
        tf_dict = {self.t_tf: self.t, self.z_tf: self.z, self.theta_tf: self.theta,
        self.t_tf2: self.t_vector, self.z_tf2: self.x_vector, self.psi_tf: self.psi,self.psi_eval_tf: psi_eval_neg}

        start_time = time.time()
        # Adam
        for it in range(N_iter):
            
            self.sess.run(self.train_op_Adam, tf_dict)
            theta_loss_val, psi_loss_val, phys_loss_val, total_loss_val, p_theta, p_K = self.sess.run([self.theta_loss, self.psi_loss, self.loss_phys, self.loss, 
                     self.theta_eval_pred, self.K_eval_pred], tf_dict)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %(it, loss_value, elapsed))
                start_time = time.time()
                # 计算 NEF 对比误差
                nef_t = calc_nef(self.theta_true_eval, p_theta)
                nef_k = calc_nef(self.K_true_eval, p_K)
                rmse_k = calc_rmse(self.K_true_eval, p_K)

                self.loss_records.append({
                    'Stage': 'Adam',
                    'Epoch': it,
                    'Theta Loss': theta_loss_val,
                    'Psi Loss': psi_loss_val,
                    'Physics Loss': phys_loss_val,
                    'Total Loss': total_loss_val,
                    'Theta_NEF': nef_t, 
                    'K_NEF': nef_k,
                    'K_RMSE': rmse_k
                })

        # L-BFGS-B
        '''
        
        '''
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.theta_loss, self.psi_loss, self.loss_phys, self.theta_eval_pred, self.K_eval_pred],
                                loss_callback = self.callback)

        loss_value = self.sess.run(self.loss, tf_dict)
        
        
        
        


    def callback(self, total_loss, theta_loss, psi_loss, phys_loss, p_theta, p_K):
        self.epohNum = self.epohNum + 1
        nef_t = calc_nef(self.theta_true_eval, p_theta)
        nef_k = calc_nef(self.K_true_eval, p_K)
        rmse_k = calc_rmse(self.K_true_eval, p_K)
        self.loss_records.append({
            'Stage': 'L-BFGS-B',
            'Epoch': self.epohNum,
            'Theta Loss': theta_loss,
            'Psi Loss': psi_loss,
            'Physics Loss': phys_loss,
            'Total Loss': total_loss,
            'Theta_NEF': nef_t, 'K_NEF': nef_k, 'K_RMSE': rmse_k,
        })

    def HCF(self, psi_star):
        # Inference graph setup to avoid unnecessary placeholders (t, z)
        psi_inf_tf = tf.placeholder(tf.float32, shape=[None, 1])
        tf_log_h_inf = tf.math.log(-psi_inf_tf + 1e-6)

        # Get the trained weights once
        k_weights_np, k_biases_np = self.sess.run([self.weights_K, self.biases_K])
        
        # Use a temporary session for inference
        with tf.Session() as temp_sess:
            k_weights_inf = [tf.constant(w) for w in k_weights_np]
            k_biases_inf = [tf.constant(b) for b in k_biases_np]
            K_inf_pred = self.net_K(-tf_log_h_inf, k_weights_inf, k_biases_inf)
            
            K = temp_sess.run(K_inf_pred, feed_dict={psi_inf_tf: psi_star})
        return K
    
    def save_losses_to_csv(self, path):
        df = pd.DataFrame(self.loss_records)
        df.to_csv(path, index=False)


    def net(self, t, z):  # PINNs
        X = tf.concat([t, z],1)
        psi = self.net_psi(X, self.weights_psi, self.biases_psi)

        log_h = tf.math.log(-psi + 1e-6)
        theta = net_theta(-log_h, self.WRC_weights, self.WRC_biases)
        K = self.net_K(-log_h, self.weights_K, self.biases_K)

        theta_t = tf.gradients(theta, t)[0]
        psi_z = tf.gradients(psi, z)[0]
        psi_zz = tf.gradients(psi_z, z)[0]
        K_z = tf.gradients(K, z)[0]

        # residual for Richards equation
        f = theta_t - K_z*psi_z- K*psi_zz - K_z

        return theta, psi, K, f, theta_t, psi_z, psi_zz, K_z
    
    def f_model(self, t, z):  # PINNs
        X = tf.concat([t, z], 1)
        psi = self.net_psi(X, self.weights_psi, self.biases_psi)

        log_h = tf.math.log(-psi + 1e-6)  # 避免 log(0) 错误
        theta = net_theta(-log_h, self.WRC_weights, self.WRC_biases)
        K = self.net_K(-log_h, self.weights_K, self.biases_K)

        theta_t = tf.gradients(theta, t)[0]
        psi_z = tf.gradients(psi, z)[0]
        psi_zz = tf.gradients(psi_z, z)[0]
        K_z = tf.gradients(K, z)[0]

        # Richards 方程残差
        f = theta_t - K_z * psi_z - K * psi_zz - K_z
        return f
    
    def net_psi(self, X, weights, biases):  # NN for psi in [-100, 0]
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        psi = -100.0 * tf.sigmoid(0.5 * Y)  # ensures psi in (-100, 0)
        return psi
    # --- K Network Function ---
    def net_K(self, X, weights, biases):  # NN for K
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        K = tf.exp(tf.add(tf.matmul(H, W), b))  # force K to be positive
        return K


# ====================================================================
# MAIN EXECUTION LOOP
# ====================================================================
def main_loop(hydrus, noise, num_layers_psi, num_neurons_psi, num_layers_theta, num_neurons_theta, num_layers_K, num_neurons_K, number_random, lb, ub, sensorN):
    
    tf.reset_default_graph()
    tf.set_random_seed(number_random)
    random.seed(number_random)
    np.random.seed(number_random)

    # --- 1. Data Loading ---
    # ASSUMPTION: 'hydrus_output.csv' contains 't', 'z', 'theta', 'head' (psi)
    try:
        # 假设文件存在于当前目录下
        data = pd.read_csv(f"./hydrus_output.csv")
    except FileNotFoundError:
        print("Error: hydrus_output.csv not found. Please ensure it exists.")
        return

    t = data['t'].values[:,None]
    z = data['z'].values[:,None]
    theta_star = data['theta'].values[:,None]
    psi_star = data['head'].values[:,None]
    
    Z_star = np.hstack((t, z))
    theta_star = theta_star.flatten()[:,None]
    psi_star = psi_star.flatten()[:,None]
    
    layers_psi = np.concatenate([[2], num_neurons_psi*np.ones(num_layers_psi), [1]]).astype(int).tolist()
    layers_theta = np.concatenate([[1], num_neurons_theta*np.ones(num_layers_theta), [1]]).astype(int).tolist()
    layers_K = np.concatenate([[1], num_neurons_K*np.ones(num_layers_K), [1]]).astype(int).tolist()
    
    # --- 2. Training Data Selection (Sensor Points) ---
    fixed_position = np.linspace(lb, ub, sensorN)
    fixed_position = np.round(fixed_position, 1)
    print(f"Sensor positions (z): {fixed_position}")

    N = 500

    start, end = 0.1, 200.0

    # log/几何分布：在 log10 空间等间距
    sequence = np.geomspace(start, end, N)

    # 如果你想内部就离散到 1 位小数，可以直接四舍五入
    confined_t = np.round(sequence, 3)

    tolerance = 0.0002
    fixed_list = []

    for z_val in fixed_position:
        for t_val in confined_t:
            matches = data.index[(np.abs(data['z'] - z_val) <= tolerance) & (np.abs(data['t'] - t_val) <= tolerance)].values
            fixed_list.extend(matches.tolist()) 
    fixed_list = np.array(fixed_list, dtype=int) 
    
    # Training Data
    
    # 假设张力计有 5% 的相对测量误差
    # noise_level_psi = 0.05 

    # 生成噪音 (基于 psi_star 的绝对值来缩放噪音幅度)
    noise_psi = noise * np.abs(psi_star) * np.random.randn(psi_star.shape[0], psi_star.shape[1])

    # 将噪音加到原数据上
    psi_train_noisy = psi_star[fixed_list, :] + noise_psi[fixed_list, :]

    # 物理边界防御：确保加噪后的 psi 绝对不会越界变成正数 (最高限制在 -0.001)
    psi_train = np.minimum(psi_train_noisy, -0.001)


    # =========================================================
    # 训练数据加噪 (Theta)
    # =========================================================
    # 1. 设置 theta 的相对噪音水平 (例如 5% 的相对误差)
    # noise_level_theta = 0.05 
    
    # 2. 生成与 theta_star 成比例的高斯噪音
    noise_theta_matrix = noise * np.abs(theta_star) * np.random.randn(theta_star.shape[0], theta_star.shape[1])
    
    # 3. 将噪音叠加到提取出的传感器点位上
    theta_train_noisy = theta_star[fixed_list, :] + noise_theta_matrix[fixed_list, :]
    
    # 4. 【物理边界防御】强制裁剪：体积含水率必须严格在 (0, 1) 之间
    # 下限防负数截断设为 0.001，上限防溢出截断设为 0.5 (与你网络设计的 0.5 * sigmoid 对应)
    theta_train = np.clip(theta_train_noisy, 0.001, 0.5)


    Z_train = Z_star[fixed_list,:] # (t, z) coordinates
    # training data
    t_train = Z_train[:, 0:1]
    z_train = Z_train[:, 1:2]

    # =========================================================
    # 【新增】获取传感器点位上的原始数据，并计算相对误差
    # =========================================================
    theta_orig = theta_star[fixed_list, :]
    psi_orig = psi_star[fixed_list, :]

    # 计算相对误差 (分母加上 1e-8 防止除以 0 导致报错)
    theta_rel_err = (theta_train - theta_orig) / (np.abs(theta_orig) + 1e-8)
    psi_rel_err = (psi_train - psi_orig) / (np.abs(psi_orig) + 1e-8)


    # --- 3. Setup Folders ---
    folder_path = f"./Interval60/Interval{lb}-{ub}-{sensorN}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 【修改】在 DataFrame 中加入原始数据和相对误差列
    train_data = pd.DataFrame({
        't': Z_train[:, 0].flatten(),
        'z': Z_train[:, 1].flatten(),
        'theta_orig': theta_orig.flatten(),        # 原始无噪 theta
        'theta_train': theta_train.flatten(),      # 加噪后用于训练的 theta
        'theta_rel_err': theta_rel_err.flatten(),  # theta 的相对误差
        'psi_orig': psi_orig.flatten(),            # 原始无噪 psi
        'psi_train': psi_train.flatten(),          # 加噪后用于训练的 psi
        'psi_rel_err': psi_rel_err.flatten()       # psi 的相对误差
    })

    
    train_data.to_csv(f"{folder_path}/train_data_pinn.csv")

    
    # ====================================================================
    # STEP 1: Train WRC
    # ====================================================================
    print("\n" + "="*20 + " STEP 1: Training WRC (Data Loss) " + "="*20)
    wrc_model = WRC_NN(theta_train, psi_train, layers_theta)
    wrc_model.train(5000)

    print("--- Starting L-BFGS-B Training ---")
    wrc_model.train_lbfgsb() 
    
    wrc_weights, wrc_biases = wrc_model.get_params()
    wrc_model.save_losses_to_csv(f"{folder_path}/loss_log_wrc.csv")
    
    # WRC Lookup for final output
    h_look = np.arange(0.01, 100.01, 0.01) 
    psi_look = -h_look.reshape(-1, 1)      
    theta_look = wrc_model.WRC(psi_look)
    # wrc_lookup = pd.DataFrame({'psi': psi_look.flatten(), 'theta_wrc': theta_look.flatten()})
    wrc_lookup = pd.DataFrame({'psi': psi_look.flatten(), 'theta': theta_look.flatten()})
    wrc_lookup.to_csv(f"{folder_path}/lookupWRC.csv", index = False)



    # ====================================================================
    # STEP 2: Train HCF (WRC Fixed, HCF uses PINN Loss)
    # ====================================================================
    # print("\n" + "="*20 + " STEP 2: Training HCF (PINN Loss) " + "="*15)
    
    # Re-initialize TF graph and placeholders for HCF/PINN (crucial)
    tf.reset_default_graph()
    tf.set_random_seed(number_random)
    random.seed(number_random)
    np.random.seed(number_random)
    
    
    
    hcf_model = HCF_NN(t_train, z_train, theta_train, psi_train, wrc_weights, wrc_biases, layers_K, layers_psi, lb, ub)
    
    # Train HCF model
    hcf_model.train(30000) 
    
    hcf_model.save_losses_to_csv(f"{folder_path}/loss_log_hcf_pinn.csv")

    # HCF Lookup (Inference)
    K_look = hcf_model.HCF(psi_look)
    # hcf_lookup = pd.DataFrame({'psi': psi_look.flatten(), 'K_hcf': K_look.flatten()})
    hcf_lookup = pd.DataFrame({'psi': psi_look.flatten(), 'K': K_look.flatten()})

    # --- Final Output ---
    lookup = pd.merge(wrc_lookup, hcf_lookup, on='psi')
    lookup['h'] = lookup['psi'].abs()
    # lookup.to_csv(f"{folder_path}/lookup_wrc_hcf_pinn.csv", index = False)
    lookup.to_csv(f"{folder_path}/lookup.csv", index = False)
    # print(f"\nFinal WRC and HCF results (PINN estimated) saved to {folder_path}/lookup_wrc_hcf_pinn.csv")


# --- Study Parameters ---
hydrus = 'sandy'
noise = VG_PARAMS['noise_level']
num_layers_theta = 2
num_neurons_theta = 40
num_layers_K = 2
num_neurons_K = 40

num_layers_psi = 4
num_neurons_psi = 80

number_random = 111


# ======================
# Data (Example)
# intervals = [(67.5, 97.5)]
# counts_dict = {(67.5, 97.5): [16]}


# ======================
# Data
intervals = [(10 + 10 * i, 70 + 10 * i) for i in range(3)]


counts_dict = {
    (10 + 10 * i, 70 + 10 * i): [5, 7, 9, 11, 13, 15, 17, 19]
    for i in range(3)
}


# ======================
# Execution
for i, (a, b) in enumerate(intervals):
    counts = counts_dict[(a, b)]
    for j, n in enumerate(counts):
        output = f"Interval{a}-{b}-{n}"
        print("\n" + "#"*70)
        print(f"STARTING TWO-STEP PINN TRAINING FOR: {output}")
        print("#"*70)
        
        main_loop(hydrus, noise, num_layers_psi, num_neurons_psi, num_layers_theta, num_neurons_theta, num_layers_K, num_neurons_K, number_random, a, b, n)