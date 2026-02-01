#!/usr/bin/env python
# coding: utf-8

# uninstall tensorflow 2.x on Google Colab
# %tensorflow_version 2.x
# !pip uninstall -y tensorflow
# !pip install tensorflow-gpu==1.14.0

import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
import random
import pickle


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
# STEP 1: WRC
# ====================================================================
class WRC_NN:
    def __init__(self, theta, psi, layers_theta):
        self.loss_records = []
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
        
        
        if grad_theta_extra is None:
             self.grad_loss = tf.constant(0.0) 
        else:
             self.grad_loss = tf.reduce_mean(tf.square(grad_theta_extra))


        # Total loss for WRC
        self.loss = self.theta_loss * 500 + self.grad_loss * 10 # Example weights

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', options = {'maxiter': 50000, 'maxfun': 50000, 'maxcor': 50, 'maxls': 50, 'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        self.epohNum = 0
        self.sess.run(tf.global_variables_initializer())

    
    def train(self, N_iter):
        tf_dict = {self.psi_tf: self.psi, self.theta_tf: self.theta}
        start_time = time.time()
        for it in range(N_iter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            
            if it % 100 == 0:
                theta_loss_val, grad_loss_val, total_loss_val = self.sess.run([self.theta_loss, self.grad_loss, self.loss], tf_dict)
                elapsed = time.time() - start_time
                print('It: %d, T_Loss: %.3e, $\\theta$ Loss: %.3e, Grad Loss: %.3e, Time: %.2f' %(it, total_loss_val, theta_loss_val, grad_loss_val, elapsed))
                start_time = time.time()
                
                
                self.loss_records.append({
                    'Stage': 'Adam', 
                    'Epoch': it, 
                    'Theta Loss': theta_loss_val, 
                    'Grad Loss': grad_loss_val, 
                    'Total Loss': total_loss_val
                })
    
    
    def train_lbfgsb(self):
        tf_dict = {self.psi_tf: self.psi, self.theta_tf: self.theta}
        print('Starting L-BFGS-B Optimization...')
        self.epohNum = 0
        self.optimizer.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss, self.theta_loss, self.grad_loss], loss_callback=self.callback)

    # L-BFGS-B
    def callback(self, total_loss, theta_loss, grad_loss):
        self.epohNum += 1
        self.loss_records.append({'Stage': 'L-BFGS-B', 'Epoch': self.epohNum, 'Theta Loss': theta_loss, 'Grad Loss': grad_loss, 'Total Loss': total_loss})
        if self.epohNum % 500 == 0:
            print('L-BFGS-B Epoch: %d, T_Loss: %.3e, $\\theta$ Loss: %.3e, Grad Loss: %.3e' % (self.epohNum, total_loss, theta_loss, grad_loss))

    def WRC(self, psi_star):
        
        tf_log_h = tf.math.log(-psi_star + 1e-6)
        tf_dict = {self.psi_tf: psi_star}
        theta = self.sess.run(self.theta_pred, tf_dict)
        return theta
    
    def get_params(self):
        
        return self.sess.run(self.weights_theta), self.sess.run(self.biases_theta)
    
    def save_losses_to_csv(self, path):
        
        df = pd.DataFrame(self.loss_records)
        df.to_csv(path, index=False)


# ====================================================================
# STEP 2: HCF
# ====================================================================
class HCF_NN:
    def __init__(self, t, z, theta, psi, WRC_weights, WRC_biases, layers_K, layers_psi, lb, ub):
        
        self.loss_records = []
        
        # Training data for system identification

        self.t= t
        self.z = z
        self.theta = theta
        self.psi = psi

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

        
        sequence = np.geomspace(start, end, N)

        
        self.t_values = np.round(sequence, 3)


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

        # prediction from PINNs
        self.theta_pred, self.psi_pred, self.K_pred,self.f_pred, self.theta_t_pred, self.psi_z_pred, self.psi_zz_pred, self.K_z_pred = self.net(self.t_tf, self.z_tf)

        self.f_pred2 = self.f_model(self.t_tf2, self.z_tf2)        

        self.theta_loss = tf.reduce_mean(tf.square(self.theta_tf - self.theta_pred))
        self.psi_loss = tf.reduce_mean(tf.square(self.psi_tf - self.psi_pred))

        # loss for identification
        # --------------------------------------Change-4
        self.loss_data = self.theta_loss * 150 + 1e-2 * self.psi_loss


        self.loss_phys = tf.reduce_mean(tf.square(self.f_pred2))



        self.loss = self.loss_data + self.loss_phys * 150

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
        self.t_tf2: self.t_vector, self.z_tf2: self.x_vector, self.psi_tf: self.psi}

        start_time = time.time()
        # Adam
        for it in range(N_iter):
            
            self.sess.run(self.train_op_Adam, tf_dict)
            theta_loss_val, psi_loss_val, phys_loss_val, total_loss_val = self.sess.run([self.theta_loss, self.psi_loss, self.loss_phys, self.loss], tf_dict)
            self.loss_records.append({
                'Stage': 'Adam',
                'Epoch': it,
                'Theta Loss': theta_loss_val,
                'Psi Loss': psi_loss_val,
                'Physics Loss': phys_loss_val,
                'Total Loss': total_loss_val
            })
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %(it, loss_value, elapsed))
                start_time = time.time()

        # L-BFGS-B
        '''
        
        '''
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.theta_loss, self.psi_loss, self.loss_phys],
                                loss_callback = self.callback)

        loss_value = self.sess.run(self.loss, tf_dict)
        
        
        
        


    def callback(self, total_loss, theta_loss, psi_loss, phys_loss):
        self.epohNum = self.epohNum + 1
        self.loss_records.append({
            'Stage': 'L-BFGS-B',
            'Epoch': self.epohNum,
            'Theta Loss': theta_loss,
            'Psi Loss': psi_loss,
            'Physics Loss': phys_loss,
            'Total Loss': total_loss
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

        log_h = tf.math.log(-psi + 1e-6)
        theta = net_theta(-log_h, self.WRC_weights, self.WRC_biases)
        K = self.net_K(-log_h, self.weights_K, self.biases_K)

        theta_t = tf.gradients(theta, t)[0]
        psi_z = tf.gradients(psi, z)[0]
        psi_zz = tf.gradients(psi_z, z)[0]
        K_z = tf.gradients(K, z)[0]

        # Richards
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
    def net_K(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        K = 4.421 * tf.sigmoid(tf.add(tf.matmul(H, W), b))  # range (0, 2)
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
    # confined_t = np.arange(0.1, 100.001, 0.1) 

    N = 500

    start, end = 0.1, 200.0

    
    sequence = np.geomspace(start, end, N)

    
    confined_t = np.round(sequence, 3)

    tolerance = 0.0002
    fixed_list = []

    for z_val in fixed_position:
        for t_val in confined_t:
            matches = data.index[(np.abs(data['z'] - z_val) <= tolerance) & (np.abs(data['t'] - t_val) <= tolerance)].values
            fixed_list.extend(matches.tolist()) 
    fixed_list = np.array(fixed_list, dtype=int) 
    
    # Training Data
    noise_theta = noise * np.random.randn(theta_star.shape[0], theta_star.shape[1])
    # theta_train = theta_star[fixed_list, :] + noise_theta[fixed_list, :] # theta with noise
    theta_train = theta_star[fixed_list, :]
    psi_train = psi_star[fixed_list, :] # psi (head) without noise
    Z_train = Z_star[fixed_list,:] # (t, z) coordinates
    # training data
    t_train = Z_train[:, 0:1]
    z_train = Z_train[:, 1:2]

    # --- 3. Setup Folders ---
    folder_path = f"./Interval80/Interval{lb}-{ub}-{sensorN}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    train_data = pd.DataFrame({'z': Z_train[:, 1].flatten(), 't': Z_train[:, 0].flatten(),
                               'theta_train': theta_train.flatten(), 'psi_train': psi_train.flatten()})
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
    hcf_model.train(1000) 
    
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
noise = 0.002 
num_layers_theta = 2
num_neurons_theta = 40
num_layers_K = 2
num_neurons_K = 40

num_layers_psi = 4
num_neurons_psi = 80

number_random = 1


# ======================
# Data (Example)
# intervals = [(67.5, 97.5)]
# counts_dict = {(67.5, 97.5): [16]}


# ======================
# Data
intervals = [(10 + 10 * i, 90 + 10 * i) for i in range(1)]


counts_dict = {
    (10 + 10 * i, 90 + 10 * i): [5, 7, 9, 11, 13, 15, 17, 19]
    for i in range(1)
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