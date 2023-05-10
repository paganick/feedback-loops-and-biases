import numpy as np
import matplotlib.pyplot as plt
from individual import individual
from functions import get_truncated_normal
import tikzplotlib

class population():
    def __init__(self, input_parameters):
        #self.n_0 = input_parameters.n_0
        #self.n_1 = input_parameters.n_1
        self.n   = input_parameters.n
        self.p0  = input_parameters.p0
        self.individuals = []
        self.theta_0 = input_parameters.theta_0
        self.theta_1 = input_parameters.theta_1
        self.results_folder = input_parameters.results_folder
        self.initialize_population(input_parameters)
    
    def add_individual(self, index, input_parameters):
        group = np.random.uniform(0, 1)
        if (group<=self.p0):
            A = 0
            theta = get_truncated_normal(mean = self.theta_0[0], sd = self.theta_0[1], low=0, upp=1).rvs()
            self.n_0 = self.n_0 + 1
        else:
            A = 1
            theta = get_truncated_normal(mean = self.theta_1[0], sd = self.theta_1[1], low=0, upp=1).rvs()
            self.n_1 = self.n_1 + 1
        if (index == -1):
            self.individuals.append(individual(theta, A, input_parameters.feature_noise_parameters))
        else:
            self.individuals[index]=individual(theta, A, input_parameters.feature_noise_parameters)
     
    def initialize_population(self, input_parameters):
        self.n_0 = 0
        self.n_1 = 0
        for i in range(self.n):
            self.add_individual(-1, input_parameters)
          
    def get_thetas(self, A):
        theta_vector = []
        for i in range(self.n):
            if self.individuals[i].A == A:
                theta_vector.append(self.individuals[i].theta)
        return theta_vector  
    
    def get_x(self, A):
        x_vector = []
        for i in range(self.n):
            if self.individuals[i].A == A:
                x_vector.append(self.individuals[i].x)
        return x_vector  
    
    def update_sample(self,index, d, input_parameters):
        if (d==0):
            if self.individuals[index].A == 0:
                self.n_0 = self.n_0 - 1
            else:
                self.n_1 = self.n_1 - 1
            if (self.n_1>1):
                self.p_0 = self.n_0/self.n_1
            else:
                self.p_0 = 1
            self.add_individual(index, input_parameters)
    
    def get_individual(self, index):
        return self.individuals[index]
    
    def update_theta(self,index,d):
        if (d==1):
            news = 1
        else:
            news = 0
        self.individuals[index].theta = max(0, min(1, 0.1*(news)+0.9*self.individuals[index].theta))
    
    def update_feature(self,index,d):
        self.individuals[index].x = max(0, min(1, 0.05*self.individuals[index].theta+0.95*self.individuals[index].x))

    def plot_hist_theta(self, t):
        theta0_vector = self.get_thetas(A=0)
        theta1_vector = self.get_thetas(A=1)
        plt.figure(figsize=(8,6))
        plt.hist(theta0_vector, bins=20, alpha=0.5, label="Group 1", color='#D7191C', density = True)
        plt.hist(theta1_vector, bins=20, alpha=0.5, label="Group 2", color='#2C7BB6', density = True)
        plt.legend()
        plt.xlim([0,1])
        tikzplotlib.save(self.results_folder+'histogram_theta_'+str(t)+'.tikz') 
        return theta0_vector, theta1_vector
        
    def plot_hist_group(self, t):
        theta0_vector = self.get_thetas(A=0)
        theta1_vector = self.get_thetas(A=1)
        plt.figure(figsize=(8,6))
        plt.bar(['Group 1', 'Group 2'], [len(theta0_vector), len(theta1_vector)], color=['#D7191C', '#2C7BB6'])
        plt.legend()
        tikzplotlib.save(self.results_folder+'histogram_group_'+str(t)+'.tikz')
        return len(theta0_vector), len(theta1_vector)
        
        
    def plot_hist_feature(self, t):
        x0_vector = self.get_x(A=0)
        x1_vector = self.get_x(A=1)
        plt.figure(figsize=(8,6))
        plt.hist(x0_vector, bins=20, alpha=0.5, label="Group 1", color='#D7191C')
        plt.hist(x1_vector, bins=20, alpha=0.5, label="Group 2", color='#2C7BB6')
        plt.legend()
        tikzplotlib.save(self.results_folder+'histogram_feature_'+str(t)+'.tikz')

    
    def plot_hist_feature_error(self, t):
        x0_vector = self.get_x(A=0)
        x1_vector = self.get_x(A=1)
        theta0_vector = self.get_thetas(A=0)
        theta1_vector = self.get_thetas(A=1)
        error0_vector = np.array(x0_vector) - np.array(theta0_vector)
        error1_vector = np.array(x1_vector) - np.array(theta1_vector)
        plt.figure(figsize=(8,6))
        plt.hist(error0_vector, bins=20, alpha=0.5, label="Group 1", color='#D7191C')
        plt.hist(error1_vector, bins=20, alpha=0.5, label="Group 2", color='#2C7BB6')
        plt.title('Feature measurement error')
        plt.legend()
        tikzplotlib.save(self.results_folder+'Feature_measurement_error_'+str(t)+'.tikz')
        return error0_vector, error1_vector
   
    
    
    