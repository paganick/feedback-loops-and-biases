from functions import get_truncated_normal, x2y
from prediction_block import prediction_block
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


class ML_model_block():
    
    def __init__(self, input_parameters):
        n0_sample = input_parameters.n0_training
        n1_sample = input_parameters.n1_training
        self.outcome_bias = input_parameters.outcome_bias
        self.decision_treshold = input_parameters.decision_treshold
        self.feedback_loop_type = input_parameters.feedback_loop_type
        self.A = n0_sample*[0]+n1_sample*[1]
        self.X =[]
        for i in range (n0_sample):
            x = get_truncated_normal(mean = input_parameters.theta_0[0], sd = input_parameters.theta_0[1], low=0, upp=1).rvs()
            self.X.append([x])
        for i in range (n1_sample):
            x = get_truncated_normal(mean = input_parameters.theta_1[0], sd = input_parameters.theta_1[1], low=0, upp=1).rvs()
            self.X.append([x])
        self.Y = []
        for x in self.X:
            y = x2y(x, input_parameters.ML_outcome_noise_parameters[0], input_parameters.ML_outcome_noise_parameters[1])
            self.Y.append(y)
    
    def plot(self, prediction_block, results_folder, t ):
        x0 = []
        x1 = []
        y0 = []
        y1 = []
        x_p = np.linspace(0, 1, 100)
        y_p = []
        for x in x_p:
            y_p.append(prediction_block.p(x))            
        for i in range(len(self.X)):
            if (self.A[i] == 0):
                x0.append(self.X[i])
                y0.append(self.Y[i])
            else:
                x1.append(self.X[i])
                y1.append(self.Y[i])
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(x0, y0, color='#D7191C', label='Group 1', alpha=0.5)
        ax1.scatter(x1, y1, color='#2C7BB6', label='Group 2', alpha=0.5)
        ax1.plot(x_p, y_p, color='black', label='Prediction')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.legend()
        tikzplotlib.save(results_folder+'ML_prediction_'+str(t)+'.tikz') 
        plt.show()
        return x0, y0, x1, y1, x_p, y_p
    
    def compare_predictions(self, x0, y0, x1, y1, xp1, yp1, xp2, yp2, results_folder, t, prediction_block, decision_block, outcome_block, x_points, n_points, population):
        n_i = n_points
        x_p = []
        y_p = []
        y_p_mod = []
        for individual in population.individuals:
            x = individual.x
            x_p.append(x)
            y = []
            y_mod = []
            for i in range(n_i):
                y.append(x2y(x, outcome_block.outcome_noise_parameters[individual.A][0], outcome_block.outcome_noise_parameters[individual.A][1]))
                if (decision_block.d(prediction_block.p(x))==1):
                    y_mod.append(x2y(x, self.outcome_bias, outcome_block.outcome_noise_parameters[individual.A][1]))
                else:
                    y_mod.append(x2y(x, -self.outcome_bias, outcome_block.outcome_noise_parameters[individual.A][1]))
            y_avg = np.mean(np.array(y))
            y_mod_avg = np.mean(np.array(y_mod))
            y_p.append(y_avg)
            y_p_mod.append(y_mod_avg) 
        linspace = np.linspace(0,1, x_points)
        x_linspace = []
        y_linspace = []
        y_linspace_mod = []
        for i in range(len(linspace)+1):
            x_linspace.append([])
            y_linspace.append([])
            y_linspace_mod.append([])
        for i in range(len(x_p)):
            n_bucket = int(round(x_p[i]*x_points))
            x_linspace[n_bucket].append(x_p[i])
            y_linspace[n_bucket].append(y_p[i])
            y_linspace_mod[n_bucket].append(y_p_mod[i])
        x_plot = []
        y_plot = []
        y_plot_mod = []
        for i in  range(len(x_linspace)):
            x_plot.append(np.mean(x_linspace[i]))
            y_plot.append(np.mean(y_linspace[i]))
            y_plot_mod.append(np.mean(y_linspace_mod[i])) 
        x_plot = np.array(x_plot).reshape(-1)
        y_plot = np.array(y_plot).reshape(-1)
        y_plot_mod = np.array(y_plot_mod).reshape(-1)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(x0, y0, color='#D7191C', label='Group 1', alpha=0.5)
        ax1.scatter(x1, y1, color='#2C7BB6', label='Group 2', alpha=0.5)
        ax1.plot(xp1, yp1, color='black', label='Prediction Initial')
        ax1.plot(xp2, yp2, color='magenta', label='Prediction Final')
        if (self.feedback_loop_type == 'Outcome'):
            ax1.plot(x_plot, y_plot, color = 'black', linestyle = '--', label='True Initial Prediction')
            ax1.plot(x_plot, y_plot_mod, color = 'magenta', linestyle = '--', label='True Final Prediction')
        else:
            ax1.plot(x_plot, y_plot, color = 'black', linestyle = '--', label='True Prediction')
        #plt.axhline(y = 0.5, color = 'black', linestyle = '--')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.legend()
        tikzplotlib.save(results_folder+'compare_ML_prediction_'+str(t)+'.tikz') 
        plt.show()
        
    def update(self, x, y, A):
        self.X.append([x])
        self.Y.append(y)
        self.A.append(A)