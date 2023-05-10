from population import population
from sampling_block import sampling_block
from individual_block import individual_block
from feature_block import feature_block
from outcome_block import outcome_block
from prediction_block import prediction_block
from ML_model_block import ML_model_block
from decision_block import decision_block
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from functions import x2y


class model():
    def __init__(self, input_parameters):
        self.input_parameters = input_parameters
        self.results_folder = input_parameters.results_folder

        self.population       = population(input_parameters)
        self.sampling_block   = sampling_block()
        self.individual_block = individual_block()
        self.feature_block    = feature_block(input_parameters.feature_noise_parameters)
        self.outcome_block    = outcome_block(input_parameters.outcome_noise_parameters)
        self.ML_model_block   = ML_model_block(input_parameters)
        self.prediction_block = prediction_block(input_parameters.prediction_type, self.ML_model_block)
        self.decision_block   = decision_block(input_parameters.decision_treshold)


    def run(self, T=10):
        for t in range(T):
            self.run_time_step()
    
    def run_time_step(self):
        index = self.sampling_block.s(self.population)
        individual = self.individual_block.g(self.population, index)
        a = individual.A
        x = self.feature_block.r(individual)
        y = self.outcome_block.t(individual)
        hat_y = self.prediction_block.p(x)
        d = self.decision_block.d(hat_y)
        if(self.input_parameters.feedback_loop_type == 'Outcome'):
            y = self.outcome_feedback_loop(d,x,y)
            if (self.input_parameters.retraining):
                self.ML_model_update(x,y,a)
        elif(self.input_parameters.feedback_loop_type == "ML-model"):
            if (d == 1):
                self.ML_model_update(x,y,a)
        elif(self.input_parameters.feedback_loop_type == "Sampling"):
            self.population.update_sample(index, d, self.input_parameters)
            if (self.input_parameters.retraining):
                self.ML_model_update(x,y,a)
        elif(self.input_parameters.feedback_loop_type == "Individual"):
            self.population.update_theta(index,d)
            if (self.input_parameters.retraining):
                self.ML_model_update(x,y,a)
        elif(self.input_parameters.feedback_loop_type == "Feature"):
            self.population.update_feature(index,d)
            if (self.input_parameters.retraining):
                self.ML_model_update(x,y,a)
    
    def outcome_feedback_loop(self, d, x, y):
        if (d==1):
            if (y==1):
                return y
            outcome_bias = self.input_parameters.outcome_bias
        else:
            if (y==0):
                return y
            outcome_bias = - self.input_parameters.outcome_bias
        return x2y(x, outcome_bias, 0.05)
            
    
    def ML_model_update(self, x, y, A):
        self.ML_model_block.update(x,y, A)
        self.prediction_block.update(self.ML_model_block)
    
    def plot_ML_model(self, t):
        x0, y0, x1, y1, x_p, y_p = self.ML_model_block.plot(self.prediction_block, self.results_folder, t)
        return x0, y0, x1, y1, x_p, y_p

    def plot_ML_error_wrt_theta(self, t):
        err0x = []
        err0y = []
        err1x = []
        err1y = []
        for individual in self.population.individuals:
            y = []
            err = np.array(self.prediction_block.p(individual.theta))-np.array(individual.theta)
            if (individual.A == 0):
                err0x.append(individual.x)
                err0y.append(err[0])
            else:
                err1x.append(individual.x)
                err1y.append(err[0])
        plt.hist(err0y, bins=20, alpha=0.5, label="Group 1", color='#D7191C')
        plt.hist(err1y, bins=20, alpha=0.5, label="Group 2", color='#2C7BB6')
        plt.axvline(x = 0, color = 'black', linestyle = '--')
        tikzplotlib.save(self.results_folder+'ML_error_wrt_theta_'+str(t)+'.tikz')
        return err0x, err0y, err1x, err1y

    def plot_ML_error(self, t):
        err0x = []
        err0y = []
        err1x = []
        err1y = []
        for individual in self.population.individuals:
            y = []
#           simulate the outcome 50 times
            for i in range(50):
                y.append(self.outcome_block.t(individual))
            y_avg = np.mean(np.array(y))
            err = np.array(self.prediction_block.p(individual.x))-np.array(y_avg)
            if (individual.A == 0):
                err0x.append(individual.x)
                err0y.append(err[0])
            else:
                err1x.append(individual.x)
                err1y.append(err[0])
        plt.hist(err0y, bins=20, alpha=0.5, label="Group 1", color='#D7191C')
        plt.hist(err1y, bins=20, alpha=0.5, label="Group 2", color='#2C7BB6')
        plt.axvline(x = 0, color = 'black', linestyle = '--')
        tikzplotlib.save(self.results_folder+'ML_error_'+str(t)+'.tikz')
        return err0x, err0y, err1x, err1y
    