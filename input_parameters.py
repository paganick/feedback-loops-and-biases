class input_parameters():
    def __init__(self, feedback_loop_type, retraining = False, results_folder = 'results/'):
        self.n       = 1000
        self.theta_0 = [0.7, 0.15]
        self.theta_1 = [0.3, 0.15]
        self.n0_training = 500
        self.n1_training = 500
        self.p0      = 0.5
        self.results_folder = results_folder
        
        self.retraining = retraining
        self.prediction_type = 'sigmoid'
        self.decision_treshold = 0.5
        
        self.feature_noise_parameters =[[0, 0], [0, 0]]
        self.outcome_noise_parameters =[[0, 0.1], [0, 0.1]]
        self.ML_outcome_noise_parameters = [0, 0]
        self.outcome_bias = 0.0
        
        self.feedback_loop_type = feedback_loop_type
        if  (self.feedback_loop_type == "Sampling"):
            self.set_parameters_SFL()
        elif(self.feedback_loop_type == "Individual"):
            self.set_parameters_IFL()
        elif(self.feedback_loop_type == "Feature"):
            self.set_parameters_FFL()
        elif(self.feedback_loop_type == "ML-model"):
            self.set_parameters_MLFL()
        elif(self.feedback_loop_type == "Outcome"):
            self.set_parameters_OFL()
        
        
    
    def set_parameters_SFL(self):
        return
    
    def set_parameters_IFL(self):
        return
        
    def set_parameters_FFL(self):
        self.theta_0 = [0.5, 0.15]
        self.theta_1 = [0.5, 0.15]
        self.feature_noise_parameters =[[0, 0.1], [-0.2, 0.1]]
    
    def set_parameters_MLFL(self):
        self.ML_outcome_noise_parameters = [0, 1]
        
            
    def set_parameters_OFL(self):
        self.outcome_bias = 0.2
       