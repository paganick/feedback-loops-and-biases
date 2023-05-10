class decision_block():
    def __init__(self, decision_treshold = 0.5):
        self.decision_treshold = decision_treshold
        return

    def d(self, y_hat):
        if (y_hat>self.decision_treshold):
            return 1
        else:
            return 0