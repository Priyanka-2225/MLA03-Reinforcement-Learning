class LearningEnv:
    def reset(self):
        self.progress = 0
        return np.array([self.progress])

    def step(self, action):
        self.progress += action
        reward = self.progress
        return np.array([self.progress]), reward
