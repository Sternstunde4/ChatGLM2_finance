class AutoIncrementIDGenerator:
    def __init__(self, start_id = 1):
        self.start_id = start_id
        self.reset()

    def generate_id(self):
        current_id = self.current_id
        self.current_id += 1
        return current_id

    def reset(self):
        self.current_id = self.start_id

