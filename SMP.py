# SHARED MESSAGE POOL (COLLECTIVE BRAIN)
class SharedMessagePool:
    def __init__(self):
        self.pool = []

    def add_message(self, message: dict):
        self.pool.append(message)

    def get_messages(self) -> list:
        return self.pool
