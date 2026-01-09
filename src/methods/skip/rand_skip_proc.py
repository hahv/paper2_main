from src.methods.skip.base_skip_proc import BaseSkipProc

class RandSkipProc(BaseSkipProc):
    def should_skip(self, frame_idx, frame):
        import random
        skip = random.choice([True, False])
        return skip, {}  # Randomly skip
