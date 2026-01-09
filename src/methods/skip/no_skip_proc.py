from src.methods.skip.base_skip_proc import BaseSkipProc

class NoSkipProc(BaseSkipProc):
    def should_skip(self, frame_idx, frame):
        return False, {}  # Never skip
