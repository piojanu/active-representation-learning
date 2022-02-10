from utils.logx import InfoLogger


class DummyAgent(InfoLogger):
    def update(self, rollouts):
        del rollouts

    @staticmethod
    def log_info(logger, info):
        pass

    @staticmethod
    def compute_stats(logger):
        pass
