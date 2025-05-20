from framework.server.serverbase import Server


class ServerMADMTOP(Server):
    def __init__(self):
        super().__init__()
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def aggregate(self):
        pass

    def get_params(self):
        pass

    def set_clients(self, clients):
        self.clients = clients
