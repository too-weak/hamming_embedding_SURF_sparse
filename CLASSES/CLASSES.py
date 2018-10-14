class Pic():
    def __init__(self, des, x, y):
        self.des = des
        self.x = x
        self.y = y
        self.cluster=-1

    def set_hamming_code(self, code):
        self.hamming_code = code

    def set_cluster(self, cluster):
        self.cluster = cluster