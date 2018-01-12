import numpy as np
import scipy.stats as stats
from utils_coco import show_img
import math

def gauss(x):
    return 1. / math.sqrt(2. * math.pi) * math.e ** (-x ** 2 / 2)


class Attention():
    def __init__(self, basic_strength = 5e3, consider_range = 50):
        self.gaussion = stats.norm(0, 1)
        self.basic_strength = basic_strength
        self.cr = consider_range

    def build(self, img, discounted = 0.5):
        self.img = img.astype(np.float32) * discounted
        self.shape = img.shape

    def add_spot(self, x, y, strength = 0.5, dev = 20):
        bound = [[x - self.cr, x + self.cr], [y - self.cr, y + self.cr]]
        self._limit_bound(bound)
        
        strength = self.basic_strength * strength

        for i in range(bound[0][0], bound[0][1] + 1):
            for j in range(bound[1][0], bound[1][1] + 1):
                self.img[i][j] += strength * gauss((i - x + 0.) / dev) * gauss((j - y + 0.) / dev)

        self._limit_img(self.img)

    def show(self):
        show_img(self.img.astype(np.uint8))

    def get_image(self):
        return self.img.astype(np.uint8)

    def _limit_bound(self, bound):
        for i in range(2):
            for j in range(2):
                bound[i][j] = min(bound[i][j], self.shape[i] - 1)
                bound[i][j] = max(bound[i][j], 0)
                bound[i][j] = int(bound[i][j])

    def _limit_img(self, img):
        img[img < 0] = 0
        img[img > 255] = 255
