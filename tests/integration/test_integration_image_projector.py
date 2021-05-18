import unittest
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class TestOld(unittest.TestCase):
    
    def setUp(self):
        self.file='fixtures/views2.txt'
        plt.ion()