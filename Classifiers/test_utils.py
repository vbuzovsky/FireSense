import unittest
from utils import shuffle_dataset

class TestModule(unittest.TestCase):
      def test_shuffle_dataset(self):
         x = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
         y = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
         shuffle_dataset(x, y)
         print(x)
         self.assertEqual(x, y)

if __name__ == '__main__':
   unittest.main()
