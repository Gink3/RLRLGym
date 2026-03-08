import unittest

from rlrlgym.content.animals import load_animals


class TestAnimalsJson(unittest.TestCase):
    def test_base_animals_json_loads(self):
        animals = load_animals("data/base/animals.json")
        for key in ("rabbit", "deer", "fox", "wolf", "mountain_lion", "cow", "sheep", "pig", "chicken"):
            self.assertIn(key, animals)


if __name__ == "__main__":
    unittest.main()
