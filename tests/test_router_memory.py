import unittest
from laya.router import Router

class MockAgent:
    def __init__(self, name):
        self.name = name

class RouterMemoryTests(unittest.TestCase):
    def test_evict_and_unload_memory_release(self):
        r = Router(max_loaded=1)
        r.attach("english", MockAgent("en"))
        r.attach("multilingual", MockAgent("multi"))
        self.assertEqual(len(r.loaded), 2)
        
        # Trigger unload of specific model
        r.unload("english")
        self.assertNotIn("english", r.loaded)
        self.assertIn("multilingual", r.loaded)
        
        # Trigger unload all
        r.unload()
        self.assertEqual(len(r.loaded), 0)

if __name__ == "__main__":
    unittest.main()
