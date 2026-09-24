import unittest
from laya.router import Router

class MockAgent:
    def __init__(self, name):
        self.name = name

class ContextManagerTests(unittest.TestCase):
    def test_router_context_manager_cleanup(self):
        with Router(max_loaded=2) as r:
            r.attach("english", MockAgent("en"))
            r.attach("multilingual", MockAgent("multi"))
            self.assertEqual(len(r.loaded), 2)
        
        # After exiting context block, all models must be unloaded
        self.assertEqual(len(r.loaded), 0)

if __name__ == "__main__":
    unittest.main()
