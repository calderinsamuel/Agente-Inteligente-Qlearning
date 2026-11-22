import unittest
import numpy as np
from src.gridworld import GridWorld
from src.qlearning import train, evaluate

class TestAgent(unittest.TestCase):
    def test_pared_no_permite_pasar(self):
        env=GridWorld();env.reset();s,r,done,info=env.step(0)
        self.assertEqual(s,0);self.assertTrue(info['wall']);self.assertFalse(done)
    def test_llegar_a_meta(self):
        env=GridWorld();env.position=(7,6)
        _, reward, done, info=env.step(1)
        self.assertEqual(reward,10);self.assertTrue(done);self.assertTrue(info['success'])
    def test_resultado_reproducible(self):
        a=train(episodes=50,seed=10);b=train(episodes=50,seed=10)
        np.testing.assert_allclose(a.q_table,b.q_table)
    def test_aprendizaje(self):
        result=train(episodes=3000,seed=42)
        self.assertGreaterEqual(evaluate(result.q_table)['success_rate'],.9)

if __name__=='__main__':unittest.main()
