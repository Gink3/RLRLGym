import random
import unittest

from rlrlgym import EnvConfig, MultiAgentRLRLGym


class TestInvariants(unittest.TestCase):
    def test_random_rollout_invariants(self):
        env = MultiAgentRLRLGym(EnvConfig(width=14, height=10, n_agents=3, max_steps=60))
        env.reset(seed=5)

        rng = random.Random(9)
        for _ in range(40):
            actions = {aid: rng.randint(0, 10) for aid in env.possible_agents}
            env.step(actions)

            for aid in env.possible_agents:
                a = env.state.agents[aid]
                r, c = a.position
                self.assertTrue(0 <= r < env.config.height)
                self.assertTrue(0 <= c < env.config.width)
                self.assertGreaterEqual(a.hp, 0)
                if a.alive:
                    tile_id = env.state.grid[r][c]
                    self.assertTrue(env.tiles[tile_id].walkable)


if __name__ == "__main__":
    unittest.main()
