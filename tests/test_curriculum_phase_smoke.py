import unittest

from train.rllib_trainer import RLlibTrainer


class TestCurriculumPhaseSmoke(unittest.TestCase):
    def test_phase_index_progresses_across_all_smoke_phases(self):
        phases = [
            {"name": "phase_1_smoke_survival_32", "until_episode": 1},
            {"name": "phase_2_smoke_crafting_64", "until_episode": 2},
            {"name": "phase_3_smoke_station_96", "until_episode": 3},
            {"name": "phase_4_smoke_animals_96", "until_episode": 4},
            {"name": "phase_5_smoke_team_96", "until_episode": 5},
            {"name": "phase_6_smoke_combat_96", "until_episode": 6},
            {"name": "phase_7_smoke_full_96", "until_episode": 0},
        ]

        for episode in range(1, 7):
            self.assertEqual(RLlibTrainer._phase_index_for_episode(None, episode, phases), episode)
        self.assertEqual(RLlibTrainer._phase_index_for_episode(None, 7, phases), 7)
        self.assertEqual(RLlibTrainer._phase_index_for_episode(None, 99, phases), 7)
        self.assertEqual(RLlibTrainer._phase_name_for_index(None, 7, phases), "phase_7_smoke_full_96")


if __name__ == "__main__":
    unittest.main()
