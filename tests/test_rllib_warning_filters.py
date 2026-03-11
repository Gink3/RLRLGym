import logging
import unittest

from train.rllib_warning_filters import install_rllib_warning_filters


class TestRLlibWarningFilters(unittest.TestCase):
    def test_installs_message_filter_on_ray_loggers(self):
        install_rllib_warning_filters()
        logger = logging.getLogger("ray.rllib.utils.deprecation")
        self.assertTrue(logger.filters)
        self.assertFalse(
            all(
                flt.filter(
                    logging.makeLogRecord(
                        {
                            "msg": "`RLModule(config=[RLModuleConfig object])` has been deprecated. Use the new constructor."
                        }
                    )
                )
                for flt in logger.filters
            )
        )


if __name__ == "__main__":
    unittest.main()
