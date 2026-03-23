"""Tab modules for simulator UI."""

from .tab_configuration import build_configuration_tab
from .tab_info import build_info_tab
from .tab_logs import build_logs_tab
from .tab_results import build_results_tab
from .tab_test import build_test_tab
from .tab_train import build_train_tab

__all__ = [
	"build_info_tab",
	"build_configuration_tab",
	"build_train_tab",
	"build_test_tab",
	"build_results_tab",
	"build_logs_tab",
]
