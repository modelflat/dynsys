import sys
from json import load

from .common import *

from .phase_portrait import PhasePortrait
from .parameter_surface import ParameterSurface
from .cobweb_diagram import CobwebDiagram
from .parameter_map import ParameterMap
from .bifurcation_tree import BifurcationTree
from .basins_of_attraction import BasinsOfAttraction


class SimpleApp(QtGui.QWidget):

    def __init__(self, title):
        self.app = QtGui.QApplication(sys.argv)
        super(SimpleApp, self).__init__(None)
        self.setWindowTitle(title)
        self.configFile = None
        self.config = None
        if len(sys.argv) > 1:
            self.configFile = sys.argv[1]
            print("Loading config from file: ", self.configFile)
            try:
                with open(self.configFile) as f:
                    self.config = load(f)
            except Exception as e:
                raise RuntimeWarning("Cannot load configuration from file %s: %s" % (self.configFile, str(e)))
            else:
                print("Loaded configuration:", self.config)

        self.ctx, self.queue = create_context_and_queue(self.config)

    def makePhasePortrait(self, bounds, system_function_source, param_count=2, width=512, height=512, type_config=float_config):
        return PhasePortrait(self.ctx, self.queue, width, height, bounds, system_function_source,
                             param_count=param_count, type_config=type_config)

    def makeParameterSurface(self, bounds, parameter_surface_color_function, width=512, height=512, type_config=float_config):
        return ParameterSurface(self.ctx, self.queue, width, height, bounds, parameter_surface_color_function, type_config=type_config)

    def makeCobwebDiagram(self, bounds, carrying_function_source, param_count=1, width=512, height=512, type_config=float_config):
        return CobwebDiagram(self.ctx, self.queue, width, height, bounds, carrying_function_source,
                             param_count=param_count, type_config=type_config)

    def makeParameterMap(self, bounds, map_function_source, var_count=1, width=512, height=512, type_config=float_config):
        return ParameterMap(self.ctx, self.queue, width, height, bounds, map_function_source, var_count=var_count,
                            type_config=type_config)

    def makeBifurcationTree(self, map_function_source, param_count=1, width=512, height=512, type_config=float_config):
        return BifurcationTree(self.ctx, self.queue, width, height, map_function_source,
                               param_count=param_count, type_config=type_config)

    def makeBasinsOfAttraction(self, bounds, system_function_source, width=512, height=512, param_count=2, type_config=float_config):
        return BasinsOfAttraction(self.ctx, self.queue, width, height, bounds, system_function_source,
                                  param_count=param_count, type_config=type_config)

    def run(self):
        self.show()
        sys.exit(self.app.exec_())