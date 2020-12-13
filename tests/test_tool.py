# Originally auto-generated on 2020-12-13-00:47:21 -0500 EST
# By '--verbose --verbose tool'

from unittest import TestCase
from gg.devtools.testing.annotations import tests
import tool


@tests(tool.Tool)
class TestTool(TestCase):
    @tests(tool.Tool.__init__)
    def test___init__(self):
        # __init__(self, angle=40.0, depth=3.0, radius=10.0, tip_height=0.0, shaft_extension=6.35, number=1, rpm=2000, feed=200, flutes=4, mist=False, flood=False, mill='both')
        pass  # TODO-impl tool.Tool.__init__ test

    @tests(tool.Tool.__str__)
    def test___str__(self):
        # __str__(self)
        pass  # TODO-impl tool.Tool.__str__ test

    @tests(tool.Tool.add_config_args)
    def test_add_config_args(self):
        # add_config_args(p: configargparse.ArgumentParser)
        pass  # TODO-impl tool.Tool.add_config_args test

    @tests(tool.Tool.cutter_poly)
    def test_cutter_poly(self):
        # cutter_poly(self, shaft_length=40.0)
        pass  # TODO-impl tool.Tool.cutter_poly test

    @tests(tool.Tool.plot)
    def test_plot(self):
        # plot(self, title='', do_show=True)
        pass  # TODO-impl tool.Tool.plot test

    @tests(tool.Tool.shaft_poly)
    def test_shaft_poly(self):
        # shaft_poly(self, shaft_length=40.0)
        pass  # TODO-impl tool.Tool.shaft_poly test

    @tests(tool.Tool.to_json)
    def test_to_json(self):
        # to_json(self, indent=None)
        pass  # TODO-impl tool.Tool.to_json test


@tests(tool)
class TestModTool(TestCase):
    """Tests for stand-alone functions in tool module"""

    @tests(tool.test)
    def test_test(self):
        # test(args)
        pass  # TODO-impl tool.test test
