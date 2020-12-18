# Originally auto-generated on 2020-12-13-00:47:21 -0500 EST
# By '--verbose --verbose tool'

import json
from unittest import TestCase
from gg.devtools.testing.annotations import tests
import tool
from gg.devtools.testing.extended import TestCaseExtended
from tool import Tool


@tests(tool.Tool)
class TestTool(TestCaseExtended):
    SAVE_MATCH = False


    @tests(tool.Tool.__init__)
    def test___init__(self):
        t = Tool(angle=1.0, depth=2.0, radius=3.0, tip_height=4.0, shaft_extension=5.0, number=6, rpm=7, feed=8, flutes=9, mist=True, flood=True, mill='both')
        self.assertAlmostEqual(t.angle_degrees, 1.0)
        with self.assertRaises(ValueError):
            Tool(mill='what?')
        # __init__(self, angle=40.0, depth=3.0, radius=10.0, tip_height=0.0, shaft_extension=6.35, number=1, rpm=2000, feed=200, flutes=4, mist=False, flood=False, mill='both')

    @tests(tool.Tool.__str__)
    def test___str__(self):
        t = Tool(angle=1.0, depth=2.0, radius=3.0, tip_height=4.0, shaft_extension=5.0, number=6, rpm=7, feed=8, flutes=9, mist=True, flood=True, mill='both')
        self.assertEqual('(Angle: 1.0, Depth: 2.0, Radius: 3.0, TipHeight: 4.0, Extension: 5.0, Flutes: 9)', str(t))

    @tests(tool.Tool.__eq__)
    def test___eq__(self):
        t = Tool(angle=1.0, depth=2.0, radius=3.0, tip_height=4.0, shaft_extension=5.0, number=6, rpm=7, feed=8, flutes=9, mist=True, flood=True, mill='both')
        tt = Tool(angle=1.0, depth=2.0, radius=3.0, tip_height=4.0, shaft_extension=5.0, number=6, rpm=7, feed=8, flutes=9, mist=True, flood=True, mill='both')
        self.assertEqual(t, tt)
        tt.radius += 1
        self.assertNotEqual(t, tt)

    @tests(tool.Tool.add_config_args)
    @tests(tool.Tool.from_config_args)
    @tests(tool.Tool.from_config_file)
    def test_add_config_args(self):
        t = Tool.from_config_file('cutter40.cfg')
        self.assertEqual(40.0, t.angle_degrees)

    @tests(tool.Tool.cutter_poly)
    def test_cutter_poly(self):
        # cutter_poly(self, shaft_length=40.0)
        t = Tool()
        for val in [10, 40]:
            with self.subTest(val):
                self.assertMatch(t.cutter_poly(val), val)
        # TODO-test with various tool shapes

    @tests(tool.Tool.plot)
    def test_plot(self):
        # plot(self, title='', do_show=True)
        pass  # TODO-impl tool.Tool.plot test

    @tests(tool.Tool.shaft_poly)
    def test_shaft_poly(self):
        t = Tool()
        for val in [10, 40]:
            with self.subTest(val):
                self.assertMatch(t.shaft_poly(val), val)

    @tests(tool.Tool.to_json)
    @tests(tool.Tool.from_dict)
    def test_to_json(self):
        t = Tool()
        js = t.to_json()
        tt = Tool.from_dict(json.loads(js))
        self.assertEqual(t, tt)


@tests(tool)
class TestModTool(TestCase):
    """Tests for stand-alone functions in tool module"""

    @tests(tool.test)
    def test_test(self):
        # test(args)
        pass  # TODO-impl tool.test test
