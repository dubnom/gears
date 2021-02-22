from typing import List, Tuple
from unittest import TestCase
from x7.lib.annotations import tests
import tool
from x7.geom.testing import TestCaseGeomExtended
from tool import Tool


@tests(tool.Tool)
class TestTool(TestCaseGeomExtended):
    SAVE_MATCH = False

    @staticmethod
    def tools_for_tests() -> List[Tuple[str, Tool]]:
        """Return a list of tag, tool"""
        t1 = Tool(angle=40.0, depth=5.0, feed=200.0, flood=True, flutes=4, mill='conventional',
                  mist=False, number=40, radius=9.5, rpm=4000.0, shaft_extension=6.35, tip_height=0.0, tip_radius=0.0)
        t2 = Tool(angle=40.0, depth=5.0, feed=200.0, flood=True, flutes=4, mill='conventional',
                  mist=False, number=40, radius=9.5, rpm=4000.0, shaft_extension=6.35, tip_height=3.0, tip_radius=0.5)
        t3 = Tool(angle=0.0, depth=5.0, feed=200.0, flood=True, flutes=4, mill='conventional',
                  mist=False, number=40, radius=9.5, rpm=4000.0, shaft_extension=6.35, tip_height=2.0, tip_radius=0.5)
        return [('t1', t1), ('t2', t2), ('t3', t3)]

    @tests(tool.Tool.__init__)
    def test___init__(self):
        t = Tool(angle=1.0, depth=2.0, radius=3.0, tip_height=4.0, shaft_extension=5.0, number=6, rpm=7, feed=8, flutes=9, mist=True, flood=True, mill='both')
        self.assertAlmostEqual(t.angle_degrees, 1.0)
        with self.assertRaises(ValueError):
            Tool(mill='what?')
        with self.assertRaises(ValueError):
            Tool(angle=-1)
        with self.assertRaises(ValueError):
            Tool(depth=-1)
        with self.assertRaises(ValueError):
            Tool(radius=1, depth=2)
        with self.assertRaises(ValueError):
            Tool(tip_height=-1)
        with self.assertRaises(ValueError):
            Tool(tip_height=1, tip_radius=0.6)
        # __init__(self, angle=40.0, depth=3.0, radius=10.0, tip_height=0.0, shaft_extension=6.35, number=1, rpm=2000, feed=200, flutes=4, mist=False, flood=False, mill='both')

    @tests(tool.Tool.__str__)
    def test_str(self):
        t = Tool(angle=1.0, depth=2.0, radius=3.0, tip_height=4.0, shaft_extension=5.0, number=6, rpm=7, feed=8, flutes=9, mist=True, flood=True, mill='both')
        self.assertEqual('(Angle: 1.0, Depth: 2.0, Radius: 3.0, TipHeight: 4.0, Extension: 5.0, Flutes: 9)', str(t))

    @tests(tool.Tool.__str__)
    def test_repr(self):
        t = Tool(angle=1.0, depth=2.0, radius=3.0, tip_height=4.0, shaft_extension=5.0, number=6, rpm=7, feed=8, flutes=9, mist=True, flood=True, mill='both')
        expected = "Tool(angle=1.0, depth=2.0, radius=3.0, tip_height=4.0, tip_radius=0.0, shaft_extension=5.0, number=6, rpm=7, feed=8, flutes=9, mist=True, flood=True, mill='both')"
        self.assertEqual(expected, repr(t))

    @tests(tool.Tool.__eq__)
    def test_eq(self):
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
        tools = self.tools_for_tests()
        t1 = tools[0][1]
        t2 = tools[1][1]
        self.assertNotEqual(t1.cutter_poly(), t2.cutter_poly())
        self.assertNotEqual(t1.cutter_poly(10), t1.cutter_poly(40))

        for tt, t in tools:
            for val in [5, 50]:
                tag = '%s-%s' % (tt, val)
                with self.subTest():
                    self.assertMatch(t.cutter_poly(val), tag)

    @tests(tool.Tool.plot)
    def test_plot(self):
        with self.assertMatchPlot(shrink=8):
            t = Tool()
            t.plot(do_show=False)

    @tests(tool.Tool.shaft_poly)
    def test_shaft_poly(self):
        t = Tool()
        for val in [10, 40]:
            with self.subTest(val):
                self.assertMatch(t.shaft_poly(val), val)

    @tests(tool.Tool.to_json)
    @tests(tool.Tool.from_dict)
    def test_to_json(self):
        for tag, t in self.tools_for_tests():
            with self.subTest(tag):
                self.assertMatch(t.to_json(), tag)
                js = t.to_json()
                tt = Tool.from_json(js)
                self.assertEqual(t, tt)


@tests(tool)
class TestModTool(TestCase):
    """Tests for stand-alone functions in tool module"""

    @tests(tool.test)
    def test_test(self):
        # Interactive function, nothing to test
        pass
