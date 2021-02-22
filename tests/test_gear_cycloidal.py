from typing import List, Tuple
from unittest import TestCase
from x7.geom.geom import Point
from x7.lib.annotations import tests
from x7.geom.testing import TestCaseGeomExtended
import gear_cycloidal


@tests(gear_cycloidal.CycloidalPair)
class TestCycloidalPair(TestCaseGeomExtended):
    SAVE_MATCH = False
    maxDiff = 10000

    @staticmethod
    def pairs(tiny=False) -> List[Tuple[str, gear_cycloidal.CycloidalPair]]:
        """List of (tag, pair) for testing"""
        config = [
            ('a', 17, 9, 3, 0.0),
            ('b', 117, 19, 2, 0.0),
        ]
        if tiny:
            config = [('a', 11, 7, 1, 0.0)]
        cp = gear_cycloidal.CycloidalPair
        return [(tag, cp(wheel_teeth=wt, pinion_teeth=pt, module=m, generating_radius=gr))
                for tag, wt, pt, m, gr in config]

    @tests(gear_cycloidal.CycloidalPair.__init__)
    @tests(gear_cycloidal.CycloidalPair.__str__)
    def test_init(self):
        # __init__(self, wheel_teeth=30, pinion_teeth=8, module=1.0, generating_radius=0.0)
        pair = gear_cycloidal.CycloidalPair(wheel_teeth=17, pinion_teeth=9, module=3, generating_radius=0.0)
        self.assertEqual(17 * 3 / 2, pair.wheel_pitch_radius)
        self.assertEqual(9 * 3 / 2, pair.pinion_pitch_radius)
        self.assertEqual(9 * 3 / 2 / 2, pair.generating_radius)
        self.assertEqual('CycloidalPair: wheel=17 pinion=9 module=3', str(pair))

    @tests(gear_cycloidal.CycloidalPair.calc_addendum_factor)
    def test_calc_addendum_factor(self):
        # calc_addendum_factor(self) -> Tuple[float, float]
        pass  # TODO-impl gear_cycloidal.CycloidalPair.calc_addendum_factor test

    @tests(gear_cycloidal.CycloidalPair.calc_cycloid)
    @tests(gear_cycloidal.CycloidalPair.cycloid_path)
    def test_cycloid_path(self):
        # cycloid_path(self, theta_min=0.0, theta_max=1.5707963267948966, steps=5) -> List[ForwardRef('BasePoint')]
        for tag, pair in self.pairs():
            with self.subTest(pair=tag):
                path = pair.cycloid_path(theta_max=pair.wheel_tooth_theta, steps=10)
                self.assertMatch(path, case=tag)

    @tests(gear_cycloidal.CycloidalPair.gen_pinion_tooth)
    def test_gen_pinion_tooth(self):
        # gen_pinion_tooth(self) -> List[ForwardRef('BasePoint')]
        for tag, pair in self.pairs():
            with self.subTest(pair=tag):
                path = pair.gen_pinion_tooth()
                self.assertMatch(path, case=tag)

    @tests(gear_cycloidal.CycloidalPair.gen_wheel_tooth)
    def test_gen_wheel_tooth(self):
        # gen_wheel_tooth(self) -> List[ForwardRef('BasePoint')]
        for tag, pair in self.pairs():
            with self.subTest(pair=tag):
                path = pair.gen_wheel_tooth()
                self.assertMatch(path, case=tag)

    @tests(gear_cycloidal.CycloidalPair.pinion)
    def test_pinion(self):
        # pinion(self)
        for tag, pair in self.pairs(tiny=True):
            with self.subTest(pair=tag):
                pinion = pair.pinion()
                self.assertEqual(
                    Point(pair.wheel_pitch_radius + pair.pinion_pitch_radius, 0),
                    pinion.center
                )
                # use round() to shrink .td file size.  Detailed path already tested via tooth
                path = [(round(x, 2), round(y, 2)) for x, y in pinion.poly_at()]
                self.assertMatch(path, case=tag)

    @tests(gear_cycloidal.CycloidalPair.pinion_factors)
    def test_pinion_factors(self):
        # pinion_factors(self) -> Tuple[float, float]
        pass  # TODO-impl gear_cycloidal.CycloidalPair.pinion_factors test

    @tests(gear_cycloidal.CycloidalPair.plot)
    def test_plot(self):
        # plot(self, color='blue', rotation=0.0, pinion: Union[str, bool] = True)
        for tag, pair in self.pairs(tiny=True):
            with self.subTest(pair=tag):
                with self.assertMatchPlot(case=tag):
                    pair.plot()

    @tests(gear_cycloidal.CycloidalPair.wheel)
    def test_wheel(self):
        # wheel(self)
        for tag, pair in self.pairs(tiny=True):
            with self.subTest(pair=tag):
                wheel = pair.wheel()
                self.assertEqual(Point(0, 0), wheel.center)
                # use round() to shrink .td file size.  Detailed path already tested via tooth
                path = [(round(x, 2), round(y, 2)) for x, y in wheel.poly_at()]
                self.assertMatch(path, case=tag)


@tests(gear_cycloidal)
class TestModGearCycloidal(TestCase):
    """Tests for stand-alone functions in gear_cycloidal module"""
