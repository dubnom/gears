import math
from typing import List, Tuple
from unittest import TestCase

from x7.geom.geom import Point
from x7.lib.annotations import tests
from x7.testing.extended import TestCaseExtended

import gear_involute
from gear_involute import GearInvolute, Involute, InvoluteWithOffsets

include_interactive = False


@tests(gear_involute.GearInvolute)
class TestGearInvolute(TestCaseExtended):
    SAVE_MATCH = False
    maxDiff = 10000

    @staticmethod
    def gears_for_tests() -> List[Tuple[str, GearInvolute]]:
        """Return a list of tag, gear"""
        g1 = GearInvolute(teeth=5)
        g2 = GearInvolute(teeth=7, center=Point(1, 2), rot=1.5, module=3.5, relief_factor=1.1, steps=3,
                          tip_arc=1.1, root_arc=1.2, curved_root=True, debug=False, pressure_angle=14.5)
        g3 = GearInvolute(teeth=5, center=Point(2, 1), rot=1.5, module=1.5, steps=3, profile_shift=0.3)
        return [('g1', g1), ('g2', g2), ('g3', g3)]

    @tests(gear_involute.GearInvolute.copy)
    @tests(gear_involute.GearInvolute.__eq__)
    def test_eq(self):
        gears = self.gears_for_tests()
        for tag, gear in gears:
            with self.subTest(gear=tag):
                self.assertEqual(gear, gear.copy())

    @tests(gear_involute.GearInvolute.__init__)
    def test_init(self):
        g = GearInvolute(teeth=7, center=Point(1, 2), rot=1.5, module=3.5, relief_factor=1.1, steps=3,
                         tip_arc=1.1, root_arc=1.2, curved_root=True, debug=False, pressure_angle=14.5)
        self.assertEqual(g.teeth, 7)
        self.assertEqual(g.tip_arc, 1.1)

    @tests(gear_involute.GearInvolute.cuts_for_mill)
    def test_cuts_for_mill(self):
        # cuts_for_mill(self, tool_angle, tool_tip_height=0.0) -> List[Tuple[float, float, float]]
        pass  # TODO-impl gear_involute.GearInvolute.cuts_for_mill test

    @tests(gear_involute.GearInvolute.gen_by_rack)
    def test_gen_by_rack(self):
        # gen_by_rack(self)
        pass  # TODO-impl gear_involute.GearInvolute.gen_by_rack test

    @tests(gear_involute.GearInvolute.gen_cuts_by_rack)
    def test_gen_cuts_by_rack(self):
        # gen_cuts_by_rack(self) -> Tuple[List[x7.geom.geom.Line], List[x7.geom.geom.Line]]
        pass  # TODO-impl gear_involute.GearInvolute.gen_cuts_by_rack test

    @tests(gear_involute.GearInvolute._finish_tooth_parts)
    @tests(gear_involute.GearInvolute.gen_gear_tooth_parts)
    @tests(gear_involute.GearInvolute.gen_gear_tooth)
    def test_gen_gear_tooth(self):
        gears = self.gears_for_tests()
        for tag, gear in gears:
            with self.subTest(gear=tag):
                coords = [p.xy() for p in gear.gen_gear_tooth()]
                self.assertMatch(coords, tag)

    @tests(gear_involute.GearInvolute.gen_rack_tooth)
    def test_gen_rack_tooth(self):
        # gen_rack_tooth(self)
        pass  # TODO-impl gear_involute.GearInvolute.gen_rack_tooth test

    @tests(gear_involute.GearInvolute.instance)
    def test_instance(self):
        # instance(self)
        pass  # TODO-impl gear_involute.GearInvolute.instance test

    @tests(gear_involute.GearInvolute.min_teeth_without_undercut)
    def test_min_teeth_without_undercut(self):
        # min_teeth_without_undercut(self)
        pass  # TODO-impl gear_involute.GearInvolute.min_teeth_without_undercut test

    if include_interactive:
        @tests(gear_involute.GearInvolute.plot_show)
        @tests(gear_involute.GearInvolute.plot)
        def test_plot(self):
            gears = self.gears_for_tests()
            for tag, gear in gears:
                gear.plot(mill_space=False, gear_space=False)
            gears[-1][1].plot_show()

    def test_plot_show(self):
        # plot_show(self, zoom_radius=0)
        pass  # TODO-impl gear_involute.GearInvolute.plot_show test

    @tests(gear_involute.GearInvolute.restore)
    def test_restore(self):
        # restore(self, other: 'GearInvolute')
        pass  # TODO-impl gear_involute.GearInvolute.restore test


@tests(gear_involute.Involute)
class TestInvolute(TestCaseExtended):
    SAVE_MATCH = False

    @tests(gear_involute.Involute.__init__)
    def test_init(self):
        inv = Involute(7, 9, 7.5)
        self.assertEqual(inv.radius, 7)
        self.assertEqual(inv.max_radius, 9)
        self.assertEqual(inv.min_radius, 7.5)

    @tests(gear_involute.Involute.calc_angle)
    def test_calc_angle(self):
        inv = Involute(7, 9)
        self.assertEqual(inv.calc_angle(7), 0)
        self.assertEqual(inv.calc_angle(8), 0.5532833351724881)

    @tests(gear_involute.Involute.calc_point)
    def test_calc_point(self):
        inv = Involute(7, 9)
        self.assertEqual(inv.calc_point(0), (7, 0))
        self.assertEqual(inv.calc_point(0.2), (7.138603108001778, 0.0185921065876902))

    @tests(gear_involute.Involute.calc_undercut_t_at_r)
    def test_calc_undercut_t_at_r(self):
        # calc_undercut_t_at_r(self, r)
        pass  # TODO-impl gear_involute.Involute.calc_undercut_t_at_r test

    @tests(gear_involute.Involute.path)
    def test_path(self):
        inv = Involute(7, 9)
        self.assertMatch(inv.path(10), '0')
        inv = Involute(7, 9, 7.5)
        self.assertAlmostEqual(math.hypot(*inv.path(3)[0]), 7.5)
        self.assertMatch(inv.path(3), '1')


@tests(gear_involute.InvolutePair)
class TestInvolutePair(TestCase):
    @tests(gear_involute.InvolutePair.__init__)
    def test_init(self):
        # __init__(self, wheel_teeth=30, pinion_teeth=6, module=1.0, relief_factor=1.25, steps=4, tip_arc=0.0, root_arc=0.0, curved_root=False, debug=False, pressure_angle=20.0)
        pass  # TODO-impl gear_involute.InvolutePair.__init__ test

    @tests(gear_involute.InvolutePair.pinion)
    def test_pinion(self):
        # pinion(self)
        pass  # TODO-impl gear_involute.InvolutePair.pinion test

    @tests(gear_involute.InvolutePair.plot)
    def test_plot(self):
        # plot(self, color='blue', rotation=0.0, plotter=None)
        pass  # TODO-impl gear_involute.InvolutePair.plot test

    @tests(gear_involute.InvolutePair.wheel)
    def test_wheel(self):
        # wheel(self)
        pass  # TODO-impl gear_involute.InvolutePair.wheel test


@tests(gear_involute.InvoluteWithOffsets)
class TestInvoluteWithOffsets(TestCaseExtended):
    SAVE_MATCH = False

    @tests(gear_involute.InvoluteWithOffsets.__init__)
    def test_init(self):
        # __init__(self, radius=0.0, offset_angle=0.0, offset_radius=0.0, offset_norm=0.0, radius_min=0.0, radius_max=0.0)
        inv = InvoluteWithOffsets(7, 0.5, 0.6, 0.7, 7.5, 9)
        self.assertEqual(inv.radius, 7)
        self.assertEqual(inv.offset_angle, 0.5)
        self.assertEqual(inv.offset_radius, 0.6)
        self.assertEqual(inv.offset_norm, 0.7)
        self.assertEqual(inv.radius_max, 9)
        self.assertEqual(inv.radius_min, 7.5)

    @tests(gear_involute.InvoluteWithOffsets.calc_angle)
    def test_calc_angle(self):
        # calc_angle(self, distance) -> float
        pass  # TODO-impl gear_involute.InvoluteWithOffsets.calc_angle test

    @tests(gear_involute.InvoluteWithOffsets.calc_point)
    def test_calc_point(self):
        # calc_point(self, angle, clip=True)
        pass  # TODO-impl gear_involute.InvoluteWithOffsets.calc_point test

    @tests(gear_involute.InvoluteWithOffsets.mid_angle)
    def test_mid_angle(self):
        # mid_angle(self)
        pass  # TODO-impl gear_involute.InvoluteWithOffsets.mid_angle test

    @tests(gear_involute.InvoluteWithOffsets.path)
    def test_path(self):
        # path(self, steps=10, offset=0.0, up=1, clip=False) -> List[Tuple[float, float]]
        inv = Involute(7, 9, 7.5)
        inv_wo = InvoluteWithOffsets(7, radius_max=9, radius_min=7.5)
        self.assertAlmostEqual(inv.path(), inv_wo.path())
        inv_wo = InvoluteWithOffsets(7, 0.5, 0.6, 0.7, 9, 7.5)
        self.assertMatch(inv_wo.path(), '0')

    @tests(gear_involute.InvoluteWithOffsets.x_calc_undercut_t_at_r)
    def test_x_calc_undercut_t_at_r(self):
        # x_calc_undercut_t_at_r(self, r)
        pass  # TODO-impl gear_involute.InvoluteWithOffsets.x_calc_undercut_t_at_r test


@tests(gear_involute)
class TestModGearInvolute(TestCase):
    """Tests for stand-alone functions in gear_involute module"""
