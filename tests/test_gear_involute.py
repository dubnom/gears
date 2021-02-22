import math
from typing import List, Tuple
from unittest import TestCase

from x7.geom.geom import Point
from x7.lib.annotations import tests
from x7.testing.extended import TestCaseExtended

import gear_involute
from gear_involute import GearInvolute, Involute, InvoluteWithOffsets, InvolutePair

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

    @tests(gear_involute.GearInvolute.restore)
    @tests(gear_involute.GearInvolute.copy)
    @tests(gear_involute.GearInvolute.__eq__)
    def test_eq(self):
        gears = self.gears_for_tests()
        for tag, gear in gears:
            with self.subTest(gear=tag+'_copy'):
                self.assertEqual(gear, gear.copy())
        copies = [gear.copy() for tag, gear in (gears[1:]+gears[:1])]
        for (tag, gear), copy in zip(gears, copies):
            with self.subTest(gear=tag+'_restore'):
                self.assertNotEqual(gear, copy)
                copy.restore(gear)
                self.assertEqual(gear, copy)

    @tests(gear_involute.GearInvolute.__init__)
    def test_init(self):
        g = GearInvolute(teeth=7, center=Point(1, 2), rot=1.5, module=3.5, relief_factor=1.1, steps=3,
                         tip_arc=1.1, root_arc=1.2, curved_root=True, debug=False, pressure_angle=14.5)
        self.assertEqual(g.teeth, 7)
        self.assertEqual(g.tip_arc, 1.1)

    @tests(gear_involute.GearInvolute._finish_tooth_parts)
    @tests(gear_involute.GearInvolute.gen_gear_tooth_parts)
    @tests(gear_involute.GearInvolute.gen_gear_tooth)
    def test_gen_gear_tooth(self):
        gears = self.gears_for_tests()
        for tag, gear in gears:
            with self.subTest(gear=tag):
                coords = [p.xy() for p in gear.gen_gear_tooth()]
                self.assertMatch(coords, tag)

    @tests(gear_involute.GearInvolute._finish_tooth_parts)
    @tests(gear_involute.GearInvolute.gen_gear_tooth_parts)
    def test_gen_gear_tooth_parts(self):
        gears = self.gears_for_tests()
        for tag_root, gear in gears:
            tag = tag_root + '_plain'
            with self.subTest(gear=tag):
                coords = [(n, [p.xy() for p in path]) for n, path in gear.gen_gear_tooth_parts()]
                self.assertMatch(coords, tag)
            tag = tag_root + '_closed'
            with self.subTest(gear=tag):
                coords = [(n, [p.xy() for p in path]) for n, path in gear.gen_gear_tooth_parts(closed=True)]
                self.assertMatch(coords, tag)
            tag = tag_root + '_extra'
            with self.subTest(gear=tag):
                plain_coords = [(n, [p.xy() for p in path]) for n, path in gear.gen_gear_tooth_parts()]
                parts = gear.gen_gear_tooth_parts(include_extras=True)
                coords = [(n, [p.xy() for p in part]) for n, part in parts if not n.startswith('_')]
                self.assertEqual(plain_coords, coords)

                def part_fmt(part):
                    if isinstance(part, Point):
                        return repr(part.round(7))
                    else:
                        return '%s.%s' % (type(part).__module__, type(part).__qualname__)
                extras = [(n, part_fmt(p)) for n, p in parts if n.startswith('_')]
                self.assertMatch(extras, tag)

    @tests(gear_involute.GearInvolute.gen_rack_tooth)
    def test_gen_rack_tooth(self):
        gears = self.gears_for_tests()
        for tag, gear in gears:
            with self.subTest(gear=tag):
                coords = [p.xy() for p in gear.gen_rack_tooth(as_pt=True)]
                self.assertMatch(coords, tag)

    @tests(gear_involute.GearInvolute.instance)
    def test_instance(self):
        # instance(self)
        pass  # TODO-impl gear_involute.GearInvolute.instance test

    @tests(gear_involute.GearInvolute.min_teeth_without_undercut)
    def test_min_teeth_without_undercut(self):
        # min_teeth_without_undercut(self)
        pass  # TODO-impl gear_involute.GearInvolute.min_teeth_without_undercut test

    @tests(gear_involute.GearInvolute.plot_show)
    @tests(gear_involute.GearInvolute.plot)
    def test_plot(self):
        gears = self.gears_for_tests()
        last_gear = None
        for tag, gear in gears:
            gear.plot()
            last_gear = gear
        if include_interactive:
            last_gear.plot_show()

    @tests(gear_involute.GearInvolute.base_radius)
    def test_base_radius(self):
        self.assertEqual(10 * math.cos(math.radians(20)), GearInvolute(teeth=10, module=2).base_radius)
        self.assertEqual(10 * math.cos(math.radians(14.5)), GearInvolute(teeth=10, module=2, pressure_angle=14.5).base_radius)

    @tests(gear_involute.GearInvolute.pitch)
    def test_pitch(self):
        self.assertEqual(2 * math.pi, GearInvolute(teeth=10, module=2).pitch)

    @tests(gear_involute.GearInvolute.pitch_radius)
    def test_pitch_radius(self):
        self.assertEqual(3 * 10 / 2, GearInvolute(teeth=10, module=3).pitch_radius)

    @tests(gear_involute.GearInvolute.pitch_radius_effective)
    def test_pitch_radius_effective(self):
        self.assertEqual(3 * 10 / 2, GearInvolute(teeth=10, module=3).pitch_radius)
        self.assertEqual(3 * (10 / 2 + 0.5), GearInvolute(teeth=10, module=3, profile_shift=0.5).pitch_radius_effective)

    @tests(gear_involute.GearInvolute.pressure_angle_degrees)
    @tests(gear_involute.GearInvolute.pressure_angle_radians)
    def test_pressure_angle_degrees(self):
        g = GearInvolute()
        self.assertEqual(20, g.pressure_angle_degrees)
        self.assertEqual(math.radians(20), g.pressure_angle_radians)
        g.pressure_angle_degrees = 14.5
        self.assertEqual(math.radians(14.5), g.pressure_angle_radians)
        g.pressure_angle_radians = math.radians(30)
        self.assertAlmostEqual(30, g.pressure_angle_degrees)

    @tests(gear_involute.GearInvolute.rack)
    def test_rack(self):
        # rack(self, tall_tooth=False, angle=0.0) -> rack.Rack
        pass  # TODO-impl gear_involute.GearInvolute.rack test

    @tests(gear_involute.GearInvolute.root_radius)
    def test_root_radius(self):
        self.assertEqual(3 * (10 / 2 - 1.25), GearInvolute(teeth=10, module=3).root_radius)
        self.assertEqual(3 * (10 / 2 - 1.25 + 0.5), GearInvolute(teeth=10, module=3, profile_shift=0.5).root_radius)

    @tests(gear_involute.GearInvolute.tip_radius)
    def test_tip_radius(self):
        self.assertEqual(3 * (10 / 2 + 1), GearInvolute(teeth=10, module=3).tip_radius)
        self.assertEqual(3 * (10 / 2 + 1 + 0.5), GearInvolute(teeth=10, module=3, profile_shift=0.5).tip_radius)

    @tests(gear_involute.GearInvolute.gen_gear_tooth)
    def test_gen_gear_tooth(self):
        # gen_gear_tooth(self) -> List[ForwardRef('BasePoint')]
        pass  # TODO-impl gear_involute.GearInvolute.gen_gear_tooth test


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
    @tests(gear_involute.InvolutePair.wheel)
    @tests(gear_involute.InvolutePair.pinion)
    @tests(gear_involute.InvolutePair.__init__)
    def test_init(self):
        pair = InvolutePair(wheel_teeth=20, pinion_teeth=10, module=2)
        pinion = pair.pinion()
        wheel = pair.wheel()
        self.assertEqual(pinion.center, Point(30, 0))
        self.assertEqual(pinion.teeth, 10)
        self.assertEqual(wheel.center, Point(0, 0))
        self.assertEqual(wheel.teeth, 20)

    @tests(gear_involute.InvolutePair.plot)
    def test_plot(self):
        # plot(self, color='blue', rotation=0.0, plotter=None)
        pass  # TODO-impl gear_involute.InvolutePair.plot test


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
        inv = InvoluteWithOffsets(radius=10)
        self.assertEqual(0, inv.calc_angle(10))
        self.assertEqual(2.4, inv.calc_angle(26))
        inv = InvoluteWithOffsets(7, 0.5, 0.6, 0.7, 9, 7.5)
        self.assertEqual(0, inv.calc_angle(6.438167441127949))
        self.assertAlmostEqual(-3.5, inv.calc_angle(26))

    @tests(gear_involute.InvoluteWithOffsets.calc_point)
    def test_calc_point(self):
        inv = InvoluteWithOffsets(radius=10)
        self.assertEqual((10, 0), inv.calc_point(0))
        self.assertEqual((9.999999999999984, -62.83185307179586), inv.calc_point(math.tau))
        inv = InvoluteWithOffsets(7, 0.5, 0.6, 0.7, 9, 7.5)
        self.assertEqual((5.280930519075444, 3.6826312403901604), inv.calc_point(0))
        self.assertEqual((26.36716701938755, -34.91546577055612), inv.calc_point(math.tau))

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

    @tests(gear_involute.InvoluteWithOffsets.as_dict)
    def test_as_dict(self):
        # as_dict(self)
        pass  # TODO-impl gear_involute.InvoluteWithOffsets.as_dict test

    @tests(gear_involute.InvoluteWithOffsets.copy)
    def test_copy(self):
        # copy(self)
        pass  # TODO-impl gear_involute.InvoluteWithOffsets.copy test

    @tests(gear_involute.InvoluteWithOffsets.path_pt)
    def test_path_pt(self):
        # path_pt(self, steps=10, clip=False) -> List[ForwardRef('BasePoint')]
        pass  # TODO-impl gear_involute.InvoluteWithOffsets.path_pt test


@tests(gear_involute)
class TestModGearInvolute(TestCase):
    """Tests for stand-alone functions in gear_involute module"""

    @tests(gear_involute.main)
    def test_main(self):
        # main()
        pass  # TODO-impl gear_involute.main test
