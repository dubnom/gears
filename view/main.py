from typing import cast, Optional

import x7.view.digi
from gear_involute import GearInvolute
from x7.geom.colors import PenBrush
from x7.geom.geom import Vector
from x7.view.digi import DigitizeController
from x7.view.modes.adddrag import ModeAddDrag
from x7.view.shapes import DigitizeShape
from view.elem_gear import ViewGear, ElemGear


class ModeAddGear(ModeAddDrag):
    """Mode for adding via a single drag operation"""

    SHAPE_NAME = 'Gear'

    def __init__(self, controller: DigitizeController, parent: Optional[ViewGear] = None):
        super().__init__(controller)
        self.parent = parent
        # self.verbose = True

    def drag_begin(self, mp) -> DigitizeShape:
        """Start a drag operation at model space point mp"""
        # return ViewGear(self.controller.view, ElemEllipse('gearN', PenBrush('black'), mp, mp))
        if self.parent:
            parent_gear = self.parent.elem.gear
            # rv: Vector = mp - parent_gear.center
            gear = GearInvolute(teeth=3, center=mp,
                                module=parent_gear.module, relief_factor=parent_gear.relief_factor)
        else:
            gear = GearInvolute(teeth=3, center=mp)
        return ViewGear(self.controller.view, ElemGear('gearN', PenBrush('black'), gear))

    def drag_extend(self, start, mp, shape: 'ViewGear'):
        """Extend a drag operation to model space point mp"""
        # shape = cast(ViewGear, shape)
        elem = cast(ElemGear, shape.elem)
        gear = elem.gear
        if self.parent:
            # If parent, then gear center is set based on distance from parent gear
            parent_gear = self.parent.elem.gear
            parent_center = self.parent.elem.center()       # Computed center based on translations
            rv: Vector = mp - parent_center
            gear.teeth = max(3, round((rv.length()-parent_gear.pitch_radius) * 2 / gear.module))
            rvu = rv.unit()
            gear.center = rvu * (parent_gear.pitch_radius + gear.pitch_radius) + parent_center
            parent_rot = -rv.angle() / 360 * parent_gear.teeth - parent_gear.rot
            gear.rot = -rv.angle() / 360 * gear.teeth + parent_rot + (0.0 if gear.teeth & 0x1 else 0.5)
        else:
            rv: Vector = mp - gear.center
            gear.teeth = max(3, round(rv.length() * 2 / gear.module))
            gear.rot = -rv.angle() / 360 * gear.teeth


if __name__ == '__main__':
    x7.view.digi.test_digi()
