from typing import cast, Optional

import x7.view.digi
from x7.view.modes.common import ViewEvent

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
    HELP = 'Add Gear: click for center, drag to set pitch radius'

    def __init__(self, controller: DigitizeController, parent: Optional[ViewGear] = None):
        super().__init__(controller)
        self.parent = parent
        # self.verbose = True

    def drag_begin(self, mp) -> DigitizeShape:
        """Start a drag operation at model space point mp"""
        # return ViewGear(self.controller.view, ElemEllipse('gearN', PenBrush('black'), mp, mp))
        if self.parent:
            parent_gear = self.parent.elem.gear
            gear = parent_gear.copy()
            gear.center = mp
            gear.teeth = 3
            # gear.profile_shift = -parent_gear.profile_shift
            gear.profile_shift = 0
        else:
            gear = GearInvolute(teeth=3, center=mp, module=10)
        return ViewGear(self.controller.view, ElemGear('gearN', PenBrush('black'), gear))

    def drag_extend(self, start, event: ViewEvent, shape: 'ViewGear'):
        """Extend a drag operation to model space point mp"""
        # shape = cast(ViewGear, shape)
        elem = cast(ElemGear, shape.elem)
        gear = elem.gear
        if self.parent:
            # If parent, then gear center is set based on distance from parent gear
            parent_gear = self.parent.elem.gear
            parent_center = self.parent.elem.center()       # Computed center based on translations
            rv: Vector = event.mp - parent_center
            if not event.shift:
                gear.teeth = max(3, round((rv.length()-parent_gear.pitch_radius) * 2 / gear.module))
            rvu = rv.unit()
            center_dist = parent_gear.pitch_radius_effective + gear.pitch_radius_effective
            gear.center = rvu * center_dist + parent_center
            parent_rot = -rv.angle() / 360 * parent_gear.teeth - parent_gear.rot
            gear.rot = -rv.angle() / 360 * gear.teeth + parent_rot + (0.0 if gear.teeth & 0x1 else 0.5)
        else:
            rv: Vector = event.mp - gear.center
            if not event.shift:
                gear.teeth = max(3, round(rv.length() * 2 / gear.module))
            gear.rot = -rv.angle() / 360 * gear.teeth


if __name__ == '__main__':
    x7.view.digi.test_digi()
