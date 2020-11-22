from math import *

def rotate(a, x, y):
    """Return point x,y rotated by angle a (in radians)."""
    return x * cos(a) - y * sin(a), x * sin(a) + y * cos(a)

def foo(tool_angle, tool_tip_height):
    tool_radius = 20

    h_addendum = module
    h_dedendum = module * 1.25
    h_total = h_addendum + h_dedendum
    circular_pitch = module * pi
    pitch_diameter = module * teeth
    pitch_radius = pitch_diameter / 2.
    outside_diameter = pitch_diameter + 2 * h_addendum
    outside_radius = outside_diameter / 2.

    half_tooth = circular_pitch / 4.
    half_tool_tip = tool_tip_height / 2.
    tip_offset_y = h_dedendum
    tip_offset_z = half_tool_tip + tan(tool_angle / 2) * h_dedendum
    tool_angle_offset = tool_angle / 2. - pressure_angle

    print(pitch_radius,0)
    y = pitch_radius - h_dedendum
    z = -tan(pressure_angle) * h_dedendum
    print(y,z)

    y1, z1 = rotate(tool_angle_offset, y, z)
    print(y1,z1)
    y2, z2 = y1 + h_dedendum, z1 + tan(tool_angle/2.) * h_dedendum
    print(y2,z2)
    y3, z3 = rotate(-tool_angle_offset, y2, z2)

    return (y, z), (y3, z3), degrees(atan2(z3-z, y3-y)) 

def shorten(r, p):
    if type(r) == float:
        return round(r, p)
    else:
        return tuple(shorten(r1,p) for r1 in r)

ad = 1
steps = 5
module = 1 
teeth = 33
pressure_angle = radians(20)

print(45, shorten(foo(radians(45), 0), 4), '\n\n')
print(40, shorten(foo(radians(40), 0), 4), '\n\n')
print(0, shorten(foo(radians(0), 0), 4), '\n\n')

