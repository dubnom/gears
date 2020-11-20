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
    tip_offset_y = cos(tool_angle / 2) * h_dedendum
    tip_offset_z = half_tool_tip + sin(tool_angle / 2) * h_dedendum
    tool_angle_offset = tool_angle / 2. - pressure_angle

    z_max = outside_radius
    z_incr = z_max / (steps * 2)

    n = 0
    for n in locals():
        if n != 'n':
            print(n, '=', eval(n))
    print()


    # Test a single cut of the 0th tooth
    half_tooth = 0
    z_step = 0 
    print('z_step =', z_step)
    z = z_step * z_incr
    print('z =', z)
    y = pitch_radius
    print('y =', y)
    angle = z / y
    print('angle =', degrees(angle), degrees(atan2(z, y) - angle), cos(angle)*pitch_radius, sin(angle)*pitch_radius)
    z += half_tooth
    print('z =', z)
    y_point, z_point = rotate(tool_angle_offset, y, z)
    print('y_point, z_point =', y_point, z_point)
    angle += tool_angle_offset
    print('angle =', degrees(angle))
    y_point -= tip_offset_y
    z_point -= tip_offset_z
    print('y_point, z_point =', y_point, z_point)

    a = ad * degrees(angle)
    y = -ad * (tool_radius + y_point)
    z = z_point

    # Determine where the line ended up
    # y1, z1 should be the tip of the line
    # y2, z2 should be the end of the line

    y1, z1 = rotate(radians(a), y - tool_radius, z + half_tool_tip)
    y2, z2 = y1 + cos(tool_angle/2.) * h_total, z1 + sin(tool_angle/2.) * h_total

    y1, z1 = rotate(-radians(a), y1, z1)
    y2, z2 = rotate(-radians(a), y2, z2)
    
    # line1 = ((y1, z1), (y2, z2))
    return (a, y, z), (y1, z1), (y2, z2), degrees(atan2(y2-y1,z2-z1))


def shorten(r, p):
    if type(r) == float:
        return round(r, p)
    else:
        return tuple(shorten(r1,p) for r1 in r)

ad = 1
steps = 5
module = .9
teeth = 33
pressure_angle = radians(20)

print(45, shorten(foo(radians(45), 0), 4), '\n\n')
print(40, shorten(foo(radians(40), 0), 4), '\n\n')
print(0, shorten(foo(radians(0), .79), 4), '\n\n')
print(0, shorten(foo(radians(0), .39), 4), '\n\n')

