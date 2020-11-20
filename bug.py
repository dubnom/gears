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

    # Test a single cut
    z_step = -1
    z = z_step * z_incr
    y = pitch_radius
    angle = z / y
    z += half_tooth
    y_point, z_point = rotate(tool_angle_offset, y, z)
    angle += tool_angle_offset
    y_point -= tip_offset_y
    z_point -= tip_offset_z

    a = ad * degrees(angle)
    y = -ad * (tool_radius + y_point)
    z = z_point

    # Determine where the line ended up

    z += half_tool_tip

    y1, z1 = rotate(radians(a), y, z)
    y2, z2 = y1 + cos(tool_angle/2.) * h_total, z1 + sin(tool_angle/2.) * h_total
    y3, z3 = y1 + cos(tool_angle/2.) * h_total, z1 - sin(tool_angle/2.) * h_total

    y1, z1 = rotate(-radians(a), y1, z1)
    y2, z2 = rotate(-radians(a), y2, z2)
    y3, z3 = rotate(-radians(a), y3, z3)
    
    line1 = ((y1, z1), (y2, z2))
    line2 = ((y1, z1), (y3, z3))
    return (y1, z1), (y2, z2), (y3, z3)



ad = 1
steps = 5
module = .9
teeth = 33
pressure_angle = radians(20)

print(45, foo(radians(45), 0))
print(40, foo(radians(40), 0))
print(0, foo(radians(0), .79))
print(0, foo(radians(0), .39))

