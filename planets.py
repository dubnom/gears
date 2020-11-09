from math import *

year = 365.24
planets = [
    ("Mercury",     87.97 / year ),
    ("Venus",       224.7 / year ),
    ("Earth",       365.24 / year ),
    ("Mars",        686.98 / year ),
    #("Mars",        1.0, 1.88),
    ("Jupiter",     4332.59 / year ),
    ("Saturn",      11000 / year ),
]

# Module = diameter / teeth
totalDiameter = 6 * 25.4
minModule = .7

print("{:10} {:>3} {:>3} {:>8}  {:>6} {:>6} {:>6} {:>6} {:>6}".format('Planet', 'x', 'y','mod', 'err','x Diam', 'y Diam', 'x Rad', 'y rad'), '\n')
for planet in planets:
    best = ()
    prevError = 1000
    for y in range(20, 200):
        for x in range(20, 200):
            ratio = x/y
            error = abs((planet[1] - ratio) / planet[1])
            if error <= prevError:
                module = totalDiameter / (x + y)
                if module < minModule:
                    continue
                prevError = error
                best = (x,y,module,round(100*error,2), (x+2)*module, (y+2)*module, (x+2)*module/2, (y+2)*module/2)
#    print(planet[0], best)

    print("{:10} {:3d} {:3d} {:8.5f} {:6.2f}% {:6.2f} {:6.2f} {:6.2f} {:6.2f}".format(planet[0], *best))
