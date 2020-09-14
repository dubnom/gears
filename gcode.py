class Gcode():
    """Simple class to make gcode file creation easier."""


    def __init__(self):
        self.gcode = []

    def move(self, a=None, x=None, y=None, z=None):
        """Record a high-speed move operation"""
        self.gcode.append("G0" +
                          (' A%g'%a if a is not None else '') +
                          (' X%g'%x if x is not None else '') +
                          (' Y%g'%y if y is not None else '') +
                          (' Z%g'%z if z is not None else ''))

    def cut(self, a=None, x=None, y=None, z=None):
        """Record a cutting speed linear operation"""
        self.gcode.append("G1" +
                          (' A%g'%a if a is not None else '') +
                          (' X%g'%x if x is not None else '') +
                          (' Y%g'%y if y is not None else '') +
                          (' Z%g'%z if z is not None else ''))

    def comment(self, line=None):
        """Record comments"""
        self.gcode.append("(%s)" % line if line else '')

    def append(self, line):
        """Add to the end."""
        self.gcode.append(line)

    def output(self):
        """Create a new-line separated string of the gcode."""
        return '\n'.join(self.gcode)
