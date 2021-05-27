# flap_field_lines
FLAP package that handles magnetic field lines in fusion devices. It reads, stores and manipulates them using FLAP functionalities and projects them to 2D image coordinates, so they can be displayed on various fast camera images.

Precalculated parameters for various W7X viewports can be found in the 'views2.txt' file. To use them, create an instance of the  ImageProjector class the following way:

view = ImageProjector.from_file(<viewport>, <reference date>, <camera type>, <path to file>)
e.g.
view = ImageProjector.from_file('W7X-AEQ31', '20160218', 'edicam', 'views2.txt')

For the projection itself do the following (input should be a 2 or 3d array, with the first dimension being 3):

projected_points = view.calc_pixel_coord(points)