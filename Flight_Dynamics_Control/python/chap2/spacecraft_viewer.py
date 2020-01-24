"""
example of drawing a box-like spacecraft in python
    - Beard & McLain, PUP, 2012
    - Update history:  
        1/8/2019 - RWB
"""
import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.Vector as Vector

class spacecraft_viewer():
    def __init__(self):
        # initialize Qt gui application and window
        self.app = pg.QtGui.QApplication([])  # initialize QT
        self.window = gl.GLViewWidget()  # initialize the view object
        self.window.setWindowTitle('Spacecraft Viewer')
        self.window.setGeometry(0, 0, 1000, 1000)  # args: upper_left_x, upper_right_y, width, height
        grid = gl.GLGridItem() # make a grid to represent the ground
        grid.scale(20, 20, 20) # set the size of the grid (distance between each line)
        self.window.addItem(grid) # add grid to viewer
        self.window.setCameraPosition(distance=200) # distance from center of plot to camera
        self.window.setBackgroundColor('k')  # set background color to black
        self.window.show()  # display configured window
        self.window.raise_() # bring window to the front
        self.plot_initialized = False # has the spacecraft been plotted yet?
        # get points that define the non-rotated, non-translated spacecraft and the mesh colors
        self.points, self.meshColors = self._get_spacecraft_points()

    ###################################
    # public functions
    def update(self, state):
        """
        Update the drawing of the spacecraft.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed to be:
            state.pn  # north position
            state.pe  # east position
            state.h   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        spacecraft_position = np.array([[state.pn], [state.pe], [-state.h]])  # NED coordinates
        # attitude of spacecraft as a rotation matrix R from body to inertial
        R = self._Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining spacecraft
        rotated_points = self._rotate_points(self.points, R)
        translated_points = self._translate_points(rotated_points, spacecraft_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self._points_to_mesh(translated_points)

        # initialize the drawing the first time update() is called
        if not self.plot_initialized:
            # initialize drawing of triangular mesh.
            self.body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                      vertexColors=self.meshColors, # defines mesh colors (Nx1)
                                      drawEdges=True,  # draw edges between mesh elements
                                      smooth=False,  # speeds up rendering
                                      computeNormals=False)  # speeds up rendering
            self.window.addItem(self.body)  # add body to plot
            self.plot_initialized = True

        # else update drawing on all other calls to update()
        else:
            # reset mesh using rotated and translated points
            self.body.setMeshData(vertexes=mesh, vertexColors=self.meshColors)

        # update the center of the camera view to the spacecraft location
        view_location = Vector(state.pe, state.pn, state.h)  # defined in ENU coordinates
        self.window.opts['center'] = view_location
        # redraw
        self.app.processEvents()

    ###################################
    # private functions
    def _rotate_points(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = R @ points
        return rotated_points

    def _translate_points(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation, np.ones([1,points.shape[1]]))
        return translated_points

    def _get_spacecraft_points(self):
        """"
            Points that define the spacecraft, and the colors of the triangular mesh
            Define the points on the aircraft following diagram in Figure C.3
        """
        #points are in NED coordinates
        points = np.array([[0.4, 0, 0],  # point 1
                           [0.2, 0.1, -0.1],  # point 2
                           [0.2, -0.1, -0.1],  # point 3
                           [0.2, -0.1, 0.1],  # point 4
                           [0.2, 0.1, 0.1],  # point 5
                           [-1, 0, 0],  # point 6
                           [0, 0.65, 0],  # point 7
                           [-0.4, 0.65, 0],  # point 8
                           [-0.4, -0.65, 0],  # point 9
                           [0, -0.65, 0],  # point 10
                           [-0.75, 0.35, 0],  # point 11
                           [-1, 0.35, 0],  # point 12
                           [-1, -0.35, 0],  # point 13
                           [-0.75, -0.35, 0],  # point 14
                           [-0.75, 0, 0],  # point 15
                           [-1, 0, 0.4],  # point 16
                          ]).T
        # scale points for better rendering
        scale = 10
        points = scale * points

        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((12, 3, 4), dtype=np.float32)
        meshColors[0] = yellow  # nose
        meshColors[1] = yellow  
        meshColors[2] = yellow  
        meshColors[3] = yellow 
        meshColors[4] = blue  # fuselage
        meshColors[5] = blue  
        meshColors[6] = blue  
        meshColors[7] = blue  
        meshColors[8] = green  # wing
        meshColors[9] = green  
        meshColors[10] = green  # back wing
        meshColors[11] = green  
        return points, meshColors

    def _points_to_mesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
          (a rectangle requires two triangular mesh faces)
        """
        points=points.T
        mesh = np.array([[points[0], points[1], points[2]],  #nose
                         [points[0], points[2], points[3]],  
                         [points[0], points[3], points[4]],  
                         [points[0], points[1], points[4]],  
                         [points[1], points[2], points[5]],  # fuselage
                         [points[2], points[3], points[5]], 
                         [points[3], points[4], points[5]], 
                         [points[1], points[4], points[5]],
                         [points[9], points[6], points[7]],  # wing 
                         [points[9], points[7], points[8]], 
                         [points[13], points[10], points[11]],  # back wing
                         [points[13], points[11], points[12]],  
                         ])
        return mesh

    def _Euler2Rotation(self, phi, theta, psi):
        """
        Converts euler angles to rotation matrix (R_b^i, i.e., body to inertial)
        """
        # only call sin and cos once for each angle to speed up rendering
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        R_roll = np.array([[1, 0, 0],
                           [0, c_phi, s_phi],
                           [0, -s_phi, c_phi]])
        R_pitch = np.array([[c_theta, 0, -s_theta],
                            [0, 1, 0],
                            [s_theta, 0, c_theta]])
        R_yaw = np.array([[c_psi, s_psi, 0],
                          [-s_psi, c_psi, 0],
                          [0, 0, 1]])
        R = R_roll @ R_pitch @ R_yaw  # inertial to body (Equation 2.4 in book)
        return R.T  # transpose to return body to inertial

