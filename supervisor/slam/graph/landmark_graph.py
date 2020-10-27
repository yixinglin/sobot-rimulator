import matplotlib.pyplot as plt
from numpy.linalg import inv
from random import sample
from supervisor.slam.graph.baseclass.Vertex import Vertex
from supervisor.slam.graph.baseclass.Edge import Edge
from supervisor.slam.graph.baseclass.Graph import Graph
from supervisor.slam.graph.vetor2matrix import *
from utils.math_util import normalize_angle

"""       Define Vertex Classes        
Vertex
    - PoseVertex
    - LandmarkVertex
"""
class PoseVertex(Vertex):

    def __init__(self, pose, sigma, observation = None):
        """
        :param pose: robot pose. [x, y, yaw].T
        :param sigma: covariance matrix
        :param observation: observation of landmarks
        """
        Vertex.__init__(self, pose, sigma, observation)
        assert pose.shape[0] == 3
        assert sigma.shape == (3, 3)

    def __str__(self):
        x, y, yaw = np.squeeze(self.pose)
        return "Pose[id = {0}, pose = {1}]".format(self.id, (x, y, yaw))

    def __sub__(self, other):
        X2 = v2t(self.pose)  # current
        X1 = v2t(other.pose)    # previous
        Z = inv(X1) @ X2
        z = t2v(Z)
        return Vertex(z, None)

class LandmarkVertex(Vertex):

    def __init__(self, pose, sigma, landmark_id):
        """
        :param pose: landmark position. [x, y].T
        :param sigma: covariance matrix
        """
        Vertex.__init__(self, pose, sigma, None)
        assert pose.shape[0] == 2
        assert sigma.shape == (2, 2)
        self.landmark_id = landmark_id


    def __str__(self):
        x, y = np.squeeze(self.pose)
        return "Landmark[id = {0}, pose = {1}]".format(self.id, (x, y))

    def __sub__(self, other):
        assert isinstance(other, PoseVertex)
        return Vertex(self.pose - other.pose[0:2, :], None)

"""      Define Edge Classes       
Edge
    - PosePoseEdge
    - PoseLandmarkEdge

"""

class PosePoseEdge(Edge):

    def __init__(self, id_vertex1, id_vertex2, z, information, list_vertices):
        """
        :param id_vertex1: id of previous vertex
        :param id_vertex2: id of current vertex
        :param z: actual measurement obtained by sensor
        :param information: information matrix
        :param list_vertices: list of vertices used to look up. a single vertex is a Vertex object.
        """

        Edge.__init__(self, id_vertex1, id_vertex2, z, information, list_vertices)
        assert isinstance(self.vertex1, PoseVertex)
        assert isinstance(self.vertex2, PoseVertex)
        assert z.shape == (3, 1)
        assert information.shape == (3, 3)


    def calc_error_vector(self, x1, x2, z):
        """
        Calculate the error vector.

        :param x1: 3x1 vector of previous state.
        :param x2: 3x1 vector of current state.
        :param z:  3x1 vector of measurement from x1 to x2.
        :return: an error vector, jacobian matrices A, B.
        """
        # calculate homogeneous matrices.
        Z = v2t(z) # homogeneous matrix of vector z
        X1 = v2t(x1) # homogeneous matrix of vector x1
        X2 = v2t(x2) # # homogeneous matrix of vector x2
        e12 = t2v(inv(Z) @ inv(X1) @ X2)
        return e12

    @staticmethod
    def encode_measurement(pose, odom_pose, covariance):
        """
        Calculate the measurement vector that represents how the robot move from previous pose to the expected pose.
        The jacobian matrix of the error function is calculated by this model.
        The vector (x, y, theta) is obtained from a rotation matrix R(theta) and a translational vector t = [x, y].T
        :param pose: previous pose of the robot
        :param odom_pose: expected pose of the next step calculated by the motion model
        :param covariance: covariance matrix
        return:
                meas: a converted measurement related to the previous pose, which will be set to an edge as a measurement vector z.
                info: an information matrix
        """
        M = inv(v2t(pose)) @ v2t(odom_pose)  # 3x3 transformation matrix from x_meas to vertex1.pose
        info = inv(covariance) # information matrix
        meas = t2v(M)  # a converted measurement.
        return meas, info

    def linearize_constraint(self, x1, x2, z):
        """
        Linearize the pose-pose constraint.

        :param x1: 3x1 vector of previous pose.
        :param x2: 3x1 vector of current pose.
        :param z:  3x1 vector of measurement from x1 to x2.
        :return: an error vector, jacobian matrices A, B.
                e 3x1 error of the constraint.
                A 3x3 Jacobian wrt x1.
                B 3x3 Jacobian wrt x2.
        """
        c = cos(x1[2, 0] + z[2, 0])
        s = sin(x1[2, 0] + z[2, 0])
        tx = x2[0, 0] - x1[0, 0]
        ty = x2[1, 0] - x1[1, 0]

        A = np.array([[-c, -s, -tx*s+ty*c],
                      [s, -c, -tx*c-ty*s],
                      [0, 0, -1]])
        B = np.array([[c, s, 0],
                     [-s, c, 0],
                     [0, 0, 1]])

        e = self.calc_error_vector(x1, x2, z)
        return e, A, B

    def __str__(self):
        x, y, yaw = np.squeeze(self.z)
        s = "PPE: id1:{0},id2: {1},z: [{2}, {3}, {4}]".format(self.id1, self.id2, x, y, yaw)
        return s

class PoseLandmarkEdge(Edge):

    def __init__(self, id_vertex1, id_vertex2, z, information, list_vertices):
        Edge.__init__(self, id_vertex1, id_vertex2, z, information, list_vertices)
        assert isinstance(self.vertex1, PoseVertex)
        assert isinstance(self.vertex2, LandmarkVertex)
        assert z.shape == (2, 1)
        assert information.shape == (2, 2)

    @staticmethod
    def encode_measurement(observation, covariance):
        """
        Encode the raw_measurement of landmark observation.
        The vector meas_xy represents how the robot sees a landmark. This vector is a cartesian coordinate (x, y).
        The Jacobian matrix of the error function is calculated by this model.
        :param observation: The raw measurement from range bearing sensor (distance, angle)
        :param covariance: The covariance matrix of the measurement
        return
                meas_xy: a encoded measurement vector (x, y) related to the robot, which will be set to an edge as a measurement vector z.
                info: an information matrix
        """
        info = inv(covariance)
        meas_xy = np.array([[observation[0] * cos(observation[1])],
                            [observation[0] * sin(observation[1])]])
        return meas_xy, info

    def calc_error_vector(self, x1, x2, z):

        X1 = v2t(x1) # homogeneous matrix of vector x1
        R1 = X1[0:2, 0:2]  # rotation matrix
        e12 = R1.T @ (x2 - x1[0:2, :]) - z
        return e12

    def linearize_constraint(self, x, lm, z):
        """
        Compute the error of a pose-landmark constraint.
        :param x1: 3x1 vector (x,y,theta) of the robot pose.
        :param lm: 2x1 vector (x,y) of the landmark.
        :param z: 2x1 vector (x,y) of the measurement, the position of the landmark in.
                the coordinate frame of the robot given by the vector x.
        :return:
            e 2x1 error of the constraint.
            A 2x3 Jacobian wrt x.
            B 2x2 Jacobian wrt lm.
        """
        theta_i = x[2, 0]
        xi, yi = x[0, 0], x[1, 0]
        xl, yl = lm[0, 0], lm[1, 0]
        # Compute eij partial derivative with respect to pose x
        c = cos(theta_i)
        s = sin(theta_i)
        A = np.array([[-c, -s, -(xl - xi) * s + (yl - yi) * c],
                      [s, -c, -(xl - xi) * c - (yl - yi) * s]])
        B = np.array([[c, s],
                      [-s, c]])

        e = self.calc_error_vector(x, lm, z)
        return e, A, B

    def __str__(self):
        x, y, yaw = np.squeeze(self.z)
        s = "PLE: id1:{0},id2: {1},z: [{2}, {3}]".format(self.id1, self.id2, x, y)
        return s


"""   Define Graph Class
Graph
    -LMGraph 

"""
class LMGraph(Graph):

    def __init__(self):
        """
        Posegraph to estimate only robot's poses
        :param vertices: list of vertices. a single vertex is a pose of robot
        :param edges: list of edges. a single edge is a constraint
                            An edge can be an object of
                            1. PoseLandmarkEdge, (PoseVertex - PoseVertex)
                            2. LandmarkVertex, (PoseVertex - LandmarkVertex)
        """
        Graph.__init__(self)


    def generate_edge_object(self, vertex1, vertex2, z, information):
        """
        Add an edge to the graph.
        :param vertex1: a Vertex object.
        :param vertex2: a Vertex object.
        :param z: vector of an actual measurement.
        :param information: information matrix.
        """

        if isinstance(vertex1, PoseVertex) and isinstance(vertex2, LandmarkVertex):
            # edge is an Edge object
            edge = PoseLandmarkEdge(vertex1.id, vertex2.id, z, information, self.vertices)
        elif isinstance(vertex1, PoseVertex) and isinstance(vertex2, PoseVertex):
            edge = PosePoseEdge(vertex1.id, vertex2.id, z, information, self.vertices)
        else:
            raise ValueError()

        return edge

    def count_vertices(self):
        """
        :return:
                pose: number of PoseVertex objects
                landmarks number of LandmarkVertex objects
        """
        pose = 0
        landmarks = 0
        for v in self.vertices:
            if isinstance(v, PoseVertex):
                pose += 1
            elif isinstance(v, LandmarkVertex):
                landmarks += 1
            else:
                raise RuntimeError()
        return pose, landmarks

    def normalize_angles(self, vertices):
        for v in vertices:
            if isinstance(v, PoseVertex):
                # v.pose[2, 0] = atan2(sin(v.pose[2, 0]), cos(v.pose[2, 0]))
                v.pose[2, 0] = normalize_angle(v.pose[2, 0])

    def get_last_pose_vertex(self):
        v_pose = None
        for v in reversed(self.vertices):
            if isinstance(v, PoseVertex):
                v_pose = v
                break
        return v_pose

    def get_estimated_pose_vertices(self):
        poses = []
        for v in self.vertices:
            if isinstance(v, PoseVertex):
                poses.append(v)
        return poses

    def get_estimated_landmark_vertices(self):
        poses = []
        for v in self.vertices:
            if isinstance(v, LandmarkVertex):
                poses.append(v)
        return poses

    def draw(self):
        """
        Visualize the graph
        """
        # draw vertices
        plt.cla()
        landmarks = []
        vertices = []
        for v in self.vertices:
            if isinstance(v, LandmarkVertex):
                x, y = np.squeeze(v.pose[0:2, 0])
                landmarks.append((x, y))
            if isinstance(v, PoseVertex):
                x, y = np.squeeze(v.pose[0:2, 0])
                vertices.append((x, y))

        # draw edges
        for e in self.edges:
            x1, y1 = np.squeeze(e.vertex1.pose[0:2, 0])
            x2, y2 = np.squeeze(e.vertex2.pose[0:2, 0])
            if isinstance(e, PoseLandmarkEdge):
                plt.plot([x1, x2], [y1, y2], 'y', linewidth=0.3)
            if isinstance(e, PosePoseEdge):
                plt.plot([x1, x2], [y1, y2], 'k')

        num_landmarks = len(landmarks)
        if num_landmarks  > 0:
            lmx, lmy = zip(*landmarks)
            plt.plot(lmx, lmy, 'xb', label = 'Landmark ({0})'.format(num_landmarks))
        num_vertices = len(vertices)
        if num_vertices > 0:
            k = min(1000, num_vertices)
            vertices = sample(vertices, k)
            vx, vy = zip(*vertices)
            plt.plot(vx, vy, '*r', label='Vertex ({0})'.format(num_vertices))

        plt.title("Graph")
        plt.legend()
        plt.axis('square')
        plt.show()