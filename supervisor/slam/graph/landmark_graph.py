import matplotlib.pyplot as plt
from numpy.linalg import inv
from random import sample
from supervisor.slam.graph.baseclass.Vertex import Vertex
from supervisor.slam.graph.baseclass.Edge import Edge
from supervisor.slam.graph.baseclass.Graph import Graph
from supervisor.slam.graph.vetor2matrix import *

"""       Define Vertex Classes        
Extended from the base-classs Vertex 
    - PoseVertex
    - LandmarkVertex
"""
class PoseVertex(Vertex):

    def __init__(self, pose, sigma):
        """
        :param pose: robot pose. [x, y, yaw].T
        :param sigma: covariance matrix
        :param observation: observation of landmarks
        """
        Vertex.__init__(self, pose, sigma)
        assert pose.shape[0] == 3
        assert sigma.shape == (3, 3)

    def __str__(self):
        x, y, yaw = np.squeeze(self.pose)
        return "Pose[id = {0}, pose = {1}]".format(self.id, (x, y, yaw))

class LandmarkVertex(Vertex):

    def __init__(self, pose, sigma, landmark_id):
        """
        :param pose: landmark position. [x, y].T
        :param sigma: covariance matrix
        :param landmark_id: the landmark identifier in the map
        """
        Vertex.__init__(self, pose, sigma)
        assert pose.shape[0] == 2
        assert sigma.shape == (2, 2)
        self.landmark_id = landmark_id

    def __str__(self):
        x, y = np.squeeze(self.pose)
        return "Landmark[id = {0}, pose = {1}]".format(self.id, (x, y))

"""      Define Edge Classes       
Extended from the base-classs Edge
    - PosePoseEdge
    - PoseLandmarkEdge

"""

class PosePoseEdge(Edge):

    def __init__(self, vertex1, vertex2, z, information):
        """
        The constraint is based on motion commands used in a time interval.
        :param vertex1: id of previous vertex
        :param vertex2: id of current vertex
        :param z: actual measurement obtained by sensor
        :param information: information matrix
        """
        Edge.__init__(self, vertex1, vertex2, z, information)
        assert isinstance(self.vertex1, PoseVertex)
        assert isinstance(self.vertex2, PoseVertex)
        assert z.shape == (3, 1)
        assert information.shape == (3, 3)

    def calc_error_vector(self, x1, x2, z):
        """
        Calculate the error vector.
        :param x1: 3x1 vector of previous state.
        :param x2: 3x1 vector of current state.
        :param z:  3x1 vector of measurement from x to m.
        :return:
                e12: an error vector, jacobian matrices A, B.
        """
        # calculate homogeneous matrices.
        Z = v2t(z) # homogeneous matrix of vector z
        X1 = v2t(x1) # homogeneous matrix of vector x
        X2 = v2t(x2) # # homogeneous matrix of vector m
        e12 = t2v(inv(Z) @ inv(X1) @ X2)
        return e12

    @staticmethod
    def encode_measurement(pose, odom_pose, covariance):
        """
        Calculate the measurement vector that represents how the robot move from previous pose to the expected pose.
        The Jacobian matrix of the error function is calculated by this model.
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
        Linearize the pose-pose constraint of edge.
        :param x1: 3x1 vector of previous pose.
        :param x2: 3x1 vector of current pose.
        :param z:  3x1 vector of measurement from x to m.
        :return: an error vector, jacobian matrices A, B.
                A 3x3 Jacobian wrt x.
                B 3x3 Jacobian wrt m.
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
        return A, B

    def __str__(self):
        x, y, yaw = np.squeeze(self.z)
        s = "PPE: id1:{0},id2: {1},z: [{2}, {3}, {4}]".format(self.vertex1.id, self.vertex2.id, x, y, yaw)
        return s

class PoseLandmarkEdge(Edge):

    def __init__(self, vertex1, vertex2, z, information):
        """
        The constraint is based on an x-y sensor model
        """
        Edge.__init__(self, vertex1, vertex2, z, information)
        assert isinstance(self.vertex1, PoseVertex)
        assert isinstance(self.vertex2, LandmarkVertex)
        assert z.shape == (2, 1)
        assert information.shape == (2, 2)

    @staticmethod
    def encode_measurement(observation, covariance):
        """
        Encode the raw_measurement of range-bearing observation.
        The vector z_xy represents how the robot sees a landmark. This vector is a cartesian coordinate (x, y).
        The Jacobian matrix of the error function is calculated by this model.
        :param observation: The raw measurement from range bearing sensor (distance, angle)
        :param covariance: The covariance matrix of the measurement
        return
                z_xy: a encoded measurement vector (x, y) related to the robot, which will be set to an edge as a measurement vector z.
                info: an information matrix
        """
        info = inv(covariance)
        z_xy = np.array([[observation[0] * cos(observation[1])],
                            [observation[0] * sin(observation[1])]])
        return z_xy, info

    def calc_error_vector(self, x, m, z):
        """
        Compute the error of a pose-landmark constraint.
        :param x: 3x1 vector (x,y,theta) of the robot pose.
        :param m:  2x1 vector (x,y) of the landmark position.
        :param z:  2x1 vector (x,y) of the actual measurement.
        :return:
               e 2x1 error of the constraint.
        """
        X = v2t(x) # homogeneous matrix of pose x
        R = X[0:2, 0:2]  # rotation matrix
        e = R.T @ (m - x[0:2, :]) - z
        return e

    def linearize_constraint(self, x, m, z):
        """
        Compute the error of a pose-landmark constraint.
        :param x: 3x1 vector (x,y,theta) of the robot pose.
        :param m: 2x1 vector (x,y) of the landmark.
        :param z: 2x1 vector (x,y) of the measurement, the position of the landmark in.
                the coordinate frame of the robot given by the vector x.
        :return:
            A 2x3 Jacobian w.r.t. x.
            B 2x2 Jacobian w.r.t. m.
        """
        theta_i = x[2, 0]
        delta_x = m[0, 0] - x[0, 0]
        delta_y = m[1, 0] - x[1, 0]
        # Compute  partial derivative of e with respect to pose x and landmark m
        c = cos(theta_i)
        s = sin(theta_i)
        A = np.array([[-c, -s, -delta_x * s + delta_y * c],
                      [s, -c, -delta_x * c - delta_y * s]])
        B = np.array([[c, s],
                      [-s, c]])
        return A, B

    def __str__(self):
        x, y, yaw = np.squeeze(self.z)
        s = "PLE: id1:{0},id2: {1},z: [{2}, {3}]".format(self.vertex1.id, self.vertex2.id, x, y)
        return s


"""   Define Graph Class
Extended from the base-classs Graph
    -LMGraph 

"""
class LMGraph(Graph):

    def __init__(self):
        """
        Landmark-Graph to estimate robot's poses and landmarks
        :param vertices: list of vertices. a single vertex is a pose of robot or a landmark
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
            edge = PoseLandmarkEdge(vertex1, vertex2, z, information)
        elif isinstance(vertex1, PoseVertex) and isinstance(vertex2, PoseVertex):
            edge = PosePoseEdge(vertex1, vertex2, z, information)
        else:
            raise ValueError()
        return edge

    def count_vertices(self):
        """
        Count the number of vertices
        :return:
                pose: number of PoseVertex objects
                landmarks: number of LandmarkVertex objects
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

    def get_last_pose_vertex(self):
        """
            Return the last pose vertex.
        """
        v_pose = None
        for v in reversed(self.vertices):
            if isinstance(v, PoseVertex):
                v_pose = v
                break
        return v_pose

    def get_estimated_pose_vertices(self):
        """
            Return a list of vertices that represent poses
        """
        poses = []
        for v in self.vertices:
            if isinstance(v, PoseVertex):
                poses.append(v)
        return poses

    def get_estimated_landmark_vertices(self):
        """
            Return a list of vertices that represent landmarks
        """
        poses = []
        for v in self.vertices:
            if isinstance(v, LandmarkVertex):
                poses.append(v)
        return poses

    def get_hessian(self):
        """
        Return the hessian matrix H and the information vector b
        :return:  H, b
            H: the hessian matrix
            b: the information vector
        """
        return self.linearize_constraints(self.vertices, self.edges, 0, 1)

    def draw(self):
        """
        Visualize the graph
        """
        # draw vertices
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
