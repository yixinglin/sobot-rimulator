"""
LS-Slam
Based on implementation of dnovischi (https://github.com/aimas-upb/slam-course-solutions)

The parent class of any types of graph
The Graph class has to be inherited and the following methods has to be implemented,
        - generate_edge_object(self, vertex1, vertex2, z, information)
        - normalize_angles(self, vertices):

Please install the scikit-sparse package (https://scikit-sparse.readthedocs.io/en/latest/overview.html)
    On Debian/Ubuntu systems, the following command should suffice:
        $ sudo apt-get install python-scipy libsuitesparse-dev
        $ pip install --user scikit-sparse
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky
from utils.math_util import normalize_angle

class Graph:

    def __init__(self):
        self.edges = []
        self.vertices = []
        self.vertex_counter = 0  # it will be set as id of any vertices.

    def add_vertex(self, vertex):
        """
        Add a vertex to the graph
        :param vertex: a Vertex object
        """
        vertex.id = self.vertex_counter
        self.vertices.append(vertex)
        self.vertex_counter += 1


    def add_edge(self, vertex1, vertex2, z, information):
        """
        Add an edge to the graph.
        :param vertex1: a Vertex object.
        :param vertex2: a Vertex object.
        :param z: vector of an actual measurement from vertex1 tp vertex2
        :param information: information matrix.
        """
        edge = self.generate_edge_object(vertex1, vertex2, z, information)
        self.edges.append(edge)

    def generate_edge_object(self, vertex1, vertex2, z, information):
        """
        To generate an Edge object w.r.t vertex1 and vertex2.
        This Edge object is a subclass of the Edge class.
        :param vertex1: the first vertex of the edge.
        :param vertex2: the second vertex of the edge.
        :param z: the actual measurement
        :param information: the information matrix, i.e the inverse of the covariance matrix
        :return: An Edge object of a subclass of the Edge class
        """
        raise NotImplementedError()

    def normalize_angles(self, vertices):
        """
        In this method you're going to normalize angles of robot's orientation of the vertices
        :param vertices: vertices of the graph
        """
        for v in vertices:
            if v.dim == 3: # a robot pose
                v.pose[2, 0] = normalize_angle(v.pose[2, 0])

    def compute_global_error(self):
        """
        Compute global error of the graph
        """
        global_err = 0
        for edge in self.edges:
            global_err += edge.calc_error() # error of an edge
        return global_err

    def graph_optimization(self, number_fix = 3,
                           damp_factor = 0.01, max_iter = 10,
                           solver = "spsolve", callback = None, epsilon = 1e-2):
        """
        Optimization of the graph
        :param number_fix: fix the estimation of the initial step
        :param damp_factor: how much you want to fix a vertex.
        :param max_iter: the maximum number of iterations
        :param callback: a callback function, callback(vertices, edges, info)
        :param epsilon: difference for determining convergence
        :return: epsilon global error after optimization
        """
        global_error = np.inf
        preError = np.inf
        for i in range(max_iter):
            """     linearize the problem   """
            start_time = time.time()
            H, b = self.linearize_constraints(self.vertices, self.edges, number_fix, damp_factor)
            linearize_time_cost = time.time() - start_time
            """     solve sparse matrix    """
            start_time = time.time()
            dx = self.solve_sparse(H, b, solver = solver)
            solve_time_cost = time.time() - start_time
            """     update vertices        """
            self.apply_dx(dx) # x = x + dx;
            global_error = self.compute_global_error()
            diff = dx.T @ dx

            if callback is not None:
                info = {"global_error": global_error, "iteration": i+1,
                        "linearize_time_cost": linearize_time_cost,
                        "solve_time_cost": solve_time_cost}
                callback(self.vertices, self.edges, info)

            dError = abs(preError - global_error)  # check convergence
            preError = global_error
            if dError < epsilon or diff < epsilon:
                self.apply_covariance(H)
                break

            if np.max(np.abs(dx)) < 0.001:
                damp_factor *= 2
            else:
                damp_factor /= 2

        return global_error

    def apply_covariance(self, H):
        H = H.tocsr()
        indices, hess_size = self.get_block_index(self.vertices)
        for id, vertex in enumerate(self.vertices):
            i1, i2 = indices[id]
            Hii = H[i1:i2, i1:i2].toarray()
            vertex.sigma = np.linalg.inv(Hii)

    def draw(self):
        """
        Visualize the graph
        """
        # draw vertices
        plt.cla()
        for v in self.vertices:
            x, y = np.squeeze(v.pose[0:2, 0])
            plt.plot(x, y, '*r')

        # draw edges
        for e in self.edges:
            x1, y1 = np.squeeze(e.vertex1.pose[0:2, 0])
            x2, y2 = np.squeeze(e.vertex2.pose[0:2, 0])
            plt.plot([x1, x2], [y1, y2], 'b')
        plt.axis('square')
        plt.show()

    def linearize_constraints(self, vertices, edges, number_fix, damp_factor):
        """
        Linearize the problem (global) i.e. compute the hessian matrix and the information vector
        :return:
                H: the hessian matrix (information matrix)
                b: information vector
        """
        indices, hess_size = self.get_block_index(vertices)  # calculate indices for each block in the hessian matrix
        b = np.zeros((hess_size, 1), dtype=np.float64)  # initialize an information vector
        Hessian_data = list()  # a list to store the block matrices (flattened)
        row_indices = list()   # a list of row indices for the corresponding values in H matrix
        col_indices = list()   # a list of column indices for the corresponding values in H matrix

        """    [ISSUE] This for-loop for calculating H and b cost time!    """
        for edge in edges:
            err, A, B = edge.linearize() # calculate error and obtain the Jacobian matrices A and B
            omega = edge.information # the information matrix
            """     Compute the block matrices and vectors   """
            bi = (err.T @ omega @ A).T
            bj = (err.T @ omega @ B).T
            Hii = A.T @ omega @ A
            Hjj = B.T @ omega @ B
            Hij = A.T @ omega @ B
            Hji = B.T @ omega @ A

            """     Compute the hessian matrix and vector    """
            i1, i2 = indices[edge.vertex1.id]
            j1, j2 = indices[edge.vertex2.id]
            b[i1:i2, :] += bi
            b[j1:j2, :] += bj

            """
            H[i1:i2, i1:i2] += Hii
            H[j1:j2, j1:j2] += Hjj
            H[i1:i2, j1:j2] += Hij
            H[j1:j2, i1:i2] += Hji
            """
            """     Calculate the indices of the block in the Hessian  """
            inxii_1, inxii_2 = self.cartesian_product(i1, i2, i1, i2)
            inxjj_1, inxjj_2 = self.cartesian_product(j1, j2, j1, j2)
            inxij_1, inxij_2 = self.cartesian_product(i1, i2, j1, j2)
            inxji_1, inxji_2 = self.cartesian_product(j1, j2, i1, i2)
            row_indices += (inxii_1 + inxjj_1 + inxij_1 + inxji_1)
            col_indices += (inxii_2 + inxjj_2 + inxij_2 + inxji_2)
            Hessian_data += (Hii.flatten().tolist() + Hjj.flatten().tolist()
                             + Hij.flatten().tolist() + Hji.flatten().tolist())

        """    Add damping factor     """
        I = [damp_factor]*number_fix
        ii = [i for i in range(number_fix)]
        Hessian_data += I
        row_indices += ii
        col_indices += ii
        """ Storing the sparse matrix with a memory efficient representation.
                Fast constructing a sparse hessian matrix (COO format). 
                Note that duplicate entries are summed together.    
                Convert CSC format for faster arithmetic and matrix vector operations.
        """
        H = coo_matrix((Hessian_data, (row_indices, col_indices)),
                       shape=(hess_size, hess_size),
                       dtype=np.float64).tocsc()

        return H, b

    def solve_sparse(self, H, b, solver = "spsolve"):
        """
        Solve the sparse linear system H @ dx = -b, where dx is unknown
        :param H: A sparse hessian matrix, and also a sparse, symmetric, positive-definite matrix.
        :param b: An information matrix.
        :param solver: solver to solve the linear system.
        :return:
                dx: the solution.
        """
        if solver == "cholesky":
            factor = cholesky(H)
            dx = factor(-b)
        elif solver == "spsolve":
            dx = spsolve(H, -b)
            dx = np.array(dx)[:, np.newaxis]
        else:
            print(" Incorrect solver name!")
            raise ValueError
        return dx

    def apply_dx(self, dx):
        """
        A apply the increment dx on all vertices of the graph
        :param dx: increment
        """
        indices, _ = self.get_block_index(self.vertices)
        for i, v in enumerate(self.vertices):
            i1, i2 = indices[i]
            v.pose += dx[i1:i2]
        self.normalize_angles(self.vertices)

    @staticmethod
    def get_block_index(vertices):
        """
        Calculate block indices in hessian matrix for each vertices of the graph
        :param vertices: A list of vertices of the graph
        :return:
            indices: indices of hessian matrix blocks (from_index, to_index) ,
                    the list order is corresponding the vertex ids, i.e. the index of a term in this list
                    represents a vertex.id where id == index
            size: an integer that represents the size of the hessian matrix
        """
        size = 0
        indices = []
        for v in vertices:
            size += v.dim
            indices.append((size - v.dim, size))
        return indices, size

    @staticmethod
    def cartesian_product(i1, i2, j1, j2):
        """
        Calculate the cartesian product in a range
        :param i1: index of row [i1, i2)
        :param i2: index of row [i1, i2)
        :param j1: index of column [j1, j2)
        :param j2: index of column [j1, j2)
        :return:
                rows, a list of indices of rows
                cols, a list of indices of columns

            i1, i2, index of row, [i1, i2)
            j1, j2, index of column [j1, j2)
            e.g.
            i1, i2 = 1, 3
            j1, j2 = 4, 5
            cartesian_product(i1, i2, j1, j2) = [1,1,2,2,3,3]
                                                [4,5,4,5,4,5]

        """
        i12 = list(range(i1, i2))
        j12 = list(range(j1, j2))
        rows = np.repeat(i12, j2 - j1).tolist()
        cols = np.tile(j12, i2 - i1).tolist()
        return rows, cols



