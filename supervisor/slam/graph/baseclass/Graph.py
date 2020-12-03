"""
LS-Slam
Based on implementation of dnovischi (https://github.com/aimas-upb/slam-course-solutions)

The parent class of any types of graph
The Graph class has to be inherited and the following methods has to be implemented,
        - generate_edge_object(self, vertex1, vertex2, z, information)
        - normalize_angles(self, vertices):
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import warnings
from scipy.sparse import (SparseEfficiencyWarning)
warnings.simplefilter('ignore',SparseEfficiencyWarning)

def timer(function):
    """
    Used to test the efficiency of any functions
    """
    def wrapper(*args, **kw):
        time_start = time.time()
        result = function(*args, **kw)
        time_end = time.time()
        msg = '【TIMER】{0}: time: {1}'.format(function.__name__, time_end - time_start)
        print (msg)
        return result
    return wrapper


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
        raise NotImplementedError()

    def compute_global_error(self):
        """
        Compute global error of the graph
        """
        global_err = 0
        for edge in self.edges:
            global_err += edge.calc_error() # error of an edge
        return global_err

    def graph_optimization(self, animation = False, number_fix = 3, damp_factor = 0.01, max_iter = 10):
        """
        Optimization of the graph
        :param animation: decide whether animation shall be shown
        :param number_fix: fix the estimation of the initial step
        :param damp_factor: how fasten you want to fix a vertex.
        :param max_iter: the maximum number of iterations
        :return: global error after optimization
        """
        global_error = np.inf
        preError = np.inf
        for i in range(max_iter):
            """     linearize the problem   """
            H, b = self.linearize_constraints(self.vertices, self.edges, number_fix, damp_factor)
            """     solve sparse matrix    """
            dx = self.__solve_sparse(H, b)
            """     update vertices        """
            self.__apply_dx(dx) # x = x + dx;
            global_error = self.compute_global_error()
            diff = dx.T @ dx
            #print ("iter: {0}, diff: {1}, Global Error: {2}".format(i, diff, global_error))
            if animation == True:
                self.draw()

            dError = abs(preError - global_error)  # check convergence
            preError = global_error
            if dError < 0.01 or diff < 0.01:
                self.__apply_covariance(H)
                break
        return global_error

    def __apply_covariance(self, H):
        H = H.tocsr()
        indices, hess_size = self.get_block_index_(self.vertices)
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
        indices, hess_size = self.get_block_index_(vertices)  # calculate indices for each block of the hessian matrix
        b = np.zeros((hess_size, 1), dtype=np.float32)  # information vector
        Hessian_data = list()  # a list of block matrices
        row_indices = list()   # row indices of the corresponding values in H matrix
        col_indices = list()   # column indices of the corresponding values in H matrix

        for edge in edges:
            err, A, B = edge.linearize() # calculate error and jacobian matrix
            omega = edge.information
            """     Compute the block matrices and vectors   """
            b1 = (err.T @ omega @ A).T
            b2 = (err.T @ omega @ B).T
            Hii = A.T @ omega @ A
            Hjj = B.T @ omega @ B
            Hij = A.T @ omega @ B
            Hji = B.T @ omega @ A

            """     Compute the hessian matrix and vector    """
            i1, i2 = indices[edge.id1]
            j1, j2 = indices[edge.id2]
            b[i1:i2, :] += b1
            b[j1:j2, :] += b2

            """
            H[i1:i2, i1:i2] += Hii
            H[j1:j2, j1:j2] += Hjj
            H[i1:i2, j1:j2] += Hij
            H[j1:j2, i1:i2] += Hji
            """
            """     Calculate the indices of the block in the Hessian """
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
        """   Create a sparse hessian matrix and store it by a memory efficient representation.    """
        H = coo_matrix((Hessian_data, (row_indices, col_indices)), shape=(hess_size, hess_size), dtype=np.float32)
        return H, b

    def __solve_sparse(self, H, b):
        """
        Solve the linear system H @ dx = -b, where dx is unknown
        :param H: A sparse hessian matrix
        :param b: An information matrix
        :return:
                dx: the solution.
        """
        dx = spsolve(H, -b)
        dx = np.array(dx)[:, np.newaxis]
        return dx

    def __apply_dx(self, dx):
        """
        A apply the increment dx on all vertices of the graph
        :param dx: increment
        """
        indices, _ = self.get_block_index_(self.vertices)
        for i, v in enumerate(self.vertices):
            i1, i2 = indices[i]
            v.pose += dx[i1:i2]
        self.normalize_angles(self.vertices)

    @staticmethod
    def get_block_index_(vertices):
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



