"""
The parent class of any types of edge
The Edge class has to be inherited and the following methods has to be implemented,
        - calc_error_vector(self, x, m, z): In this method you are going to define the error of the constraint in the graph.
        - linearize_constraint(self, x, m, z): In this method you are going to linearize the problem by calculating
            the jacobian matrices of the error w.r.t x and m
"""
class Edge:

    def __init__(self, vertex1, vertex2, z, information):
        """
        A Edge class. It is a component of a graph
        :param vertex1: previous vertex
        :param vertex2: icurrent vertex
        :param z: actual measurement obtained by sensor
        :param information: information matrix
        """
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.z = z  # a measurement from sensor, [r, phi] for range-bearing, [x_l, y_r] for wheel encoder
        self.information = information  # information matrix
        self.error = None  # error vector
        self.calc_error()

    def calc_error_vector(self, x1, x2, z):
        """
        Calculate the error vector between the expected measurement and the actual measurement.
        :param x1: state vector of the first vertex
        :param x2: state vector of the second vertex
        :param z:  vector of actual measurement from sensors
        :return:
            error = z1 - z2
                - z1 can be calculated through the states of 2 vertices x, m
                - z2 is obtained by sensors
        """
        raise NotImplementedError()

    def linearize_constraint(self, x1, x2, z):
        """
        Calculate error vector and jacobian matrices
        :return:
            error: an error vector
            A: a jacobian matrix, A = d(error)/d(x)
            B: a jacobian matrix, B = d(error)/d(m)
        """
        raise NotImplementedError()

    def calc_error(self):
        """
        Compute error of this constraint
        :return:
            error: a scalar.
                   error = e.T @ Omega @ e
        """
        # calculate the error vector from the constraint.
        self.error = self.calc_error_vector(self.vertex1.pose, self.vertex2.pose, self.z)
        return (self.error.T @ self.information @ self.error)[0, 0]


    def linearize(self):
        """
        Linearize the constraint.

        :return: an error vector, jacobian matrices A, B.
                e error vector of the constraint.
                A Jacobian wrt the pose vector of vertex 1.
                B Jacobian wrt the pose vector of vertex 2.
        """
        A, B = self.linearize_constraint(self.vertex1.pose, self.vertex2.pose, self.z)
        e = self.calc_error_vector(self.vertex1.pose, self.vertex2.pose, self.z)
        return e, A, B





