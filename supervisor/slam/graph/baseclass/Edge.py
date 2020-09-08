class Edge:

    def __init__(self, id_vertex1, id_vertex2, z, information, list_vertices):
        """
        :param id_vertex1: id of previous vertex
        :param id_vertex2: id of current vertex
        :param z: actual measurement obtained by sensor
        :param information: information matrix
        :param list_vertices: list of vertices used to look up. a single vertex is a Vertex object.
        """
        self.list_vertices = list_vertices  # list of vertices in the graph
        self.id1 = id_vertex1  # index of previous vertex
        self.id2 = id_vertex2  # index of current vertex
        self.vertex1 = self.find_vertice_by_id(id_vertex1)
        self.vertex2 = self.find_vertice_by_id(id_vertex2)
        self.z = z  # a measurement from sensor, [r, phi] for range-bearing, [x_l, y_r] for wheel encoder
        self.information = information  # information matrix
        self.error = None  # error vector
        self.calc_error()

    def linearize(self):
        """
        Linearize the constraint.

        :return: an error vector, jacobian matrices A, B.
                e error vector of the constraint.
                A Jacobian wrt the pose vector of vertex 1.
                B Jacobian wrt the pose vector of vertex 2.
        """

        return self.linearize_constraint(self.vertex1.pose, self.vertex2.pose, self.z)

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


    def linearize_constraint(self, x1, x2, z):
        """
        Calculate error vector and jacobian matrices
        :return:
            error: an error vector
            A: a jacobian matrix, A = d(error)/d(x1)
            B: a jacobian matrix, B = d(error)/d(x2)
        """
        raise NotImplementedError()

    def calc_error_vector(self, x1, x2, z):
        """
        Calculate the error vector of the expected measurement and the actual measurement.
        :param x1: state vector of the first vertex
        :param x2: state vector of the second vertex
        :param z:  vector of actual measurement from sensors
        :return:
            error = z1 - z2
                - z1 can be calculated through the states of 2 vertices x1, x2
                - z2 is obtained by sensors
        """
        raise NotImplementedError()


    def find_vertice_by_id(self, id):
        """
        Look up a vertex with its id
        :param id:
        :return: vertex object
        """
        vertex = None
        for v in self.list_vertices:
            if v.id == id:
                vertex = v
                break
        return vertex




