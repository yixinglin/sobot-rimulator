"""
Test the performance of the graph optimization
The used dataset_g2o can be found in https://lucacarlone.mit.edu/datasets/
"""

from supervisor.slam.graph.baseclass.Vertex import Vertex
from supervisor.slam.graph.baseclass.Edge import Edge
from supervisor.slam.graph.baseclass.Graph import Graph
from numpy.linalg import inv
from utils.math_util import normalize_angle
from supervisor.slam.graph.vetor2matrix import *
import matplotlib.pyplot as plt

class PoseVertex(Vertex):

  def __init__(self, id, pose):
    Vertex.__init__(self, pose, None)
    assert pose.shape[0] == 3
    self.id = id

  def __str__(self):
      x, y, theta = np.squeeze(self.pose)
      return "Vertex[id = {0}, pose = {1}]".format(self.id, (x, y, theta))

class PosePoseEdge(Edge):

  def __init__(self, vertex1, vertex2, z, information):
    Edge.__init__(self, vertex1, vertex2, z, information)

  def calc_error_vector(self, x1, x2, z):
      """
      Calculate the error vector
      :param x1: the previous vertex
      :param x2: the current vertex
      :param z: the actual measurement.
      :return: an error vector
      """
      Z = v2t(z)
      X1 = v2t(x1)
      X2 = v2t(x2)
      err = t2v(inv(Z) @ inv(X1) @ X2)
      return err

  def linearize_constraint(self, x1, x2, z):
    """
    Calculate the Jacobian matrices w.r.t. the current configuration.
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
      s = "PPE: id1:{0},id2: {1},z: [{2}, {3}, {4}]".format(self.id1, self.id2, x, y, yaw)
      return s

class PoseGraph(Graph):

  def __init__(self, vertices, edges):
    Graph.__init__(self)
    self.vertices = vertices
    self.edges = edges
    self.vertex_counter = len(vertices)

  def normalize_angles(self, vertices):
    for v in vertices:
        if isinstance(v, PoseVertex):
            v.pose[2, 0] = normalize_angle(v.pose[2, 0])

  def draw(self):
      """
      Visualize the graph
      """
      # draw vertices
      vertices = []
      for v in self.vertices:
        x, y = np.squeeze(v.pose[0:2, 0])
        vertices.append((x, y))

      # draw edges
      for e in self.edges:
        x1, y1 = np.squeeze(e.vertex1.pose[0:2, 0])
        x2, y2 = np.squeeze(e.vertex2.pose[0:2, 0])
        plt.plot([x1, x2], [y1, y2], 'k', linewidth=0.5)

      num_vertices = len(vertices)
      vx, vy = zip(*vertices)
      plt.plot(vx, vy, 'ok', label='Vertex ({0})'.format(num_vertices), fillstyle='none')
      plt.axis("equal")
      plt.show()

  def get_hessian(self):
    return self.linearize_constraints(self.vertices, self.edges, 0, 1)

  def plot_hessian(self):
    H, _ = self.get_hessian()
    H = np.abs(H.toarray())
    H = H / np.max(H)
    H[H > 0] = 1
    plt.matshow(H, cmap="Greys", fignum=False)
    plt.title("Hessian Matrix (Binary)")
    plt.show()

def find_vertex_by_id(id, vertices):
  for v in vertices:
    if id == v.id:
      return v
  return None


def read_data(filename):
  """
  Read the g2o file
  :param filename: file name
  :return: a PoseGraph object
  """
  f = open(filename, "r")
  vertices = list()
  edges = list()
  for line in f.readlines():
    ele = line.strip().split(" ")
    # A vertex
    if ele[0] == "VERTEX_SE2":
      # id, x, y, theta
      i, x, y, yaw = int(ele[1]), float(ele[2]), float(ele[3]), float(ele[4])
      pose = np.array([x, y, yaw], dtype=np.float32)[:, np.newaxis]
      vertex = PoseVertex(i, pose)
      vertices.append(vertex)

    # An edge
    elif ele[0] == "EDGE_SE2":
      i, j = int(ele[1]), int(ele[2]) # identifier of two vertices
      x, y, yaw = float(ele[3]), float(ele[4]), float(ele[5]) # actual measurement.

      a, b, c, d, e, g = float(ele[6]), float(ele[7]), float(ele[8]), \
                         float(ele[9]), float(ele[10]), float(ele[11])  # value of information matrix
      info = np.array([[a, b, c],
                       [b, d, e],
                       [c, e, g]], dtype=np.float32)
      z = np.array([x, y, yaw], dtype=np.float32)[:, np.newaxis]
      vi = find_vertex_by_id(i, vertices)
      vj = find_vertex_by_id(j, vertices)
      edge = PosePoseEdge(vi, vj, z, info)
      edges.append(edge)
  f.close()

  vertices.sort(key = lambda v: v.id)
  return PoseGraph(vertices, edges)

if __name__ == "__main__":
  #posegraph = read_data("input_INTEL_g2o.g2o")
  #posegraph = read_data("input_MITb_g2o.g2o")
  posegraph = read_data("input_M3500_g2o.g2o")
  #posegraph.graph_optimization(max_iter=30, damp_factor=5)
  #posegraph = read_data("input_M3500a_g2o.g2o")
  posegraph.graph_optimization(max_iter=16, damp_factor=1)
  posegraph.draw()
