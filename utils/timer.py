import time

def timer(function):
  """
  Used to test the efficiency of any functions
  """

  def wrapper(*args, **kw):
    time_start = time.time()
    result = function(*args, **kw)
    time_end = time.time()
    msg = '【TIMER】{0}: time: {1}'.format(function.__name__, time_end - time_start)
    print(msg)
    return result

  return wrapper