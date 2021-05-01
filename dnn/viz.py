import imageio
from pygraphblas.demo.gviz import draw_matrix
@timing
def render_frame(prefix, Y, i):
    im = draw_matrix(Y, scale=2, labels=False)
    imageio.imwrite(prefix + str(i) + ".png", im)

