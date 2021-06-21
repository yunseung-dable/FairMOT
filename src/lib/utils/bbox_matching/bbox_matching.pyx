# matching_bbox.pyx

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float_t DTYPE_t

def matching_conf(
        np.ndarray[DTYPE_t, ndim=2] head_boxes,
        np.ndarray[DTYPE_t, ndim=2] visible_boxes):

    cdef unsigned int hlen = head_boxes.shape[0]
    cdef unsigned int vlen = visible_boxes.shape[0]
    cdef unsigned int h,v

    cdef np.ndarray[DTYPE_t, ndim=2] conf_mat = np.zeros((hlen, vlen), dtype=DTYPE)
    cdef DTYPE_t iw, ih, head_area
    
    #boxes =  (left top right bottom)
    for h in range(hlen):
        head_area = (
            (head_boxes[h, 2] - head_boxes[h, 0] + 1) *
            (head_boxes[h, 3] - head_boxes[h, 1] + 1)
        )
        for v in range(vlen):
            iw = (
                min(head_boxes[h, 2], visible_boxes[v, 2]) -
                max(head_boxes[h, 0], visible_boxes[v, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(head_boxes[h, 3], visible_boxes[v, 3]) -
                    max(head_boxes[h, 1], visible_boxes[v, 1]) + 1
                )
                if ih > 0:
                    conf_mat[h,v] = iw * ih / head_area
    
    return conf_mat