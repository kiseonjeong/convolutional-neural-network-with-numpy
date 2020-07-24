import numpy as np

def im2col(input_data, filter_h, filter_w, stride, pad):
    """
    (function) im2col
    -----------------
    - Convert the shape of the data from image to column

    Parameter
    ---------
    - input_data : input data\n
    - filter_h : filter height\n
    - filter_w : filter width\n
    - stride : sliding interval\n
    - pad : boundary padding length\n

    Return
    ------
    - reshaped column data\n
    """
    # Calculate result resolution information
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # Do padding on the input data
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # Generate the column data
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col

# Convert column to image
def col2im(col, input_shape, filter_h, filter_w, stride, pad):
    """
    (function) col2im
    -----------------
    - Convert the shape of the data from column to image
    
    Parameter
    ---------
    - col : column data\n
    - input_shape : original shape on the input\n
    - filter_h : filter height\n
    - filter_w : filter width\n
    - stride : sliding interval\n
    - pad : boundary padding length\n

    Return
    ------
    - reshaped image data\n
    """
    # Calculate result resolution information
    N, C, H, W = input_shape.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # Generate the image data
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]