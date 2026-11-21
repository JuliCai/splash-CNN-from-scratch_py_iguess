# answer for the todo 'conv2d_forward' - run this cell to define it
def conv2d_forward(x, weight, bias, C, H, W, K):
    """K x K convolution of a C x H x W image (flat CHW list) with Cout filters (flat Cout*C*K*K list).

    the whole layer is: read the patches thru im2col_layout, then one matrix multiply.
    gives back the Cout x OH x OW feature maps as one flat list (CHW order, ready for the next layer)"""
    n = C * K * K                                           # numbers in one patch = numbers in one filter
    patches = rows(gather(x, im2col_layout(C, H, W, K)), n)  # OH*OW rows, one patch per output pixel
    filters = rows(weight, n)                                # Cout rows, one filter each
    out = matmul_t(filters, patches)                         # Cout x (OH*OW), row k = feature map k
    P = len(patches)
    return [v + b for b, fmap in zip(bias, rows(out, P)) for v in fmap]
