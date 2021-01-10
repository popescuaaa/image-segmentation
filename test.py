from convertor import rgb_to_hsv, rgb_to_ycbcr
import matplotlib.image as img
from copy import deepcopy

if __name__ == '__main__':
    # tests
    image = img.imread('images/room.jpeg')
    assert image.shape == (360, 360, 3), 'failed loading proper image'

    # test converter ~ visually
    hsv_image = deepcopy(image)
    ycbcr_image = deepcopy(image)

    for r in range(len(image)):
        for c in range(len(image[r])):
            r, g, b = image[r][c]
            hsv_image[r][c] = rgb_to_hsv(r, g, b)
            ycbcr_image[r][c] = rgb_to_ycbcr(r, g, b)

    img.imsave('images/room_hsv.jpeg', hsv_image)
    img.imsave('images/room_ycbcr.jpeg', ycbcr_image)

    # assert sizes
    image = img.imread('images/room_hsv.jpeg')
    assert image.shape == (360, 360, 3), 'failed hsv saving process'

    # assert sizes
    image = img.imread('images/room_ycbcr.jpeg')
    assert image.shape == (360, 360, 3), 'failed ycbcr size saving process'
