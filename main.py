import sys
import cv2
import numpy

# We want users to upload a badge: an avatar within a circle.
# Create a function taking a png as input and verifying that:
# - Size = 512x512
# - The only non transparent pixels are within a circle
# - That the colors is the badge give a "happy" feeling
# Additionally, you can create a parallel function that convert the given image (of any format) into the specified object.


class Checker:
    def __init__(self, opencv_ndarray) -> None:
        self.picture = opencv_ndarray

    def check_size(self) -> bool:
        image_height, image_width, _ = self.picture.shape
        if image_height < 512 or image_width < 512:
            return False
        return True

    def check_circle(self) -> bool:
        image_height, image_width, _ = self.picture.shape
        circle_canvas = numpy.zeros((image_height, image_width), dtype=numpy.uint8)
        circle_center = (image_width // 2, image_height // 2)
        circle_radius = min(image_width, image_height) // 2

        cv2.circle(circle_canvas, circle_center, circle_radius, 255, thickness=-1)
        circle_mask = circle_canvas == 255
        circle_mask = numpy.repeat(circle_mask[:, :, numpy.newaxis], 3, axis=2)
        pixels_outside_circle = numpy.logical_and(~circle_mask, self.picture == 0)

        if numpy.any(pixels_outside_circle):
            return False
        return True

    def check_happy(self, threshold):
        hsv_image = cv2.cvtColor(self.picture, cv2.COLOR_BGR2HSV)

        lower_red = numpy.array([0, 100, 100])
        upper_red = numpy.array([30, 255, 255])

        lower_orange = numpy.array([15, 100, 100])
        upper_orange = numpy.array([45, 255, 255])

        lower_yellow = numpy.array([30, 100, 100])
        upper_yellow = numpy.array([60, 255, 255])

        mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
        mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
        mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        combined_mask = mask_red + mask_orange + mask_yellow

        total_pixels = self.picture.shape[0] * self.picture.shape[1]
        warm_color_pixels = numpy.count_nonzero(combined_mask)
        warm_color_percentage = (warm_color_pixels / total_pixels) * 100

        print(f"Warm %  : {warm_color_percentage}")

        if warm_color_percentage <= threshold:
            return False
        return True


def open_and_validate_picture(picture_path: str) -> numpy.ndarray:
    opencv_img = cv2.imread(picture_path, cv2.IMREAD_COLOR)
    checker = Checker(opencv_img)
    # print(f"image_height: { image_height }, image_w: { image_width }")

    # Check size
    if not checker.check_size():
        raise Exception(
            f"Picture too small, need to be 512x512. Current WxH: { image_width }x{ image_height }"
        )

    if not checker.check_circle():
        raise Exception("The image is not circular")

    if checker.check_happy(15):
        raise Exception(
            "Image is not happy enough try more bright colors such as red, orange or yellow"
        )


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise Exception("Please provide a picture path")
    photo_path = sys.argv[1]

    try:
        open_and_validate_picture(photo_path)
    except Exception as error:
        print(error)
        exit(-1)
