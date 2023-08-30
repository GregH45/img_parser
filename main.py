import sys
import cv2
import numpy
import matplotlib.pyplot as pyplot


class Checker:
    def __init__(self, opencv_ndarray, file_path: str) -> None:
        self.picture = opencv_ndarray
        self.img_path = file_path

    def check_size(self) -> bool:
        """Check the image size, return false if picture is not at least 512x512

        Returns:
            bool: Wether or not the size is correct
        """
        image_height, image_width, _ = self.picture.shape
        if image_height < 512 or image_width < 512:
            return False
        return True

    def check_circle(self) -> bool:
        """Check if the image can be approximated to a 8 sided polygon thats kind of a circle

        Returns:
            bool: Wether or not the picture can be approximated to a circle
        """
        gray_scale = cv2.cvtColor(self.picture, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_scale, threshold1=50, threshold2=150)
        contours, _ = cv2.findContours(
            edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, closed=True)
            epsilon = 0.04 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)

            if len(approx) == 8:  # 8-sided polygon approximation is close to a circle
                return True
            else:
                return False

    def check_pixels_out_of_circle(self) -> bool:
        """Not working // Supposed to find the biggest circle in the picture and check if there is any non transparent pixels out of it

        Returns:
            bool: test result
        """
        image = cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED)
        print(image)

        # Find the contours in image
        contours, _ = cv2.findContours(
            image[:, :, 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Iterate through the contours and find the first circle
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2000:  # find only big circles
                continue

            # Fit a circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Check if the circle contains only non-transparent pixels
            circle_mask = numpy.zeros(image.shape[:2], dtype=numpy.uint8)
            cv2.circle(circle_mask, center, radius, 255, 5)
            pixels_outside_circle = (image[:, :, 2] != 0) & (circle_mask == 0)
            # print((image[:, :, 2] != 0) & (circle_mask == 0))
            # print(numpy.all(image[:, :, 2] & circle_mask))
            if numpy.any(pixels_outside_circle):
                # image_with_circle = cv2.cvtColor(image.copy(), cv2.COLOR_BGRA2BGR)
                # cv2.circle(image_with_circle, center, radius, (0, 255, 0), 5)
                # cv2.imshow('Image with Circle', image_with_circle)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                return False
        return True

    def check_happy(self, threshold):
        """Check if the image has a certain amout of happy colors (red orange yellow)

        Args:
            threshold (int): the % of warm colors that have to be present for the image to be happy

        Returns:
            bool: test result
        """
        hsv_image = cv2.cvtColor(self.picture, cv2.COLOR_BGR2HSV)

        # HUE colors def
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

        if warm_color_percentage <= threshold:
            return False
        return True


# For this question I could also have done a bayesian classifier but it doesnt really convert so after classification, you can transform the image
# open the image and classify it
# Based on the classification result, change the image to the shape it was classified as
def img_converter(image_path: str) -> None:
    """Function that convert the image into its overall shape

    Args:
        image_path (str): path of the image
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresholded = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)

    height, width = image.shape
    black_background = numpy.zeros((height, width), dtype=numpy.uint8)

    cv2.drawContours(black_background, [largest_contour], -1, 255, thickness=cv2.FILLED)

    pyplot.figure(figsize=(10, 5))
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(image, cmap="gray")
    pyplot.title("Base image")

    pyplot.subplot(1, 2, 2)
    pyplot.imshow(black_background, cmap="gray")
    pyplot.title("Shape")
    pyplot.show()


def open_and_validate_picture(picture_path: str) -> numpy.ndarray:
    opencv_img = cv2.imread(picture_path, cv2.IMREAD_UNCHANGED)
    checker = Checker(opencv_img, picture_path)
    # print(f"image_height: { image_height }, image_w: { image_width }")

    # Check size
    if not checker.check_size():
        raise Exception(f"Picture too small, need to be 512x512.")

    if not checker.check_circle():
        raise Exception("The image is not circular")

    if not checker.check_pixels_out_of_circle():
        raise Exception("Pixels out of main circle")

    if checker.check_happy(15):
        raise Exception(
            "Image is not happy enough try more bright colors such as red, orange or yellow"
        )


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise Exception("Please provide a picture path")
    photo_path = sys.argv[1]

    try:
        # open_and_validate_picture(photo_path)
        img_converter(photo_path)
    except Exception as error:
        print(error)
        exit(-1)
