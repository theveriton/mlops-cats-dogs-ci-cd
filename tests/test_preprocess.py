from PIL import Image

from src.data import preprocess_pil_image


def test_preprocess_returns_expected_shape() -> None:
    image = Image.new("RGB", (300, 300), color=(255, 0, 0))
    tensor = preprocess_pil_image(image)
    assert tuple(tensor.shape) == (3, 224, 224)
